#include "processing/FrameProcessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace video_filter {
namespace {

constexpr double defaultFrameIntervalSeconds = 1.0 / 30.0;
constexpr double chromaNeutralValue = 128.0;
constexpr double minimumChromaSampleRatio = 0.02;

PercentileStats clampStats(const PercentileStats& stats) {
    PercentileStats clamped(stats);
    clamped.p5 = std::clamp(clamped.p5, 0.0, 254.0);
    clamped.p95 = std::clamp(clamped.p95, clamped.p5 + 1.0, 255.0);
    clamped.p50 = std::clamp(clamped.p50, clamped.p5, clamped.p95);
    return clamped;
}

ToneMapState clampToneMapState(const ToneMapState& state, double minimumRange) {
    ToneMapState clamped(state);
    clamped.low = std::clamp(clamped.low, 0.0, 254.0);
    clamped.high = std::clamp(clamped.high, clamped.low + 1.0, 255.0);
    if ((clamped.high - clamped.low) < minimumRange) {
        const double center = (clamped.low + clamped.high) * 0.5;
        const double halfRange = minimumRange * 0.5;
        clamped.low = std::max(0.0, center - halfRange);
        clamped.high = std::min(255.0, center + halfRange);
        if ((clamped.high - clamped.low) < minimumRange) {
            clamped.low = std::max(0.0, clamped.high - minimumRange);
            clamped.high = std::min(255.0, clamped.low + minimumRange);
        }
    }
    clamped.gamma = std::max(0.0, clamped.gamma);
    return clamped;
}

}

FrameMetadata::FrameMetadata() {
    ptsNs = -1;
    durationNs = 0;
}

FrameMetadata::FrameMetadata(std::int64_t ptsNsValue, std::int64_t durationNsValue) {
    ptsNs = ptsNsValue;
    durationNs = durationNsValue;
}

PercentileStats::PercentileStats() {
    p5 = 0.0;
    p50 = 127.0;
    p95 = 255.0;
}

PercentileStats::PercentileStats(double p5Value, double p50Value, double p95Value) {
    p5 = p5Value;
    p50 = p50Value;
    p95 = p95Value;
}

FrameProcessorConfiguration::FrameProcessorConfiguration() {
    emaTimeConstantSeconds = 0.8;
    targetMidtone = 0.52;
    minPercentileRange = 56.0;

    chromaCorrectionStrength = 0.25;
    chromaBiasTimeConstantSeconds = 1.10;
    maxChromaCorrection = 6.0;
    chromaMidtoneMargin = 0.18;

    claheClipLimit = 1.8;
    claheGridSize = cv::Size(8, 8);
    claheStrength = 0.35;

    gammaStrength = 0.38;
    toneMapTimeConstantSeconds = 1.30;
    toneMapDeadbandLevels = 3.0;
    toneMapGammaDeadband = 0.06;
    toneMapMaxLevelStepPerSecond = 7.0;
    toneMapMaxGammaStepPerSecond = 0.06;
    minGamma = 0.88;
    maxGamma = 1.10;
    outputBlackFloor = 12.0;
    outputWhiteCeiling = 250.0;

    sharpenAmount = 0.18;
    sharpenThreshold = 7;
    sharpenSigma = 1.0;
}

ToneMapState::ToneMapState() {
    low = 0.0;
    high = 255.0;
    gamma = 1.0;
}

ToneMapState::ToneMapState(double lowValue, double highValue, double gammaValue) {
    low = lowValue;
    high = highValue;
    gamma = gammaValue;
}

ChromaCorrectionState::ChromaCorrectionState() {
    crOffset = 0.0;
    cbOffset = 0.0;
}

ChromaCorrectionState::ChromaCorrectionState(double crOffsetValue, double cbOffsetValue) {
    crOffset = crOffsetValue;
    cbOffset = cbOffsetValue;
}

FrameProcessor::FrameProcessor(FrameProcessorConfiguration configurationValue) {
    configuration = std::move(configurationValue);
    initializePersistentResources();
    resetTemporalState();
}

void FrameProcessor::processFrame(const cv::Mat& inputBgr, cv::Mat& outputBgr, const FrameMetadata& frameMetadata) {
    if (inputBgr.empty()) {
        throw std::runtime_error("Cannot filter an empty frame.");
    }
    if (inputBgr.type() != CV_8UC3) {
        throw std::runtime_error("FrameProcessor expects CV_8UC3 BGR input.");
    }

    ensureGpuResources(inputBgr.size());

    uploadedFrame.upload(inputBgr, stream);
    cv::cuda::cvtColor(uploadedFrame, yCrCbFrame, cv::COLOR_BGR2YCrCb, 0, stream);
    cv::cuda::split(yCrCbFrame, yCrCbChannels, stream);

    const PercentileStats rawStats = computeStats(yCrCbChannels[0]);
    const PercentileStats activeStats = smoothStats(rawStats, frameMetadata);
    const ChromaCorrectionState activeChromaCorrection = smoothChromaCorrection(computeTargetChromaCorrection(activeStats));
    updateToneMapLut(activeStats);

    toneMapLookup->transform(yCrCbChannels[0], toneMappedLuma, stream);
    clahe->apply(toneMappedLuma, claheLuma, stream);
    cv::cuda::addWeighted(toneMappedLuma, 1.0 - configuration.claheStrength, claheLuma, configuration.claheStrength, 0.0, contrastLuma, CV_8U, stream);

    gaussianFilter->apply(contrastLuma, blurredLuma, stream);
    cv::cuda::absdiff(contrastLuma, blurredLuma, detailMask, stream);
    cv::cuda::threshold(detailMask, detailMask, static_cast<double>(configuration.sharpenThreshold), 255.0, cv::THRESH_BINARY, stream);
    cv::cuda::addWeighted(contrastLuma, 1.0 + configuration.sharpenAmount, blurredLuma, -configuration.sharpenAmount, 0.0, sharpenedLuma, CV_8U, stream);

    contrastLuma.copyTo(finalLuma, stream);
    sharpenedLuma.copyTo(finalLuma, detailMask, stream);
    yCrCbChannels[0] = finalLuma;
    applyChromaCorrection(activeChromaCorrection);

    cv::cuda::merge(yCrCbChannels, yCrCbFrame, stream);
    cv::cuda::cvtColor(yCrCbFrame, uploadedFrame, cv::COLOR_YCrCb2BGR, 0, stream);
    uploadedFrame.download(outputBgr, stream);
    stream.waitForCompletion();
}

PercentileStats FrameProcessor::getCurrentStats() const {
    return smoothedStats;
}

void FrameProcessor::initializePersistentResources() {
    toneMapTable = cv::Mat(1, 256, CV_8UC1);
    clahe = cv::cuda::createCLAHE(configuration.claheClipLimit, configuration.claheGridSize);
}

void FrameProcessor::resetTemporalState() {
    hasSmoothedStats = false;
    hasToneMapState = false;
    hasChromaCorrection = false;
    smoothedStats = PercentileStats();
    toneMapState = ToneMapState();
    chromaCorrectionState = ChromaCorrectionState();
    lastPtsNs = -1;
    lastFrameIntervalSeconds = defaultFrameIntervalSeconds;
    currentFrameSize = cv::Size();
}

void FrameProcessor::ensureGpuResources(const cv::Size& frameSize) {
    if (frameSize == currentFrameSize) {
        return;
    }

    currentFrameSize = frameSize;
    uploadedFrame.create(frameSize, CV_8UC3);
    yCrCbFrame.create(frameSize, CV_8UC3);
    toneMappedLuma.create(frameSize, CV_8UC1);
    claheLuma.create(frameSize, CV_8UC1);
    contrastLuma.create(frameSize, CV_8UC1);
    chromaLowerMask.create(frameSize, CV_8UC1);
    chromaUpperMask.create(frameSize, CV_8UC1);
    chromaMask.create(frameSize, CV_8UC1);
    blurredLuma.create(frameSize, CV_8UC1);
    sharpenedLuma.create(frameSize, CV_8UC1);
    detailMask.create(frameSize, CV_8UC1);
    finalLuma.create(frameSize, CV_8UC1);

    yCrCbChannels.resize(3);
    for (cv::cuda::GpuMat& channel : yCrCbChannels) {
        channel.create(frameSize, CV_8UC1);
    }

    clahe = cv::cuda::createCLAHE(configuration.claheClipLimit, configuration.claheGridSize);
    gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1,cv::Size(computeGaussianKernelSize(configuration.sharpenSigma), computeGaussianKernelSize(configuration.sharpenSigma)), configuration.sharpenSigma);
}

PercentileStats FrameProcessor::computeStats(const cv::cuda::GpuMat& luma) {
    cv::cuda::calcHist(luma, histGpu, stream);
    histGpu.download(histCpu, stream);
    stream.waitForCompletion();

    if (histCpu.empty()) {
        throw std::runtime_error("Failed to compute luma histogram.");
    }

    cv::Mat flatHist = histCpu.reshape(1, 1);
    if (flatHist.cols < 256) {
        throw std::runtime_error("Unexpected histogram shape from CUDA histogram.");
    }

    const std::int64_t totalPixels = static_cast<std::int64_t>(luma.rows) * static_cast<std::int64_t>(luma.cols);
    if (totalPixels <= 0) {
        throw std::runtime_error("Invalid frame dimensions for histogram statistics.");
    }

    auto readBin = [&flatHist](int index) -> double {
        if (flatHist.type() == CV_32SC1) {
            return static_cast<double>(flatHist.at<int>(0, index));
        }
        if (flatHist.type() == CV_32FC1) {
            return static_cast<double>(flatHist.at<float>(0, index));
        }
        throw std::runtime_error("Unsupported histogram data type returned by CUDA histogram.");
    };

    auto percentileToValue = [&](double fraction) -> double {
        const double target = std::max(1.0, fraction * static_cast<double>(totalPixels));
        double cumulative = 0.0;
        for (int value = 0; value < 256; ++value) {
            cumulative += readBin(value);
            if (cumulative >= target) {
                return static_cast<double>(value);
            }
        }
        return 255.0;
    };

    return clampStats(PercentileStats(
        percentileToValue(0.05),
        percentileToValue(0.50),
        percentileToValue(0.95)));
}

PercentileStats FrameProcessor::smoothStats(const PercentileStats& rawStats, const FrameMetadata& frameMetadata) {
    double frameIntervalSeconds = defaultFrameIntervalSeconds;
    if (frameMetadata.durationNs > 0) {
        frameIntervalSeconds = static_cast<double>(frameMetadata.durationNs) * 1e-9;
    } else if (lastPtsNs >= 0 and frameMetadata.ptsNs > lastPtsNs) {
        frameIntervalSeconds = static_cast<double>(frameMetadata.ptsNs - lastPtsNs) * 1e-9;
    }
    frameIntervalSeconds = std::clamp(frameIntervalSeconds, 1.0 / 240.0, 1.0);
    lastFrameIntervalSeconds = frameIntervalSeconds;

    if (not hasSmoothedStats) {
        smoothedStats = clampStats(rawStats);
        hasSmoothedStats = true;
        lastPtsNs = frameMetadata.ptsNs;
        return smoothedStats;
    }

    double alpha = 1.0 - std::exp(-frameIntervalSeconds / configuration.emaTimeConstantSeconds);
    alpha = std::clamp(alpha, 0.02, 0.50);

    smoothedStats.p5 = ((1.0 - alpha) * smoothedStats.p5) + (alpha * rawStats.p5);
    smoothedStats.p50 = ((1.0 - alpha) * smoothedStats.p50) + (alpha * rawStats.p50);
    smoothedStats.p95 = ((1.0 - alpha) * smoothedStats.p95) + (alpha * rawStats.p95);
    smoothedStats = clampStats(smoothedStats);
    lastPtsNs = frameMetadata.ptsNs;
    return smoothedStats;
}

ChromaCorrectionState FrameProcessor::computeTargetChromaCorrection(const PercentileStats& stats) {
    const double range = std::max(1.0, stats.p95 - stats.p5);
    double lowThreshold = std::clamp(stats.p5 + (range * configuration.chromaMidtoneMargin), 0.0, 255.0);
    double highThreshold = std::clamp(stats.p95 - (range * configuration.chromaMidtoneMargin), 0.0, 255.0);
    if (highThreshold <= lowThreshold) {
        const double center = (lowThreshold + highThreshold) * 0.5;
        lowThreshold = std::clamp(center - 0.5, 0.0, 254.0);
        highThreshold = std::clamp(center + 0.5, lowThreshold + 1.0, 255.0);
    }

    cv::cuda::compareWithScalar(yCrCbChannels[0], cv::Scalar::all(lowThreshold), chromaLowerMask, cv::CMP_GE, stream);
    cv::cuda::compareWithScalar(yCrCbChannels[0], cv::Scalar::all(highThreshold), chromaUpperMask, cv::CMP_LE, stream);
    cv::cuda::bitwise_and(chromaLowerMask, chromaUpperMask, chromaMask, cv::noArray(), stream);
    stream.waitForCompletion();

    const double maskSum = cv::cuda::sum(chromaMask)[0];
    const double selectedPixelCount = maskSum / 255.0;
    const double totalPixels = static_cast<double>(yCrCbChannels[0].rows) * static_cast<double>(yCrCbChannels[0].cols);
    if (selectedPixelCount < (totalPixels * minimumChromaSampleRatio)) {
        return ChromaCorrectionState();
    }

    const double meanCr = cv::cuda::sum(yCrCbChannels[1], chromaMask)[0] / selectedPixelCount;
    const double meanCb = cv::cuda::sum(yCrCbChannels[2], chromaMask)[0] / selectedPixelCount;

    const double desiredCrOffset = std::clamp(-(meanCr - chromaNeutralValue) * configuration.chromaCorrectionStrength, -configuration.maxChromaCorrection, configuration.maxChromaCorrection);
    const double desiredCbOffset = std::clamp(-(meanCb - chromaNeutralValue) * configuration.chromaCorrectionStrength, -configuration.maxChromaCorrection, configuration.maxChromaCorrection);

    return ChromaCorrectionState(desiredCrOffset, desiredCbOffset);
}

ChromaCorrectionState FrameProcessor::smoothChromaCorrection(const ChromaCorrectionState& targetChromaCorrection) {
    if (not hasChromaCorrection) {
        chromaCorrectionState = targetChromaCorrection;
        hasChromaCorrection = true;
        return chromaCorrectionState;
    }

    const double alpha = std::clamp(
        1.0 - std::exp(-lastFrameIntervalSeconds / configuration.chromaBiasTimeConstantSeconds),
        0.02,
        0.35);
    chromaCorrectionState.crOffset = ((1.0 - alpha) * chromaCorrectionState.crOffset) + (alpha * targetChromaCorrection.crOffset);
    chromaCorrectionState.cbOffset = ((1.0 - alpha) * chromaCorrectionState.cbOffset) + (alpha * targetChromaCorrection.cbOffset);
    
    chromaCorrectionState.crOffset = std::clamp(
        chromaCorrectionState.crOffset,
        -configuration.maxChromaCorrection,
        configuration.maxChromaCorrection);

    chromaCorrectionState.cbOffset = std::clamp(
        chromaCorrectionState.cbOffset,
        -configuration.maxChromaCorrection,
        configuration.maxChromaCorrection);

    return chromaCorrectionState;
}

void FrameProcessor::applyChromaCorrection(const ChromaCorrectionState& chromaCorrection) {
    cv::cuda::addWithScalar(yCrCbChannels[1], cv::Scalar::all(chromaCorrection.crOffset), yCrCbChannels[1], cv::noArray(), -1, stream);
    cv::cuda::addWithScalar(yCrCbChannels[2], cv::Scalar::all(chromaCorrection.cbOffset), yCrCbChannels[2], cv::noArray(), -1, stream);
}

ToneMapState FrameProcessor::computeTargetToneMapState(const PercentileStats& stats) const {
    PercentileStats adjustedStats = clampStats(stats);

    double low = adjustedStats.p5;
    double high = adjustedStats.p95;

    if ((high - low) < configuration.minPercentileRange) {
        const double halfRange = configuration.minPercentileRange * 0.5;
        low = adjustedStats.p50 - halfRange;
        high = adjustedStats.p50 + halfRange;
        if (low < 0.0) {
            high = std::min(255.0, high - low);
            low = 0.0;
        }
        if (high > 255.0) {
            low = std::max(0.0, low - (high - 255.0));
            high = 255.0;
        }
    }

    const double range = std::max(1.0, high - low);
    const double currentMidtone = std::clamp((adjustedStats.p50 - low) / range, 0.18, 0.82);
    const double desiredGamma = std::clamp(std::log(configuration.targetMidtone) / std::log(currentMidtone), 0.80, 1.20);
    const double gamma = std::clamp(
        1.0 + ((desiredGamma - 1.0) * configuration.gammaStrength),
        configuration.minGamma,
        configuration.maxGamma);

    return clampToneMapState(ToneMapState(low, high, gamma), configuration.minPercentileRange);
}

ToneMapState FrameProcessor::smoothToneMapState(const ToneMapState& targetToneMapState) {
    if (not hasToneMapState) {
        toneMapState = clampToneMapState(targetToneMapState, configuration.minPercentileRange);
        hasToneMapState = true;
        return toneMapState;
    }

    ToneMapState deadbandTarget(targetToneMapState);
    deadbandTarget.low = applyDeadband(deadbandTarget.low, toneMapState.low, configuration.toneMapDeadbandLevels);
    deadbandTarget.high = applyDeadband(deadbandTarget.high, toneMapState.high, configuration.toneMapDeadbandLevels);
    deadbandTarget.gamma = applyDeadband(deadbandTarget.gamma, toneMapState.gamma, configuration.toneMapGammaDeadband);

    const double alpha = std::clamp(
        1.0 - std::exp(-lastFrameIntervalSeconds / configuration.toneMapTimeConstantSeconds),
        0.02,
        0.35);
        
    const double maxLevelStep = configuration.toneMapMaxLevelStepPerSecond * lastFrameIntervalSeconds;
    const double maxGammaStep = configuration.toneMapMaxGammaStepPerSecond * lastFrameIntervalSeconds;

    auto smoothLevel = [&](double currentValue, double targetValue) -> double {
        const double proposedValue = currentValue + ((targetValue - currentValue) * alpha);
        const double delta = std::clamp(proposedValue - currentValue, -maxLevelStep, maxLevelStep);
        return currentValue + delta;
    };
    auto smoothGamma = [&](double currentValue, double targetValue) -> double {
        const double proposedValue = currentValue + ((targetValue - currentValue) * alpha);
        const double delta = std::clamp(proposedValue - currentValue, -maxGammaStep, maxGammaStep);
        return currentValue + delta;
    };

    toneMapState.low = smoothLevel(toneMapState.low, deadbandTarget.low);
    toneMapState.high = smoothLevel(toneMapState.high, deadbandTarget.high);
    toneMapState.gamma = smoothGamma(toneMapState.gamma, deadbandTarget.gamma);
    toneMapState = clampToneMapState(toneMapState, configuration.minPercentileRange);
    toneMapState.gamma = std::clamp(toneMapState.gamma, configuration.minGamma, configuration.maxGamma);
    return toneMapState;
}

double FrameProcessor::applyDeadband(double targetValue, double referenceValue, double deadband) const {
    if (std::abs(targetValue - referenceValue) <= deadband) {
        return referenceValue;
    }
    return targetValue;
}

void FrameProcessor::updateToneMapLut(const PercentileStats& stats) {
    const ToneMapState activeToneMapState = smoothToneMapState(computeTargetToneMapState(stats));
    const double low = activeToneMapState.low;
    const double high = activeToneMapState.high;
    const double gamma = activeToneMapState.gamma;
    const double range = std::max(1.0, high - low);
    const double outputLow = std::clamp(configuration.outputBlackFloor, 0.0, 254.0);
    const double outputHigh = std::clamp(configuration.outputWhiteCeiling, outputLow + 1.0, 255.0);
    const double outputRange = outputHigh - outputLow;

    for (int value = 0; value < 256; ++value) {
        const double stretched = std::clamp((static_cast<double>(value) - low) / range, 0.0, 1.0);
        const double corrected = std::pow(stretched, gamma);
        const double remapped = outputLow + (corrected * outputRange);
        toneMapTable.at<std::uint8_t>(0, value) = static_cast<std::uint8_t>(std::lround(remapped));
    }

    toneMapLookup = cv::cuda::createLookUpTable(toneMapTable);
}

int FrameProcessor::computeGaussianKernelSize(double sigma) const {
    const int radius = std::max(1, static_cast<int>(std::ceil(sigma * 3.0)));
    return (radius * 2) + 1;
}

}
