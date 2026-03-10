#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "processing/VideoProcessor.hpp"

namespace video_filter {

struct PercentileStats {
    PercentileStats();
    PercentileStats(double p5Value, double p50Value, double p95Value);

    double p5;
    double p50;
    double p95;
};

struct FrameProcessorConfiguration {
    FrameProcessorConfiguration();

    double emaTimeConstantSeconds;
    // Unitless normalized luma target in the range 0.0 - 1.0
    double targetMidtone;
    // Luma levels in the range 0 - 255
    double minPercentileRange;
    // Unitless correction fraction in the range 0.0 - 1.0
    double chromaCorrectionStrength;
    double chromaBiasTimeConstantSeconds;
    // Chroma levels in the range 0 - 255
    double maxChromaCorrection;
    // Unitless fraction of the active luma range excluded from each end during chroma estimation
    double chromaMidtoneMargin;
    // Unitless OpenCV CLAHE clip-limit multiplier
    double claheClipLimit;
    // Tile counts as {columns, rows}
    cv::Size claheGridSize;
    // Unitless blend factor in the range 0.0 - 1.0
    double claheStrength;
    // Unitless strength applied to the gamma correction delta
    double gammaStrength;
    double toneMapTimeConstantSeconds;
    // Luma levels in the range 0 - 255
    double toneMapDeadbandLevels;
    // Unitless gamma delta
    double toneMapGammaDeadband;
    double toneMapMaxLevelStepPerSecond;
    double toneMapMaxGammaStepPerSecond;
    // Unitless gamma value.
    double minGamma;
    // Unitless gamma value.
    double maxGamma;
    // Luma levels in the range 0 - 255
    double outputBlackFloor;
    // Luma levels in the range 0 - 255
    double outputWhiteCeiling;
    // Unitless unsharp-mask gain
    double sharpenAmount;
    // Luma levels in the range 0 - 255
    int sharpenThreshold;
    double sharpenSigma;
};

struct ToneMapState {
    ToneMapState();
    ToneMapState(double lowValue, double highValue, double gammaValue);

    double low;
    double high;
    double gamma;
};

struct ChromaCorrectionState {
    ChromaCorrectionState();
    ChromaCorrectionState(double crOffsetValue, double cbOffsetValue);

    double crOffset;
    double cbOffset;
};

class FrameProcessor : public VideoProcessor {
public:
    explicit FrameProcessor(FrameProcessorConfiguration configuration = FrameProcessorConfiguration());

    void processFrame(const cv::Mat& inputBgr, cv::Mat& outputBgr, const FrameMetadata& frameMetadata) override; 
    /// @brief Reads the latest temporal statistics snapshot; Do not call concurrently with processFrame unless externally synchronized 
    PercentileStats getCurrentStats() const;

private:
    void initializePersistentResources();
    void resetTemporalState();
    void ensureGpuResources(const cv::Size& frameSize);
    PercentileStats computeStats(const cv::cuda::GpuMat& luma);
    PercentileStats smoothStats(const PercentileStats& rawStats, const FrameMetadata& frameMetadata);
    ChromaCorrectionState computeTargetChromaCorrection(const PercentileStats& stats);
    ChromaCorrectionState smoothChromaCorrection(const ChromaCorrectionState& targetChromaCorrection);
    void applyChromaCorrection(const ChromaCorrectionState& chromaCorrection);
    ToneMapState computeTargetToneMapState(const PercentileStats& stats) const;
    ToneMapState smoothToneMapState(const ToneMapState& targetToneMapState);
    double applyDeadband(double targetValue, double referenceValue, double deadband) const;
    void updateToneMapLut(const PercentileStats& stats);
    int computeGaussianKernelSize(double sigma) const;

    FrameProcessorConfiguration configuration;

    /// @brief Temporal state reused across consecutive frames
    bool hasSmoothedStats;
    bool hasToneMapState;
    bool hasChromaCorrection;
    PercentileStats smoothedStats;
    ToneMapState toneMapState;
    ChromaCorrectionState chromaCorrectionState;
    std::int64_t lastPtsNs;
    double lastFrameIntervalSeconds;

    /// @brief Reused CUDA-OpenCV interop resources; these objects are tied to this instance and must outlive every in-flight call; they are not thread-safe.
    cv::Ptr<cv::cuda::CLAHE> clahe;
    cv::Ptr<cv::cuda::Filter> gaussianFilter;
    cv::Ptr<cv::cuda::LookUpTable> toneMapLookup;
    cv::Size currentFrameSize;
    cv::Mat toneMapTable;
    cv::cuda::GpuMat uploadedFrame;
    cv::cuda::GpuMat yCrCbFrame;
    std::vector<cv::cuda::GpuMat> yCrCbChannels;
    cv::cuda::GpuMat toneMappedLuma;
    cv::cuda::GpuMat claheLuma;
    cv::cuda::GpuMat contrastLuma;
    cv::cuda::GpuMat chromaLowerMask;
    cv::cuda::GpuMat chromaUpperMask;
    cv::cuda::GpuMat chromaMask;
    cv::cuda::GpuMat blurredLuma;
    cv::cuda::GpuMat sharpenedLuma;
    cv::cuda::GpuMat detailMask;
    cv::cuda::GpuMat finalLuma;
    cv::cuda::GpuMat histGpu;
    cv::Mat histCpu;
    cv::cuda::Stream stream;
};

}
