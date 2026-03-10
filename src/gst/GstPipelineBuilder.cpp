#include "gst/GstPipelineBuilder.hpp"

#include <sstream>
#include <stdexcept>

#include "gst/gstPipelineUtils.hpp"

namespace video_filter {

GstPipelineBuilder::GstPipelineBuilder() = default;

void GstPipelineBuilder::configure(const PipelineConfiguration& configurationValue) {
    configuration = configurationValue;
}

std::string GstPipelineBuilder::buildInputPipeline() const {
    std::ostringstream pipeline;
    if (configuration.inputKind == InputKind::file) {
        pipeline
            << "filesrc location=\"" << gst_utils::escapePropertyValue(configuration.inputPath) << "\" "
            << "! decodebin "
            << "! videoconvert "
            << "! video/x-raw,format=BGR "
            << "! appsink name=inputSink";
        return pipeline.str();
    }

    if (configuration.inputKind == InputKind::camera) {
        pipeline
            << "v4l2src device=\"" << gst_utils::escapePropertyValue(configuration.camera.device) << "\" do-timestamp=true "
            << "! video/x-raw,format=YUY2";
        if (configuration.camera.width.has_value()) {
            pipeline << ",width=" << *configuration.camera.width;
        }
        if (configuration.camera.height.has_value()) {
            pipeline << ",height=" << *configuration.camera.height;
        }
        if (configuration.camera.fpsNumerator.has_value()) {
            pipeline << ",framerate=" << *configuration.camera.fpsNumerator << '/' << configuration.camera.fpsDenominator;
        }
        pipeline
            << " ! videoconvert "
            << "! video/x-raw,format=BGR "
            << "! appsink name=inputSink";
        return pipeline.str();
    }

    pipeline
        << configuration.sourcePipeline << ' '
        << "! videoconvert "
        << "! video/x-raw,format=BGR "
        << "! appsink name=inputSink";
    return pipeline.str();
}

std::string GstPipelineBuilder::buildOutputPipeline(const GstVideoInfo& inputVideoInfo, bool inputIsLive) const {
    std::ostringstream pipeline;
    pipeline << "appsrc name=outputSource is-live=" << (inputIsLive ? "true" : "false")
             << " block=true format=time do-timestamp=false "
             << "caps=\"" << buildAppSrcCaps(inputVideoInfo) << "\" ";

    if (configuration.displayOutput and not configuration.outputPath.empty()) {
        pipeline
            << "! tee name=outputTee "
            << "outputTee. ! queue ! videoconvert ! autovideosink sync=true "
            << "outputTee. ! queue ! videoconvert ! " << chooseFileEncoderPipeline();
        return pipeline.str();
    }

    if (configuration.displayOutput) {
        pipeline << "! queue ! videoconvert ! autovideosink sync=true";
        return pipeline.str();
    }

    if (not configuration.outputPath.empty()) {
        pipeline << "! queue ! videoconvert ! " << chooseFileEncoderPipeline();
        return pipeline.str();
    }

    pipeline << "! queue ! fakesink sync=false";
    return pipeline.str();
}

std::string GstPipelineBuilder::buildAppSrcCaps(const GstVideoInfo& inputVideoInfo) const {
    const cv::Size outputFrameSize = getOutputFrameSize(inputVideoInfo);
    std::ostringstream caps;
    caps << "video/x-raw,format=BGR,width=" << outputFrameSize.width << ",height=" << outputFrameSize.height;
    if (inputVideoInfo.fps_n > 0 and inputVideoInfo.fps_d > 0) {
        caps << ",framerate=" << inputVideoInfo.fps_n << '/' << inputVideoInfo.fps_d;
    }
    return caps.str();
}

cv::Size GstPipelineBuilder::getOutputFrameSize(const GstVideoInfo& inputVideoInfo) const {
    if (configuration.sideBySideDemo) {
        return {inputVideoInfo.width * 2, inputVideoInfo.height};
    }
    return {inputVideoInfo.width, inputVideoInfo.height};
}

std::string GstPipelineBuilder::chooseFileEncoderPipeline() const {
    const std::string location = "filesink location=\"" + gst_utils::escapePropertyValue(configuration.outputPath) + "\"";
    if (gst_utils::hasElementFactory("x264enc")) {
        return "x264enc tune=zerolatency speed-preset=ultrafast bitrate=8000 key-int-max=30 ! h264parse ! mp4mux faststart=true ! " + location;
    }
    if (gst_utils::hasElementFactory("openh264enc")) {
        return "openh264enc bitrate=8000000 rate-control=bitrate complexity=low ! h264parse ! mp4mux faststart=true ! " + location;
    }
    if (gst_utils::hasElementFactory("avenc_mpeg4")) {
        return "avenc_mpeg4 bitrate=8000000 ! mp4mux faststart=true ! " + location;
    }
    throw std::runtime_error("No supported GStreamer video encoder was found for file output.");
}

}
