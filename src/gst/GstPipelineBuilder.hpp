#pragma once

#include <string>

#include <gst/video/video.h>
#include <opencv2/core.hpp>

#include "configuration/PipelineConfiguration.hpp"

namespace video_filter {

class GstPipelineBuilder {
public:
    GstPipelineBuilder();

    void configure(const PipelineConfiguration& configuration);
    std::string buildInputPipeline() const;
    std::string buildOutputPipeline(const GstVideoInfo& inputVideoInfo, bool inputIsLive) const;
    std::string buildAppSrcCaps(const GstVideoInfo& inputVideoInfo) const;
    cv::Size getOutputFrameSize(const GstVideoInfo& inputVideoInfo) const;

private:
    std::string chooseFileEncoderPipeline() const;

    PipelineConfiguration configuration;
};

} 
