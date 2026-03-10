#pragma once

#include <optional>
#include <string>

#include "configuration/CameraConfiguration.hpp"
#include "processing/FrameProcessor.hpp"

namespace video_filter {

enum class InputKind {
    file,
    camera,
    customPipeline,
};

class PipelineConfiguration {
public:
    PipelineConfiguration();

    InputKind inputKind;
    std::string inputPath;
    CameraConfiguration camera;
    std::string sourcePipeline;
    std::string outputPath;
    bool displayOutput;
    bool sideBySideDemo;
    FrameProcessorConfiguration frameProcessorConfiguration;
};

}
