#include "configuration/PipelineConfiguration.hpp"

namespace video_filter {

PipelineConfiguration::PipelineConfiguration() {
    inputKind = InputKind::file;
    inputPath.clear();
    sourcePipeline.clear();
    outputPath.clear();
    displayOutput = true;
    sideBySideDemo = false;
}

}
