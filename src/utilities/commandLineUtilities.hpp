#pragma once

#include <stdexcept>
#include <string>

#include "configuration/PipelineConfiguration.hpp"

namespace video_filter {

class CommandLineTermination : public std::runtime_error {
public:
    CommandLineTermination(const std::string& message, int exitCode);

    int getExitCode() const;

private:
    int exitCode;
};

namespace cli_utilities {

PipelineConfiguration parseCommandLineArguments(int argc, char** argv);

}

}
