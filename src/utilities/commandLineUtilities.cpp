#include "utilities/commandLineUtilities.hpp"
#include "utilities/commonUtilities.hpp"

#include <sstream>
#include <string>

#include <boost/program_options.hpp>

namespace video_filter {
namespace {

namespace po = boost::program_options;

po::options_description buildOptionsDescription() {
    po::options_description options("Options");
    options.add_options()
        ("help,h", "Show this message.")
        ("input", po::value<std::string>(), "Read from a video file through GStreamer decodebin.")
        ("camera", po::value<std::string>(), "Read from a live V4L2 camera in YUY2/YUYV.")
        ("source-pipeline", po::value<std::string>(), "Advanced: custom GStreamer source branch, without appsink.")
        ("width", po::value<int>(), "Camera width in pixels.")
        ("height", po::value<int>(), "Camera height in pixels.")
        ("fps", po::value<std::string>(), "Camera frame rate as n or n/d.")
        ("ema-seconds", po::value<double>(), "EMA time constant for luma stats smoothing.")
        ("clahe-clip-limit", po::value<double>(), "CLAHE clip limit for luma equalization.")
        ("clahe-strength", po::value<double>(), "Blend factor for conservative CLAHE on luma.")
        ("side-by-side", po::bool_switch(), "Demo mode: output original | enhanced.")
        ("output", po::value<std::string>(), "Optional output file. Requires a supported GStreamer encoder plugin.")
        ("no-display", po::bool_switch(), "Disable live display.");
    return options;
}

std::string buildHelpMessage(const char* executableName, const po::options_description& options) {
    std::ostringstream message;
    message
        << "Usage:\n"
        << "  " << executableName << " --input <video.mp4> [--output <enhanced.mp4>] [--no-display]\n"
        << "  " << executableName << " --camera </dev/video0> [--width 1280 --height 720 --fps 30]\n"
        << "  " << executableName << " --source-pipeline \"videotestsrc is-live=true ! video/x-raw,width=1280,height=720,framerate=30/1\"\n\n"
        << options << '\n';
    return message.str();
}

void applyInputSelection(PipelineConfiguration& configuration, const po::variables_map& variablesMap) {
    const bool hasInput = variablesMap.count("input") != 0;
    const bool hasCamera = variablesMap.count("camera") != 0;
    const bool hasSourcePipeline = variablesMap.count("source-pipeline") != 0;
    const int selectedInputs = static_cast<int>(hasInput) + static_cast<int>(hasCamera) + static_cast<int>(hasSourcePipeline);

    if (selectedInputs != 1) {
        throw std::runtime_error("Choose exactly one of --input, --camera, or --source-pipeline.");
    }

    if (hasInput) {
        configuration.inputKind = InputKind::file;
        configuration.inputPath = variablesMap["input"].as<std::string>();
        return;
    }
    if (hasCamera) {
        configuration.inputKind = InputKind::camera;
        configuration.camera.device = variablesMap["camera"].as<std::string>();
        return;
    }

    configuration.inputKind = InputKind::customPipeline;
    configuration.sourcePipeline = variablesMap["source-pipeline"].as<std::string>();
}

void applyCameraOptions(PipelineConfiguration& configuration, const po::variables_map& variablesMap) {
    if (variablesMap.count("width") != 0) {
        const int width = variablesMap["width"].as<int>();
        common_utils::requirePositive(width, "--width");
        configuration.camera.width = width;
    }
    if (variablesMap.count("height") != 0) {
        const int height = variablesMap["height"].as<int>();
        common_utils::requirePositive(height, "--height");
        configuration.camera.height = height;
    }
    if (variablesMap.count("fps") != 0) {
        const auto [numerator, denominator] = common_utils::parseFrameRate(variablesMap["fps"].as<std::string>());
        configuration.camera.fpsNumerator = numerator;
        configuration.camera.fpsDenominator = denominator;
    }
}

void applyFrameProcessorOptions(PipelineConfiguration& configuration, const po::variables_map& variablesMap) {
    if (variablesMap.count("ema-seconds") != 0) {
        const double emaSeconds = variablesMap["ema-seconds"].as<double>();
        common_utils::requirePositive(emaSeconds, "--ema-seconds");
        configuration.frameProcessorConfiguration.emaTimeConstantSeconds = emaSeconds;
    }
    if (variablesMap.count("clahe-clip-limit") != 0) {
        const double claheClipLimit = variablesMap["clahe-clip-limit"].as<double>();
        common_utils::requirePositive(claheClipLimit, "--clahe-clip-limit");
        configuration.frameProcessorConfiguration.claheClipLimit = claheClipLimit;
    }
    if (variablesMap.count("clahe-strength") != 0) {
        const double claheStrength = variablesMap["clahe-strength"].as<double>();
        common_utils::requireInClosedRange(claheStrength, 0.0, 1.0, "--clahe-strength");
        configuration.frameProcessorConfiguration.claheStrength = claheStrength;
    }
}

}

CommandLineTermination::CommandLineTermination(const std::string& message, int exitCodeValue)
    : std::runtime_error(message) {
    exitCode = exitCodeValue;
}

int CommandLineTermination::getExitCode() const {
    return exitCode;
}

PipelineConfiguration cli_utilities::parseCommandLineArguments(int argc, char** argv) {
    const po::options_description options = buildOptionsDescription();

    po::variables_map variablesMap;
    po::store(po::command_line_parser(argc, argv).options(options).run(), variablesMap);

    if (variablesMap.count("help") != 0) {
        throw CommandLineTermination(buildHelpMessage(argv[0], options), 0);
    }

    po::notify(variablesMap);

    PipelineConfiguration configuration;
    applyInputSelection(configuration, variablesMap);
    applyCameraOptions(configuration, variablesMap);
    applyFrameProcessorOptions(configuration, variablesMap);

    configuration.sideBySideDemo = variablesMap["side-by-side"].as<bool>();
    configuration.displayOutput = not variablesMap["no-display"].as<bool>();
    if (variablesMap.count("output") != 0) {
        configuration.outputPath = variablesMap["output"].as<std::string>();
    }

    return configuration;
}

}
