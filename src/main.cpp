#include <iostream>
#include <memory>
#include <stdexcept>

#include <opencv2/core/cuda.hpp>

#include <gst/gst.h>

#include "gst/GstPipelineRunner.hpp"
#include "processing/FrameProcessor.hpp"
#include "utilities/commandLineUtilities.hpp"

int main(int argc, char** argv) {
    gst_init(&argc, &argv);

    try {
        const video_filter::PipelineConfiguration configuration = video_filter::cli_utilities::parseCommandLineArguments(argc, argv);

        const int cudaDeviceCount = cv::cuda::getCudaEnabledDeviceCount();
        if (cudaDeviceCount <= 0) {
            throw std::runtime_error("No CUDA-capable OpenCV device is available.");
        }
        
        cv::cuda::setDevice(0);
        auto frameProcessor = std::make_unique<video_filter::FrameProcessor>(configuration.frameProcessorConfiguration);
        video_filter::GstPipelineRunner runner(configuration, std::move(frameProcessor));

        return runner.run();
    } catch (const video_filter::CommandLineTermination& termination) {
        std::ostream& outputStream = termination.getExitCode() == 0 ? std::cout : std::cerr;
        outputStream << termination.what() << '\n';
        return termination.getExitCode();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
