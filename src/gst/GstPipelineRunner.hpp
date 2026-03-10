#pragma once

#include <cstdint>
#include <memory>

#include <gst/app/app.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include "configuration/PipelineConfiguration.hpp"
#include "gst/GstPipelineBuilder.hpp"
#include "processing/FrameComposer.hpp"
#include "processing/VideoProcessor.hpp"

namespace video_filter {

class GstPipelineRunner {
public:
    GstPipelineRunner(PipelineConfiguration configuration, std::unique_ptr<VideoProcessor> processor);
    ~GstPipelineRunner();

    int run();

private:
    void configureInputSink();
    void ensureOutputPipeline(const GstSample* sample);
    void pushFrame(const cv::Mat& frame, const GstBuffer* inputBuffer);
    void waitForOutputDrain() const;
    void printSummary() const;
    void resetRuntimeState();
    void stopPipelines();

    PipelineConfiguration configuration;
    GstPipelineBuilder pipelineBuilder;
    FrameComposer frameComposer;

    /// @brief Owned exclusively by the runner and invoked from run()
    std::unique_ptr<VideoProcessor> processor;

    /// @brief These gst handles are created, configured and cleaned up solely by the runner thread; GStreamer might use internal workers but this class deos not share these pointers for concurrent access
    GstElement* inputPipeline;
    GstElement* outputPipeline;
    GstAppSink* inputSink;
    GstAppSrc* outputSource;

    /// @brief Negotiated stream metadata and run-loop state, updated from run()
    GstVideoInfo inputVideoInfo;
    bool inputVideoInfoReady;
    bool outputReady;
    bool inputIsLive;
    std::uint64_t framesProcessed;
    double totalProcessMs;
};

}
