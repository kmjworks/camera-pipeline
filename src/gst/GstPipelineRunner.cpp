#include "gst/GstPipelineRunner.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

#include "gst/gstPipelineUtils.hpp"

namespace video_filter {

GstPipelineRunner::GstPipelineRunner(PipelineConfiguration configurationValue, std::unique_ptr<VideoProcessor> processorValue) {
    configuration = std::move(configurationValue);
    processor = std::move(processorValue);
    if (not processor) {
        throw std::runtime_error("GstPipelineRunner requires a valid frame processor.");
    }

    pipelineBuilder.configure(configuration);
    resetRuntimeState();
    inputIsLive = configuration.inputKind != InputKind::file;
}

GstPipelineRunner::~GstPipelineRunner() {
    stopPipelines();
}

int GstPipelineRunner::run() {
    inputPipeline = gst_utils::createPipeline(pipelineBuilder.buildInputPipeline(), "input");
    inputSink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(inputPipeline), "inputSink"));
    if (inputSink == nullptr) {
        throw std::runtime_error("Input pipeline does not contain the expected appsink named inputSink.");
    }

    configureInputSink();

    if (gst_element_set_state(inputPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        throw std::runtime_error("Failed to start the input pipeline.");
    }

    bool inputEos = false;
    bool outputEos = false;

    while (true) {
        gst_utils::drainBus(inputPipeline, "input", inputEos);
        gst_utils::drainBus(outputPipeline, "output", outputEos);
        if (outputEos and not inputEos) {
            throw std::runtime_error("Output pipeline reached EOS before the input stream completed.");
        }

        std::unique_ptr<GstSample, decltype(&gst_sample_unref)> sample(
            gst_app_sink_try_pull_sample(inputSink, gst_utils::sampleTimeout),
            &gst_sample_unref);
        if (not sample) {
            if (inputEos or gst_app_sink_is_eos(inputSink) != 0) {
                break;
            }
            continue;
        }

        ensureOutputPipeline(sample.get());

        GstBuffer* inputBuffer = gst_sample_get_buffer(sample.get());
        gst_utils::MappedVideoFrame mappedFrame(inputVideoInfo, inputBuffer);
        // inputFrame is a borrowed view into inputBuffer and must not escape this scope
        const cv::Mat& inputFrame = mappedFrame.asMat();

        const auto processStart = std::chrono::steady_clock::now();
        cv::Mat enhancedFrame;
        processor->processFrame(
            inputFrame,
            enhancedFrame,
            FrameMetadata(
                gst_utils::clockTimeToSigned(GST_BUFFER_PTS(inputBuffer)),
                gst_utils::clockTimeToSigned(GST_BUFFER_DURATION(inputBuffer))));
        const auto processEnd = std::chrono::steady_clock::now();

        const double processMs = std::chrono::duration<double, std::milli>(processEnd - processStart).count();
        totalProcessMs += processMs;
        ++framesProcessed;

        const cv::Mat presentationFrame = frameComposer.compose(inputFrame, enhancedFrame, configuration.sideBySideDemo);
        pushFrame(presentationFrame, inputBuffer);
    }

    if (outputSource != nullptr) {
        const GstFlowReturn eosResult = gst_app_src_end_of_stream(outputSource);
        if (eosResult != GST_FLOW_OK) {
            throw std::runtime_error("Failed to push EOS to the output pipeline.");
        }
    }

    waitForOutputDrain();
    printSummary();
    return 0;
}

void GstPipelineRunner::configureInputSink() {
    gst_app_sink_set_emit_signals(inputSink, FALSE);
    gst_app_sink_set_drop(inputSink, inputIsLive ? TRUE : FALSE);
    gst_app_sink_set_max_buffers(inputSink, inputIsLive ? 2U : 8U);
    gst_app_sink_set_wait_on_eos(inputSink, FALSE);
}

void GstPipelineRunner::ensureOutputPipeline(const GstSample* sample) {
    if (outputReady) {
        return;
    }

    GstCaps* caps = gst_sample_get_caps(const_cast<GstSample*>(sample));
    if (caps == nullptr) {
        throw std::runtime_error("Input sample did not expose negotiated caps.");
    }
    if (gst_video_info_from_caps(&inputVideoInfo, caps) == 0) {
        throw std::runtime_error("Failed to parse negotiated input caps into GstVideoInfo.");
    }

    inputVideoInfoReady = true;
    outputPipeline = gst_utils::createPipeline(pipelineBuilder.buildOutputPipeline(inputVideoInfo, inputIsLive), "output");
    outputSource = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(outputPipeline), "outputSource"));
    if (outputSource == nullptr) {
        throw std::runtime_error("Output pipeline does not contain the expected appsrc named outputSource.");
    }

    gst_app_src_set_stream_type(outputSource, GST_APP_STREAM_TYPE_STREAM);
    const cv::Size outputFrameSize = pipelineBuilder.getOutputFrameSize(inputVideoInfo);
    const std::size_t outputFrameBytes =
        static_cast<std::size_t>(outputFrameSize.width) *
        static_cast<std::size_t>(outputFrameSize.height) *
        3U;
    gst_app_src_set_max_bytes(outputSource, static_cast<guint64>(outputFrameBytes * 4U));

    GstCaps* outputCaps = gst_caps_from_string(pipelineBuilder.buildAppSrcCaps(inputVideoInfo).c_str());
    if (outputCaps == nullptr) {
        throw std::runtime_error("Failed to create output caps for appsrc.");
    }
    gst_app_src_set_caps(outputSource, outputCaps);
    gst_caps_unref(outputCaps);

    if (gst_element_set_state(outputPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        throw std::runtime_error("Failed to start the output pipeline.");
    }

    outputReady = true;
}

void GstPipelineRunner::pushFrame(const cv::Mat& frame, const GstBuffer* inputBuffer) {
    if (not outputReady or outputSource == nullptr) {
        throw std::runtime_error("Cannot push a frame before the output pipeline is ready.");
    }

    GstBuffer* outputBuffer = gst_utils::copyMatToBuffer(frame);
    GST_BUFFER_PTS(outputBuffer) = GST_BUFFER_PTS(inputBuffer);
    GST_BUFFER_DTS(outputBuffer) = GST_BUFFER_DTS(inputBuffer);
    GST_BUFFER_DURATION(outputBuffer) = GST_BUFFER_DURATION(inputBuffer);
    GST_BUFFER_OFFSET(outputBuffer) = GST_BUFFER_OFFSET(inputBuffer);
    GST_BUFFER_OFFSET_END(outputBuffer) = GST_BUFFER_OFFSET_END(inputBuffer);

    const GstFlowReturn pushResult = gst_app_src_push_buffer(outputSource, outputBuffer);
    if (pushResult != GST_FLOW_OK) {
        throw std::runtime_error("Failed to push a processed frame into appsrc.");
    }
}

void GstPipelineRunner::waitForOutputDrain() const {
    if (outputPipeline == nullptr) {
        return;
    }

    GstBus* bus = gst_element_get_bus(outputPipeline);
    if (bus == nullptr) {
        throw std::runtime_error("Failed to obtain the bus for the output pipeline.");
    }

    GstMessageType messageType = static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR);
    GstMessage* message = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, messageType);

    if (message != nullptr) {
        if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
            GError* error = nullptr;
            gchar* debugInfo = nullptr;
            gst_message_parse_error(message, &error, &debugInfo);
            const std::string messageText = error != nullptr ? error->message : "Unknown output pipeline error";
            g_clear_error(&error);
            g_free(debugInfo);
            gst_message_unref(message);
            gst_object_unref(bus);
            throw std::runtime_error(messageText);
        }
        gst_message_unref(message);
    }
    gst_object_unref(bus);
}

void GstPipelineRunner::printSummary() const {
    std::cout << "Processed " << framesProcessed << " frames";
    if (framesProcessed > 0) {
        const double averageMs = totalProcessMs / static_cast<double>(framesProcessed);
        std::cout << " | average enhance time: " << std::fixed << std::setprecision(2) << averageMs << " ms/frame";
    }
    std::cout << '\n';
}

void GstPipelineRunner::resetRuntimeState() {
    inputPipeline = nullptr;
    outputPipeline = nullptr;
    inputSink = nullptr;
    outputSource = nullptr;
    gst_video_info_init(&inputVideoInfo);
    inputVideoInfoReady = false;
    outputReady = false;
    inputIsLive = false;
    framesProcessed = 0;
    totalProcessMs = 0.0;
}

void GstPipelineRunner::stopPipelines() {
    if (inputPipeline != nullptr) {
        gst_element_set_state(inputPipeline, GST_STATE_NULL);
    }
    if (outputPipeline != nullptr) {
        gst_element_set_state(outputPipeline, GST_STATE_NULL);
    }
    if (inputSink != nullptr) {
        gst_object_unref(inputSink);
        inputSink = nullptr;
    }
    if (outputSource != nullptr) {
        gst_object_unref(outputSource);
        outputSource = nullptr;
    }
    if (inputPipeline != nullptr) {
        gst_object_unref(inputPipeline);
        inputPipeline = nullptr;
    }
    if (outputPipeline != nullptr) {
        gst_object_unref(outputPipeline);
        outputPipeline = nullptr;
    }
}

}
