#include "gst/gstPipelineUtils.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace video_filter::gst_utils {

MappedVideoFrame::MappedVideoFrame(const GstVideoInfo& videoInfo, GstBuffer* buffer) {
    mapped = false;
    std::memset(&videoFrame, 0, sizeof(videoFrame));

    if (buffer == nullptr) {
        throw std::runtime_error("Input sample did not contain a valid buffer.");
    }
    if (gst_video_frame_map(&videoFrame, &videoInfo, buffer, GST_MAP_READ) == 0) {
        throw std::runtime_error("Failed to map input frame for reading.");
    }

    mapped = true;
    frame = cv::Mat(videoInfo.height, videoInfo.width, CV_8UC3, GST_VIDEO_FRAME_PLANE_DATA(&videoFrame, 0), static_cast<std::size_t>(GST_VIDEO_FRAME_PLANE_STRIDE(&videoFrame, 0)));
}

MappedVideoFrame::~MappedVideoFrame() {
    if (mapped) {
        gst_video_frame_unmap(&videoFrame);
    }
}

const cv::Mat& MappedVideoFrame::asMat() const {
    return frame;
}

std::int64_t clockTimeToSigned(GstClockTime value) {
    if (value == GST_CLOCK_TIME_NONE) {
        return -1;
    }
    return static_cast<std::int64_t>(value);
}

bool hasElementFactory(const char* factoryName) {
    GstElementFactory* factory = gst_element_factory_find(factoryName);
    if (factory == nullptr) {
        return false;
    }
    gst_object_unref(factory);
    return true;
}

std::string escapePropertyValue(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        if (character == '\\' or character == '"') {
            escaped.push_back('\\');
        }

        escaped.push_back(character);
    }
    return escaped;
}

GstElement* createPipeline(const std::string& description, const std::string& label) {
    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(description.c_str(), &error);
    if (error != nullptr) {
        const std::string message = error->message;
        g_clear_error(&error);
        throw std::runtime_error("Failed to create " + label + " pipeline: " + message + "\nPipeline: " + description);
    }
    if (pipeline == nullptr) {
        throw std::runtime_error("gst_parse_launch returned a null pipeline for " + label + '.');
    }
    return pipeline;
}

void drainBus(GstElement* pipeline, const std::string& label, bool& sawEos) {
    if (pipeline == nullptr) {
        return;
    }

    GstBus* bus = gst_element_get_bus(pipeline);
    if (bus == nullptr) {
        throw std::runtime_error("Failed to obtain the bus for the " + label + " pipeline.");
    }

    while (true) {
        GstMessageType messageType = static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_WARNING);
        GstMessage* message = gst_bus_pop_filtered(bus, messageType);
        if (message == nullptr) {
            break;
        }

        if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_WARNING) {
            GError* error = nullptr;
            gchar* debugInfo = nullptr;
            gst_message_parse_warning(message, &error, &debugInfo);
            if (error != nullptr) {
                std::cerr << label << " warning: " << error->message << '\n';
            }
            g_clear_error(&error);
            g_free(debugInfo);
        } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
            sawEos = true;
        } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
            GError* error = nullptr;
            gchar* debugInfo = nullptr;
            gst_message_parse_error(message, &error, &debugInfo);
            const std::string messageText = error != nullptr ? error->message : "Unknown GStreamer error";
            g_clear_error(&error);
            g_free(debugInfo);
            gst_message_unref(message);
            gst_object_unref(bus);
            throw std::runtime_error(label + " pipeline error: " + messageText);
        }

        gst_message_unref(message);
    }

    gst_object_unref(bus);
}

GstBuffer* copyMatToBuffer(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::runtime_error("Cannot push an empty frame.");
    }
    if (frame.type() != CV_8UC3) {
        throw std::runtime_error("Output frame must be CV_8UC3 BGR.");
    }

    const std::size_t rowBytes = static_cast<std::size_t>(frame.cols) * 3U;
    const std::size_t bufferSize = rowBytes * static_cast<std::size_t>(frame.rows);
    GstBuffer* outputBuffer = gst_buffer_new_allocate(nullptr, bufferSize, nullptr);
    if (outputBuffer == nullptr) {
        throw std::runtime_error("Failed to allocate output GstBuffer.");
    }

    GstMapInfo mapInfo;
    memset(&mapInfo, 0, sizeof(GstMapInfo));

    if (gst_buffer_map(outputBuffer, &mapInfo, GST_MAP_WRITE) == 0) {
        gst_buffer_unref(outputBuffer);
        throw std::runtime_error("Failed to map output GstBuffer for writing.");
    }

    for (int row = 0; row < frame.rows; ++row) {
        std::memcpy(mapInfo.data + (static_cast<std::size_t>(row) * rowBytes), frame.ptr(row), rowBytes);
    }

    gst_buffer_unmap(outputBuffer, &mapInfo);
    return outputBuffer;
}

}
