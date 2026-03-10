#pragma once

#include <cstdint>
#include <string>

#include <gst/gst.h>
#include <gst/video/video.h>
#include <opencv2/core.hpp>

namespace video_filter::gst_utils {

constexpr GstClockTime sampleTimeout = 100 * GST_MSECOND;

class MappedVideoFrame {
public:
    MappedVideoFrame(const GstVideoInfo& videoInfo, GstBuffer* buffer);
    ~MappedVideoFrame();

    MappedVideoFrame(const MappedVideoFrame&) = delete;
    MappedVideoFrame& operator=(const MappedVideoFrame&) = delete;


    /**
     * @brief Returns a borrowed cv::Mat view backed by the mapped GstBuffer
     * @note The view is only valid while this MappedVideoFrame object is alive
     */
    const cv::Mat& asMat() const;

private:
    // Owns the active GstVideoFrame mapping for the lifetime of this wrapper
    GstVideoFrame videoFrame;
    cv::Mat frame;
    bool mapped;
};

std::int64_t clockTimeToSigned(GstClockTime value);
bool hasElementFactory(const char* factoryName);
std::string escapePropertyValue(const std::string& value);
GstElement* createPipeline(const std::string& description, const std::string& label);
void drainBus(GstElement* pipeline, const std::string& label, bool& sawEos);
GstBuffer* copyMatToBuffer(const cv::Mat& frame);

}
