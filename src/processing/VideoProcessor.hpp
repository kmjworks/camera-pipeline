#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

namespace video_filter {

struct FrameMetadata {
    FrameMetadata();
    FrameMetadata(std::int64_t ptsNsValue, std::int64_t durationNsValue);

    std::int64_t ptsNs;
    std::int64_t durationNs;
};

class VideoProcessor {
public:
    virtual ~VideoProcessor() = default;

    /// @brief The caller must ensure that processFrame() is not called concurrently on the same instance. One instance is intended to be used by one processing thread.
    virtual void processFrame(const cv::Mat& inputBgr, cv::Mat& outputBgr, const FrameMetadata& frameMetadata) = 0;
};

}
