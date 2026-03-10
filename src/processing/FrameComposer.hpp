#pragma once

#include <opencv2/core.hpp>

namespace video_filter {

class FrameComposer {
public:
    FrameComposer();

    cv::Mat compose(const cv::Mat& inputFrame, const cv::Mat& enhancedFrame, bool sideBySideDemo) const;
};

}
