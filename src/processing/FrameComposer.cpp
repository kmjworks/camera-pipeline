#include "processing/FrameComposer.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/imgproc.hpp>

namespace video_filter {

FrameComposer::FrameComposer() = default;

cv::Mat FrameComposer::compose(const cv::Mat& inputFrame, const cv::Mat& enhancedFrame, bool sideBySideDemo) const {
    if (not sideBySideDemo) {
        return enhancedFrame;
    }

    cv::Mat labeledInput = inputFrame.clone();
    cv::Mat labeledEnhanced = enhancedFrame.clone();

    const double fontScale = std::max(0.7, static_cast<double>(inputFrame.cols) / 1280.0);
    const int thickness = std::max(1, static_cast<int>(std::lround(fontScale * 2.0)));
    const cv::Point labelOrigin(20, 40);

    auto drawLabel = [&](cv::Mat& frame, const std::string& label) {
        cv::putText(frame, label, labelOrigin, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
        cv::putText(frame, label, labelOrigin, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    };

    drawLabel(labeledInput, "Original");
    drawLabel(labeledEnhanced, "Filtered");

    cv::Mat composed;
    cv::hconcat(labeledInput, labeledEnhanced, composed);
    cv::line(
        composed,
        cv::Point(inputFrame.cols, 0),
        cv::Point(inputFrame.cols, composed.rows - 1),
        cv::Scalar(255, 255, 255),
        std::max(2, thickness),
        cv::LINE_AA);
    return composed;
}

}
