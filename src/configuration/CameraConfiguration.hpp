#pragma once

#include <optional>
#include <string>

namespace video_filter {

class CameraConfiguration {
public:
    CameraConfiguration();

    std::string device;
    std::optional<int> width;
    std::optional<int> height;
    std::optional<int> fpsNumerator;
    int fpsDenominator;
};

}
