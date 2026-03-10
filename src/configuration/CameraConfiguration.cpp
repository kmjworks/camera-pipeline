#include "configuration/CameraConfiguration.hpp"

namespace video_filter {

CameraConfiguration::CameraConfiguration() {
    device = "/dev/video0";
    width.reset();
    height.reset();
    fpsNumerator.reset();
    fpsDenominator = 1;
}

}
