#pragma once

#include <string>
#include <utility>

namespace video_filter::common_utils {

std::pair<int, int> parseFrameRate(const std::string& value);
void requirePositive(double value, const std::string& optionName);
void requirePositive(int value, const std::string& optionName);
void requireInClosedRange(double value, double minimum, double maximum, const std::string& optionName);

}
