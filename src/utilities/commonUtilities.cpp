#include "utilities/commonUtilities.hpp"

#include <sstream>
#include <stdexcept>

namespace video_filter::common_utils {

std::pair<int, int> parseFrameRate(const std::string& value) {
    const std::size_t slash = value.find('/');
    if (slash == std::string::npos) {
        const int numerator = std::stoi(value);
        requirePositive(numerator, "Frame rate");
        return {numerator, 1};
    }

    const int numerator = std::stoi(value.substr(0, slash));
    const int denominator = std::stoi(value.substr(slash + 1));
    requirePositive(numerator, "Frame rate numerator");
    requirePositive(denominator, "Frame rate denominator");
    return {numerator, denominator};
}

void requirePositive(double value, const std::string& optionName) {
    if (value <= 0.0) {
        throw std::runtime_error(optionName + " must be greater than zero.");
    }
}

void requirePositive(int value, const std::string& optionName) {
    if (value <= 0) {
        throw std::runtime_error(optionName + " must be greater than zero.");
    }
}

void requireInClosedRange(double value, double minimum, double maximum, const std::string& optionName) {
    if (value < minimum or value > maximum) {
        std::ostringstream message;
        message << optionName << " must be in the range [" << minimum << ", " << maximum << "].";
        throw std::runtime_error(message.str());
    }
}

}
