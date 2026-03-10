cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(camera_control_filter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaarithm cudaimgproc cudafilters)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

add_executable(camera_control_filter
    src/configuration/CameraConfiguration.cpp
    src/configuration/PipelineConfiguration.cpp
    src/gst/GstPipelineBuilder.cpp
    src/gst/GstPipelineRunner.cpp
    src/gst/gstPipelineUtils.cpp
    src/main.cpp
    src/processing/FrameComposer.cpp
    src/processing/FrameProcessor.cpp
    src/utilities/commandLineUtilities.cpp
    src/utilities/commonUtilities.cpp
)

target_include_directories(camera_control_filter PRIVATE src)

target_link_libraries(camera_control_filter
    PRIVATE
        CUDA::cudart
        Boost::program_options
        PkgConfig::GSTREAMER
        ${OpenCV_LIBS}
)

target_compile_options(camera_control_filter PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
)
