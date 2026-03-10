
This repository contains a C++ video enhancement pipeline.

## Build

Prerequisites on the target machine:
- CMake 3.20+
- Boost with `program_options`
- CUDA Toolkit 13.0
- OpenCV built with CUDA modules
- GStreamer 1.20 with `gstreamer-app-1.0` and `gstreamer-video-1.0`

Build with the helper script:

```bash
./scripts/build.sh Release
```

The binary will be created at:

```bash
./build/release/camera_control_filter
```

## Usage

### 1. Video file input

```bash
./build/release/camera_control_filter --input /path/to/input.mp4
```

To write a processed output file as well:

```bash
./build/release/camera_control_filter --input /path/to/input.mp4 --output /tmp/filtered.mp4 --no-display
```

Demo side-by-side view or recording:

```bash
./build/release/camera_control_filter \
  --input /path/to/input.mp4 \
  --side-by-side \
  --output /tmp/demo_side_by_side.mp4
```

Notes:
- The output file path uses a GStreamer encoder chain. The app tries `x264enc`, then `openh264enc`, then `avenc_mpeg4`.
- Output resolution and timestamps are copied from the negotiated input stream.
- `--side-by-side` is a demo mode. It keeps frame rate, but doubles output width by showing `original | filtered`

### 2. Live camera input

For a UVC camera delivering YUYV/YUY2:

```bash
./build/release/camera_control_filter --camera /dev/video0 --width 1280 --height 720 --fps 30
```

With side-by-side demo preview:

```bash
./build/release/camera_control_filter \
  --camera /dev/video0 \
  --width 1280 --height 720 --fps 50 \
  --side-by-side
```

### 3. Custom GStreamer source branch

This is useful for testing or alternative sources:

```bash
./build/release/camera_control_filter \
  --source-pipeline "videotestsrc is-live=true ! video/x-raw,width=1280,height=720,framerate=30/1" \
  --no-display
```

The custom source branch should stop before the sink. The application appends `videoconvert ! video/x-raw,format=BGR ! appsink` itself.

Example:

```bash
./build/release/camera_control_filter \
  --input /path/to/input.mp4 \
  --ema-seconds 0.35 \
  --clahe-clip-limit 2.0 \
  --clahe-strength 0.30
```

- `--ema-seconds`: higher values make the tone mapping steadier but slower to react to scene changes
- `--clahe-clip-limit`: higher values make local contrast stronger but can also exaggerate noise, analogue artefacts, and invalid shadow shaping
- `--clahe-strength`: lower values keep the result closer to the global tone map; higher values lean more into local contrast
- Tone-map brightness behavior is currently driven by internal defaults rather than additional command-line arguments, to keep the demo less error-prone

