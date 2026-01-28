# DeepStream ALPR Pipeline - Python Binding

Python implementation of the DeepStream ALPR (Automatic License Plate Recognition) pipeline for Jetson Orin.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           DeepStream ALPR Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────┐   ┌───────────┐   ┌─────────────┐   ┌──────────┐   ┌───────────────┐  │
│  │ filesrc  │──▶│ qtdemux   │──▶│ h264parse   │──▶│ decoder  │──▶│ nvstreammux   │  │
│  └──────────┘   └───────────┘   └─────────────┘   └──────────┘   └───────┬───────┘  │
│                                                                          │          │
│  ┌───────────────────────────────────────────────────────────────────────▼───────┐  │
│  │                           INFERENCE CHAIN                                      │  │
│  │  ┌─────────────────┐   ┌───────────────┐   ┌─────────────────┐   ┌──────────┐ │  │
│  │  │ PGIE (YOLO11)   │──▶│ nvtracker     │──▶│ SGIE-Plate      │──▶│ SGIE-LPR │ │  │
│  │  │ Vehicle Detect  │   │ NvSORT        │   │ (YOLO11)        │   │ (LPRNet) │ │  │
│  │  │ gie-id=1        │   │               │   │ gie-id=2        │   │ gie-id=3 │ │  │
│  │  └─────────────────┘   └───────────────┘   └─────────────────┘   └──────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┬───────┘  │
│                                                                          │          │
│  ┌───────────────────────────────────────────────────────────────────────▼───────┐  │
│  │  ┌───────────────┐   ┌─────────┐   ┌───────────────────────────────────────┐  │  │
│  │  │ nvvideoconv   │──▶│ nvdsosd │──▶│     OUTPUT (File/Display/RTSP)        │  │  │
│  │  └───────────────┘   └─────────┘   └───────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `alpr_deepstream_python.py` | Main ALPR pipeline with File/RTSP output options |
| `plate_association/` | Module for plate-vehicle association algorithms |
| `plate_association/scoring.py` | Multi-factor scoring for plate-vehicle matching |
| `plate_association/grid_lookup.py` | Spatial grid for O(1) vehicle lookup |
| `plate_association/skip_logic.py` | "Read Once, Skip Later" GPU optimization |

## Requirements

- Jetson Orin with JetPack 6.x
- DeepStream SDK 7.1
- Python 3.x with DeepStream Python bindings (pyds)
- GStreamer Python bindings (gi)

## Installation

### 1. Ensure DeepStream Python bindings are installed

```bash
# Check if pyds is available
python3 -c "import pyds; print('pyds version:', pyds.__version__)"
```

If not installed, install from the DeepStream SDK:

```bash
cd /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/
pip3 install ./pyds-*.whl
```

### 2. Copy config files to project directory

Ensure these config files are in your project directory:
- `config_infer_primary_yolo11.txt`
- `config_infer_secondary_yolo11.txt`
- `config_infer_tertiary_lprnet.txt`

Along with model files:
- `yolo11n.onnx` (or engine file)
- `license-plate-finetune-v1n_320.onnx` (or engine file)
- `us_lprnet_baseline18_deployable.etlt`
- Label files: `labels.txt`, `labels_lpd.txt`, `dict.txt`
- Custom library: `nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`
- LPR library: `libnvdsinfer_custom_impl_lpr.so`

## Usage

### Basic Usage (File Output)

```bash
cd /opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project

# Run with default settings (saves to file)
python3 alpr_deepstream_python.py

# Custom input/output paths
python3 alpr_deepstream_python.py -i /path/to/input.mp4 -o /path/to/output.mp4
```

### RTSP Streaming

```bash
# Enable RTSP streaming
python3 alpr_deepstream_python.py --rtsp

# Custom RTSP settings
python3 alpr_deepstream_python.py --rtsp --rtsp-port 8555 --rtsp-name my-stream

# Live mode with RTSP
python3 alpr_deepstream_python.py --live --rtsp
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i`, `--input` | Input video file path | `sample.mp4` |
| `-o`, `--output` | Output video path | `output_video_python.mp4` |
| `--live` | Enable live mode (sets live-source=1) | disabled |
| `--rtsp` | Enable RTSP streaming output | disabled |
| `--rtsp-port` | RTSP server port | `8554` |
| `--rtsp-name` | RTSP stream name | `alpr-stream` |

## Configuration Mapping

### deepstream_app_config.txt → Python Code

| Config Section | Python Equivalent |
|----------------|-------------------|
| `[source0]` | `filesrc` + `qtdemux` + `h264parse` + `nvv4l2decoder` |
| `[streammux]` | `nvstreammux` with properties |
| `[primary-gie]` | `nvinfer` with `config-file-path` |
| `[tracker]` | `nvtracker` with properties |
| `[secondary-gie0]` | `nvinfer` (plate detector) |
| `[secondary-gie1]` | `nvinfer` (LPR classifier) |
| `[osd]` | `nvdsosd` with properties |
| `[sink0]` (file) | `nvvideoconvert` + `nvv4l2h264enc` + `qtmux` + `filesink` |
| `[sink0]` (rtsp) | `nvvideoconvert` + encoder + `rtph264pay` + RTSP server |

### Key Property Mappings

```python
# Streammux (from [streammux])
streammux.set_property("width", 1280)              # width=1280
streammux.set_property("height", 720)              # height=720
streammux.set_property("batch-size", 1)            # batch-size=1
streammux.set_property("batched-push-timeout", 40000)  # batched-push-timeout=40000
streammux.set_property("live-source", 0)           # live-source=0

# Tracker (from [tracker])
tracker.set_property("tracker-width", 640)         # tracker-width=640
tracker.set_property("tracker-height", 384)        # tracker-height=384
tracker.set_property("ll-lib-file", "...")         # ll-lib-file=...
tracker.set_property("ll-config-file", "...")      # ll-config-file=...

# OSD (from [osd])
osd.set_property("process-mode", 0)                # process-mode=0 (CPU)
osd.set_property("display-text", 1)                # display-text=1
osd.set_property("display-bbox", 1)                # display-bbox=1
```

## Output

### Console Output

```
[ALPR] Starting DeepStream ALPR Pipeline...
[ALPR] Mode: FILE
[ALPR] Output: File
[ALPR] GPU optimization enabled: Read Once, Skip Later
[ALPR] Input:  /path/to/sample.mp4
[ALPR] Output: /path/to/output.mp4
[ALPR] Processing... (Press Ctrl+C to stop)
[Info]150/3600 | Completed: 5 | Skip ratio: 23.4%
[Info]300/3600 | Completed: 8 | Skip ratio: 35.2%
```

### Final Statistics

```
[CLEANUP] Stopping pipeline...
[DONE] Processed 3600 frames in 120.45s (Avg FPS: 29.89)
```

### GPU Optimization Stats

The "Read Once, Skip Later" algorithm shows:
- **Completed**: Number of vehicles whose plates have been read
- **Skip ratio**: Percentage of GPU processing saved

## Troubleshooting

### Common Issues

1. **"Unable to create element"**
   - Ensure DeepStream SDK is properly installed
   - Check GStreamer plugins: `gst-inspect-1.0 <element-name>`

2. **"pyds module not found"**
   - Install DeepStream Python bindings from SDK

3. **"Failed to link elements"**
   - Check element capabilities: `gst-inspect-1.0 <element>`
   - Verify memory types are compatible

4. **"Engine file not found" or model errors**
   - First run will generate engine files (takes time)
   - Ensure ONNX/ETLT model files exist
   - Check custom library paths

5. **Low FPS**
   - Reduce input resolution in streammux
   - Use smaller batch sizes
   - Adjust inference intervals

### Debug Mode

Enable GStreamer debug output:

```bash
export GST_DEBUG=3
python3 alpr_deepstream_python.py
```

For more detailed nvinfer debug:

```bash
export GST_DEBUG=nvinfer:5
python3 alpr_deepstream_python.py
```

## Comparison with deepstream-app

| Feature | deepstream-app | Python Binding |
|---------|----------------|----------------|
| Configuration | Text files | Code + Text files |
| Customization | Limited | Full Python access |
| Metadata access | Via callbacks | Probe functions |
| Output flexibility | Config-based | Programmatic |
| Debugging | GStreamer logs | Python debugging |
| Performance | Optimized | Slightly lower (probe overhead) |

## License

This code is provided for educational purposes. Ensure compliance with NVIDIA DeepStream SDK license terms.
