# DeepStream ALPR Pipeline - Python Binding

Real-time Automatic License Plate Recognition (ALPR) system built with NVIDIA DeepStream SDK 7.1, featuring intelligent GPU optimization and high-density traffic handling.

## Features

- **Multi-Stage Inference Pipeline**: Vehicle detection → Plate detection → Plate recognition
- **Real-time Object Tracking**: NvSORT tracker for stable vehicle tracking across frames
- **Voting-based Plate Stabilization**: Reduces OCR noise through majority voting
- **GPU Optimization**: "Read Once, Skip Later" - skips inference for completed vehicles
- **High-Density Heuristics**: Smart vehicle prioritization when traffic is congested
- **Multiple Output Modes**: File output, RTSP streaming, or display
- **Visual Status Indicators**: Color-coded bounding boxes for processing status

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
│  │                                                                                │  │
│  │  ┌─────────────────┐   ┌───────────────┐   ┌─────────────────┐                │  │
│  │  │ PGIE (YOLOv11)  │──▶│ nvtracker     │──▶│ Pre-SGIE Probe  │                │  │
│  │  │ Vehicle Detect  │   │ NvSORT        │   │ (Skip Logic +   │                │  │
│  │  │ gie-id=1        │   │               │   │  Heuristics)    │                │  │
│  │  └─────────────────┘   └───────────────┘   └────────┬────────┘                │  │
│  │                                                     │                          │  │
│  │                              ┌──────────────────────▼──────────────────────┐   │  │
│  │                              │  SGIE (YOLOv11)  │  TGIE (LPRNet)          │   │  │
│  │                              │  Plate Detection │  Plate Recognition      │   │  │
│  │                              │  gie-id=2        │  gie-id=3               │   │  │
│  │                              └──────────────────────────────────────────────┘   │  │
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

## Project Structure

```
alpr_project/
├── scripts/
│   ├── alpr_deepstream_python.py      # Main ALPR pipeline
│   └── plate_association/             # Optimization modules
│       ├── __init__.py
│       ├── scoring.py                 # Plate-vehicle association scoring
│       ├── grid_lookup.py             # Spatial grid for O(1) lookup
│       ├── skip_logic.py              # "Read Once, Skip Later" optimization
│       └── heuristics.py              # High-density traffic handling
│
└── DeepStream-Yolo/
    ├── config_infer_primary_yolo11.txt    # PGIE config (vehicle detection)
    ├── config_infer_secondary_yolo11.txt  # SGIE config (plate detection)
    ├── config_infer_tertiary_lprnet.txt   # TGIE config (plate recognition)
    ├── labels.txt                         # Vehicle class labels
    ├── labels_lpd.txt                     # License plate labels
    ├── dict.txt                           # LPR character dictionary
    └── nvdsinfer_custom_impl_Yolo/        # Custom YOLO parser library
        └── libnvdsinfer_custom_impl_Yolo.so
```

## Technologies Used

| Category | Technology |
|----------|------------|
| **SDK** | NVIDIA DeepStream 7.1 |
| **Framework** | GStreamer |
| **Inference** | TensorRT, CUDA |
| **Detection** | YOLOv8/YOLOv11 |
| **Recognition** | LPRNet |
| **Tracking** | NvSORT |
| **Language** | Python 3.x (pyds bindings) |

## Usage

### Basic Usage

```bash
cd /opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project/scripts

# Run with default settings
python3 alpr_deepstream_python.py

# Custom input/output
python3 alpr_deepstream_python.py -i /path/to/video.mp4 -o /path/to/output.mp4
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i`, `--input` | Input video file path | `sample.mp4` |
| `-o`, `--output` | Output video path | `output_video_python.mp4` |
| `--live` | Enable live mode (live-source=1) | disabled |
| `--rtsp` | Enable RTSP streaming output | disabled |
| `--rtsp-port` | RTSP server port | `8554` |
| `--rtsp-name` | RTSP stream name | `alpr-stream` |
| `--no-skip` | Disable GPU optimization | enabled |
| `--no-heuristics` | Disable high-density heuristics | enabled |

### RTSP Streaming

```bash
# Enable RTSP streaming
python3 alpr_deepstream_python.py --rtsp

# Custom RTSP settings
python3 alpr_deepstream_python.py --rtsp --rtsp-port 8555 --rtsp-name my-stream

# View stream: rtsp://<IP>:8555/my-stream
```

## GPU Optimization Features

### 1. Read Once, Skip Later

Once a vehicle's plate is successfully read and confirmed (via voting), the system skips further inference for that vehicle.

**How it works:**
1. Vehicle plate is read multiple times (voting)
2. When plate reaches `MIN_VOTES_FOR_STABLE` (5), it's "locked"
3. After `COMPLETION_THRESHOLD` (8) consistent readings, vehicle is "completed"
4. Completed vehicles' bboxes are shrunk below SGIE threshold (8×8 < 30×30)
5. SGIE automatically skips these vehicles → GPU saved

### 2. High-Density Heuristics

When many vehicles are present, the system intelligently prioritizes which vehicles to process.

**Activation:**
- Triggers when non-completed vehicles > `HIGH_DENSITY_THRESHOLD` (10)
- Processes only top `MAX_PROCESS_PER_FRAME` (8) vehicles

**Priority Scoring:**
```
Score = 0.45 × SIZE + 0.30 × POSITION + 0.25 × FRESHNESS

SIZE:      Larger vehicles (closer) score higher
POSITION:  Center-bottom of frame scores higher
FRESHNESS: Vehicles waiting long get priority boost (fairness)
```

**Priority Zones:**
```
┌────────────────────────────┐
│      ZONE C (Top 30%)      │  ← Low priority (far away)
├────────────────────────────┤
│     ZONE B (Middle 30%)    │  ← Medium priority
├────────────────────────────┤
│    ZONE A (Bottom 40%)     │  ← High priority (close)
└────────────────────────────┘
```

## Visual Indicators

The output video uses color-coded bounding boxes:

| Color | Border | Status | Meaning |
|-------|--------|--------|---------|
| **Red** | Default | Processing | Plate not yet read |
| **Blue** | Thick | `[SKIP]` | Skipped by heuristics (low priority) |
| **Green** | Thick | Completed | Plate successfully read & confirmed |

## Output Format

### Console Progress

```
[Info]300/3600 | Plates found: 8 | Vehicles completed: 5
[Info]600/3600 | Plates found: 12 | Vehicles completed: 8
```

## Configuration

### Adjustable Parameters

In `alpr_deepstream_python.py`:
```python
PLATE_HISTORY_SIZE = 15       # Frames to keep in voting history
MIN_VOTES_FOR_STABLE = 5      # Votes needed for stable plate
MIN_PLATE_LENGTH = 4          # Minimum valid plate length
COMPLETION_THRESHOLD = 8      # Readings to mark vehicle complete
```

In `plate_association/heuristics.py`:
```python
HIGH_DENSITY_THRESHOLD = 10   # Vehicles to trigger heuristics
MAX_PROCESS_PER_FRAME = 8     # Max vehicles to process when active
WEIGHT_SIZE = 0.45            # Weight for vehicle size
WEIGHT_POSITION = 0.30        # Weight for position
WEIGHT_FRESHNESS = 0.25       # Weight for waiting time
```