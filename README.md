# DeepStream ALPR Pipeline

Real-time Automatic License Plate Recognition (ALPR) using NVIDIA DeepStream SDK 7.1, with GPU optimization and high-density traffic handling.

## Features

- **Multi-stage inference**: Vehicle detection (PGIE) → Plate detection (SGIE) → Plate recognition (TGIE/LPRNet)
- **Object tracking**: NvSORT for stable vehicle IDs across frames
- **Plate stabilization**: Majority voting reduces OCR noise
- **Stable vs locked plates**: Stable = current consensus; locked = confirmed, never changes
- **GPU optimization**: "Read Once, Skip Later" — skip SGIE for completed vehicles
- **Parked detection**: Stationary vehicles without plates are skipped (yellow border)
- **High-density heuristics**: Prioritize top N vehicles when many are in frame
- **Plate-vehicle association**: Pipeline parent + spatial fallback; global uniqueness
- **Visual indicators**: Color-coded borders (green=completed, orange=stable, yellow=parked, blue=skipped)

## Pipeline Architecture

```
filesrc → qtdemux → h264parse → decoder → nvstreammux
                                              │
         PGIE (YOLOv11) → nvtracker → Pre-SGIE Probe
         Vehicle Detect      NvSORT    (Skip + Parked + Heuristics)
                                              │
         SGIE (YOLOv11) → TGIE (LPRNet)
         Plate Detect      Plate OCR
                                              │
         nvvideoconv → nvdsosd → Output (File/Display/RTSP)
```

## Project Structure

```
.
├── scripts/
│   ├── alpr_deepstream_python.py    # Main pipeline
│   └── plate_association/           # Association & optimization modules
│       ├── __init__.py
│       ├── scoring.py               # Plate-vehicle geometric scoring
│       ├── grid_lookup.py           # Spatial grid O(1) lookup
│       ├── skip_logic.py            # Skip logic + parked detection
│       ├── heuristics.py            # High-density + stationary detection
│       └── plate_parser.py          # Plate format validation
├── DeepStream-Yolo/
│   ├── config_infer_primary_yolo11.txt
│   ├── config_infer_secondary_yolo11.txt
│   ├── config_infer_tertiary_lprnet.txt
│   └── ...
├── .gitignore
└── README.md
```

## Module Overview

### `scoring.py` — Plate-Vehicle Association

Scores how well a plate bbox matches a vehicle bbox:

- **Bottom edge** (50%): Plate near vehicle bottom (bumper)
- **Horizontal centering** (30%): Plate centered on vehicle
- **Size ratio** (20%): Plate 2–10% of vehicle area

### `grid_lookup.py` — Spatial Grid

- Divides frame into 64×64 px cells
- Vehicles registered in overlapping cells
- O(1) lookup for candidates at plate center

### `skip_logic.py` — GPU Optimization

1. **Completed**: Plate read & confirmed → shrink bbox → SGIE skips
2. **Parked**: Stationary 45 frames, no stable plate → skip SGIE
3. **Heuristics**: High density → process only top N vehicles

### `heuristics.py` — Prioritization & Parked

- **Stationary**: Center movement ≤45 px over last 15 frames
- **Parked threshold**: 45 consecutive stationary frames
- **Priority score**: Size (45%) + Position (30%) + Freshness (25%)
- **Zones**: A=bottom 40%, B=middle 30%, C=top 30%

### `plate_parser.py` — Format Validation

- Cars: 2 letters + 3 digits + 2 letters (e.g. AB123CD)
- Motorcycles: ≥5 chars

## Usage

```bash
cd scripts

# Default (file output)
python alpr_deepstream_python.py

# Custom input/output
python alpr_deepstream_python.py -i /path/to/video.mp4 -o /path/to/out.mp4

# Disable optimizations (for comparison)
python alpr_deepstream_python.py --no-skip --no-heuristics

# Measure inference latency
python alpr_deepstream_python.py --time

# RTSP streaming
python alpr_deepstream_python.py --rtsp
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i`, `--input` | Input video path | config default |
| `-o`, `--output` | Output video path | config default |
| `--live` | Live mode (live-source=1) | off |
| `--rtsp` | RTSP streaming output | off |
| `--no-skip` | Disable skip logic | skip enabled |
| `--no-heuristics` | Disable heuristics | heuristics enabled |
| `--time` | Measure inference latency | off |

## Visual Indicators

| Border | Label | Meaning |
|--------|-------|---------|
| **Green** | `#ID PLATE (Car)` | Completed — plate confirmed |
| **Orange** | `#ID PLATE (Car)` | Stable — has plate, not yet completed |
| **Yellow** | `#ID [PARKED] (Car)` | Parked — stationary, no plate, skipped |
| **Blue** | `[SKIP] #ID (Car)` | Heuristics skip — low priority |
| **Default** | `#ID (Car)` | In progress or failed recognition |

## Key Parameters

### Plate stabilization (`alpr_deepstream_python.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `PLATE_HISTORY_SIZE` | 15 | Frames in voting history |
| `MIN_VOTES_FOR_STABLE` | 5 | Votes for stable plate |
| `MIN_PLATE_LENGTH` | 4 | Min valid plate length |
| `min_confident_readings` | 4 | Readings after stable for completion |

### Parked detection (`skip_logic.py`, `heuristics.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `STATIONARY_THRESHOLD_PX` | 45 | Max center movement (px) to be stationary |
| `POSITION_HISTORY_FRAMES` | 15 | Frames for position history |
| `FRAMES_TO_FLAG_PARKED` | 45 | Consecutive stationary frames → parked |

### Heuristics (`heuristics.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `HIGH_DENSITY_THRESHOLD` | 5 | Vehicles to trigger high-density mode |
| `MAX_PROCESS_PER_FRAME` | 4 | Max vehicles when high density |
| `MAX_PROCESS_NORMAL` | 20 | Max in normal mode |

## Plate Association Flow

1. **Parent**: Use `plate_meta.parent.object_id` when available (pipeline)
2. **Fallback**: Spatial grid + `PlateVehicleScorer` when parent is None
3. **Duplicate resolution**: One plate text per vehicle; prefer existing owner
4. **Global uniqueness**: Reject plate if another vehicle already has it

## Latency Measurement (`--time`)

- Sink probe records start time when buffer enters each model
- Src probe computes latency when buffer exits (FIFO pairing)
- Reports: PGIE, SGIE Plate, SGIE LPR, total inference latency

## Technologies

| Category | Technology |
|----------|------------|
| SDK | NVIDIA DeepStream 7.1 |
| Framework | GStreamer |
| Inference | TensorRT, CUDA |
| Detection | YOLOv11 |
| Recognition | LPRNet |
| Tracking | NvSORT |
