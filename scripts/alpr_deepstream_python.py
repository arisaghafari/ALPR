import sys
import os
import cv2

if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

# Disable EGL-related warnings in headless mode
os.environ['EGL_PLATFORM'] = 'surfaceless'

# Add the script directory to path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Ensure GStreamer plugin paths are set for DeepStream Docker
if os.path.exists("/opt/nvidia/deepstream/deepstream/lib/gst-plugins"):
    gst_plugin_path = os.environ.get("GST_PLUGIN_PATH", "")
    ds_plugin_path = "/opt/nvidia/deepstream/deepstream/lib/gst-plugins"
    if ds_plugin_path not in gst_plugin_path:
        os.environ["GST_PLUGIN_PATH"] = f"{ds_plugin_path}:{gst_plugin_path}"

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

import pyds
from pathlib import Path

# Import plate-vehicle association modules
from plate_association import (
    PlateVehicleScorer, SpatialGrid, SkipLogicManager, 
    create_pre_sgie_probe, HighDensityHeuristics,
    get_heuristics_skipped, is_heuristics_active,
    get_parked_vehicles, cleanup_parked_counts
)
from plate_association.plate_parser import is_valid_plate_format

# ==============================================================================
# Configuration - Adjust paths as needed
# ==============================================================================
CONFIG_DIR = "/opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project"

INPUT_VIDEO = f"{CONFIG_DIR}/input_videos/output_clip_fixed.mp4"
OUTPUT_VIDEO = f"{CONFIG_DIR}/output_videos/output_video_python.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
TOTAL_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

PGIE_CONFIG = f"{CONFIG_DIR}/DeepStream-Yolo/config_infer_primary_yolo11.txt"
SGIE_PLATE_DETECTOR_CONFIG = f"{CONFIG_DIR}/DeepStream-Yolo/config_infer_secondary_yolo11.txt"
SGIE_LPR_CONFIG = f"{CONFIG_DIR}/DeepStream-Yolo/config_infer_tertiary_lprnet.txt"
TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvSORT.yml"
#TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

# Path to main DeepStream app config (to read settings from)
APP_CONFIG = f"{CONFIG_DIR}/DeepStream-Yolo/deepstream_app_config.txt"

# ==============================================================================
# RTSP/Live Mode Configuration (defaults - can be overridden by args)
# ==============================================================================
LIVE_MODE = False       # --live to enable (sets live-source=1 on streammux)
ENABLE_RTSP = False     # --rtsp to enable
RTSP_PORT = 8554
RTSP_STREAM_NAME = "alpr-stream"

# ==============================================================================
# GPU Optimization Control
# ==============================================================================
ENABLE_SKIP_LOGIC = True  # --no-skip to disable for comparison
ENABLE_TIMING = False    # --time to enable latency measurement

# ==============================================================================
# Timing / Latency Measurement (--time)
# ==============================================================================
from collections import deque
_timing_start_queue = {
    'pgie': deque(maxlen=100),
    'sgie_plate': deque(maxlen=100),
    'sgie_lpr': deque(maxlen=100),
}
_timing_latency = {
    'pgie': {'total_ms': 0.0, 'count': 0},
    'sgie_plate': {'total_ms': 0.0, 'count': 0},
    'sgie_lpr': {'total_ms': 0.0, 'count': 0},
}

def _make_timing_sink_probe(model_name):
    """Probe on sink pad: record start time when buffer enters."""
    def probe(pad, info, u_data):
        if not ENABLE_TIMING:
            return Gst.PadProbeReturn.OK
        import time
        _timing_start_queue[model_name].append(time.perf_counter())
        return Gst.PadProbeReturn.OK
    return probe

def _make_timing_src_probe(model_name):
    """Probe on src pad: compute latency when buffer exits (FIFO with sink)."""
    def probe(pad, info, u_data):
        if not ENABLE_TIMING:
            return Gst.PadProbeReturn.OK
        import time
        q = _timing_start_queue[model_name]
        if q:
            start = q.popleft()
            latency_ms = (time.perf_counter() - start) * 1000
            _timing_latency[model_name]['total_ms'] += latency_ms
            _timing_latency[model_name]['count'] += 1
        return Gst.PadProbeReturn.OK
    return probe

def _get_timing_stats():
    """Get average latency (ms) per model."""
    return {
        name: (d['total_ms'] / d['count'] if d['count'] > 0 else 0.0)
        for name, d in _timing_latency.items()
    }

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DeepStream ALPR Pipeline - License Plate Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Save to file (default)
  python %(prog)s
  
  # Live mode with RTSP streaming
  python %(prog)s --live --rtsp
  
  # Live mode, save to file
  python %(prog)s --live
  
  # Custom input/output
  python %(prog)s -i /path/to/video.mp4 -o /path/to/output.mp4
  
  # Custom RTSP settings
  python %(prog)s --live --rtsp --rtsp-port 8555 --rtsp-name my-stream
        '''
    )
    
    # Mode options
    parser.add_argument('--live', action='store_true',
                        help='Enable live mode (sets live-source=1 on streammux)')
    parser.add_argument('--rtsp', action='store_true',
                        help='Enable RTSP streaming output (instead of file)')
    
    # Input/Output options
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Input video file path')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output video file path')
    
    # RTSP options
    parser.add_argument('--rtsp-port', type=int, default=8554,
                        help='RTSP server port (default: 8554)')
    parser.add_argument('--rtsp-name', type=str, default='alpr-stream',
                        help='RTSP stream name (default: alpr-stream)')
    
    # Optimization options
    parser.add_argument('--no-skip', action='store_true',
                        help='Disable "Read Once, Skip Later" optimization (for comparison)')
    parser.add_argument('--no-heuristics', action='store_true',
                        help='Disable high-density heuristics (for comparison)')
    
    # Timing options
    parser.add_argument('--time', action='store_true',
                        help='Measure latency of each inference model and pipeline FPS')
    
    return parser.parse_args()

def apply_arguments(args):
    """Apply command line arguments to global config."""
    global INPUT_VIDEO, OUTPUT_VIDEO, LIVE_MODE, ENABLE_RTSP, RTSP_PORT, RTSP_STREAM_NAME
    global ENABLE_SKIP_LOGIC, ENABLE_HEURISTICS, ENABLE_TIMING, TOTAL_FRAME
    
    LIVE_MODE = args.live
    ENABLE_RTSP = args.rtsp
    RTSP_PORT = args.rtsp_port
    RTSP_STREAM_NAME = args.rtsp_name
    ENABLE_SKIP_LOGIC = not args.no_skip  # --no-skip disables it
    ENABLE_HEURISTICS = not args.no_heuristics  # --no-heuristics disables it
    ENABLE_TIMING = args.time
    
    if args.input:
        INPUT_VIDEO = args.input
        cap = cv2.VideoCapture(INPUT_VIDEO)
        TOTAL_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.output:
        OUTPUT_VIDEO = args.output

def parse_app_config(config_path):
    """Parse deepstream_app_config.txt to extract settings."""
    config = {}
    current_section = None
    
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1]
                    config[current_section] = {}
                elif '=' in line and current_section:
                    key, value = line.split('=', 1)
                    config[current_section][key.strip()] = value.strip()
    except Exception as e:
        print(f"[WARNING] Could not parse {config_path}: {e}")
        # Return defaults if parsing fails
        return None
    
    return config


def get_config_values():
    """Get configuration values from deepstream_app_config.txt."""
    config = parse_app_config(APP_CONFIG)
    
    # Default values (fallback if config parsing fails)
    # Updated to match current deepstream_app_config.txt
    values = {
        'muxer_width': 1920,
        'muxer_height': 1080,
        'muxer_batch_timeout': 40000,
        'muxer_buffer_pool_size': 2,
        'muxer_nvbuf_memory_type': 0,
        'enable_padding': 1,  # Preserve aspect ratio
        'tracker_width': 640,
        'tracker_height': 384,
        'bitrate': 4000000,
        'iframe_interval': 30,
        'scaling_compute_hw': 1
    }
    
    if config:
        # Read from streammux section
        if 'streammux' in config:
            values['muxer_width'] = int(config['streammux'].get('width', 1920))
            values['muxer_height'] = int(config['streammux'].get('height', 1080))
            values['muxer_batch_timeout'] = int(config['streammux'].get('batched-push-timeout', 40000))
            values['muxer_buffer_pool_size'] = int(config['streammux'].get('buffer-pool-size', 2))
            values['enable_padding'] = int(config['streammux'].get('enable-padding', 1))
            values['scaling_compute_hw'] = int(config['streammux'].get('compute-hw', 1))
            values['muxer_nvbuf_memory_type'] = int(config['streammux'].get('nvbuf-memory-type', 0))
        
        # Read from tracker section
        if 'tracker' in config:
            values['tracker_width'] = int(config['tracker'].get('tracker-width', 640))
            values['tracker_height'] = int(config['tracker'].get('tracker-height', 384))
        
        # Read from sink0 section
        if 'sink0' in config:
            values['bitrate'] = int(config['sink0'].get('bitrate', 4000000))
            values['iframe_interval'] = int(config['sink0'].get('iframeinterval', 30))
    
    return values

# Load config values from deepstream_app_config.txt
CFG = get_config_values()

# ==============================================================================
# Global Variables for Performance Measurement
# ==============================================================================
frame_count = 0
start_time = None

# ==============================================================================
# Plate Recognition Stabilization - Per Vehicle
# ==============================================================================
from collections import defaultdict, Counter

# Configuration for stabilization
PLATE_HISTORY_SIZE = 15       # Number of frames to keep history
MIN_VOTES_FOR_STABLE = 5      # Minimum votes needed to consider a plate "stable"
MIN_PLATE_LENGTH = 4          # Minimum plate text length to be valid

# Store plate recognition history PER VEHICLE (using vehicle's track ID)
# Key: vehicle_track_id (from parent object)
# Value: list of recent plate texts for this specific vehicle
vehicle_plate_history = defaultdict(list)

# Store the stable/final plate text PER VEHICLE
# Key: vehicle_track_id
# Value: stable plate text for this vehicle
vehicle_stable_plates = {}


# LOCKED plate text - once set, never changes (for accurate completion)
# Key: vehicle_track_id, Value: locked plate text
vehicle_locked_plates = {}

# Completion count - only counts exact matches to locked text
# Key: vehicle_track_id, Value: count of readings matching locked text
vehicle_completion_count = defaultdict(int)

# Track TOTAL readings per vehicle (for statistics)
# Key: vehicle_track_id, Value: total count of plate readings
vehicle_total_readings = defaultdict(int)

# Track last frame each vehicle was seen (for cleanup)
vehicle_last_seen = {}

# Vehicle type per ID (for final summary display)
vehicle_type_by_id = {}

# CUMULATIVE stats (not affected by cleanup)
# Key: vehicle_id, Value: plate_text (ensures one plate per vehicle)
total_plates_by_vehicle = {}  # Stable plates (high confidence)
total_partial_plates = {}     # Best-effort plates (didn't reach stable threshold)
total_vehicles_completed_ever = set()  # All vehicle IDs ever completed

def get_stable_plate_for_vehicle(vehicle_id, new_plate_text, current_frame):
    """
    Get stable plate text for a specific vehicle using majority voting.
    
    Args:
        vehicle_id: Track ID of the parent vehicle
        new_plate_text: Newly recognized plate text
        current_frame: Current frame number
    
    Returns:
        Stable plate text (most common in recent history) for THIS vehicle
    """
    global vehicle_plate_history, vehicle_stable_plates, vehicle_locked_plates, vehicle_last_seen
    
    # Update last seen frame
    vehicle_last_seen[vehicle_id] = current_frame
    
    # Global uniqueness: same plate text cannot belong to multiple vehicles.
    # If another vehicle already has this plate (stable/locked), reject for this vehicle.
    if new_plate_text and len(new_plate_text) >= MIN_PLATE_LENGTH:
        for vid, plate in vehicle_stable_plates.items():
            if vid != vehicle_id and plate == new_plate_text:
                return vehicle_stable_plates.get(vehicle_id, vehicle_locked_plates.get(vehicle_id, ""))
        for vid, plate in vehicle_locked_plates.items():
            if vid != vehicle_id and plate == new_plate_text:
                return vehicle_stable_plates.get(vehicle_id, vehicle_locked_plates.get(vehicle_id, ""))
    
    # Add new recognition to this vehicle's history (only if valid length)
    if new_plate_text and len(new_plate_text) >= MIN_PLATE_LENGTH:
        vehicle_plate_history[vehicle_id].append(new_plate_text)
        
        # Keep only recent history
        if len(vehicle_plate_history[vehicle_id]) > PLATE_HISTORY_SIZE:
            vehicle_plate_history[vehicle_id] = vehicle_plate_history[vehicle_id][-PLATE_HISTORY_SIZE:]
    
    # Get history for this vehicle
    history = vehicle_plate_history.get(vehicle_id, [])
    
    if not history:
        return vehicle_stable_plates.get(vehicle_id, "")
    
    # Count occurrences of each plate text for this vehicle
    counter = Counter(history)
    most_common_plate, count = counter.most_common(1)[0]
    
    # Only update stable plate if we have enough consistent votes and valid length
    if count >= MIN_VOTES_FOR_STABLE and len(most_common_plate) >= MIN_PLATE_LENGTH:
        vehicle_stable_plates[vehicle_id] = most_common_plate
        # Track cumulative stats (one plate per vehicle - overwrites if changed)
        total_plates_by_vehicle[vehicle_id] = most_common_plate
        return most_common_plate
    elif vehicle_id in vehicle_stable_plates:
        # Keep previous stable plate if not enough new votes
        return vehicle_stable_plates[vehicle_id]
    else:
        # Return most common even if below threshold
        return most_common_plate

def cleanup_old_vehicles(current_frame, max_frames_missing=60):
    """Remove history for vehicles that haven't been seen recently.
    Saves partial plates for vehicles that didn't reach stable threshold."""
    global vehicle_plate_history, vehicle_stable_plates, vehicle_last_seen
    global total_partial_plates
    
    # Find vehicles not seen recently
    vehicles_to_remove = []
    for vehicle_id, last_frame in vehicle_last_seen.items():
        if current_frame - last_frame > max_frames_missing:
            vehicles_to_remove.append(vehicle_id)
    
    # Remove old vehicle data, but save partial plates
    for vehicle_id in vehicles_to_remove:
        # If vehicle didn't reach stable threshold, save best-effort plate
        if vehicle_id not in total_plates_by_vehicle:
            history = vehicle_plate_history.get(vehicle_id, [])
            if history:
                # Get most common plate from history (even if below threshold)
                counter = Counter(history)
                best_plate, count = counter.most_common(1)[0]
                if len(best_plate) >= MIN_PLATE_LENGTH:
                    total_partial_plates[vehicle_id] = (best_plate, count)
        
        vehicle_plate_history.pop(vehicle_id, None)
        vehicle_stable_plates.pop(vehicle_id, None)
        vehicle_last_seen.pop(vehicle_id, None)

def final_cleanup_save_all():
    """Save ALL remaining vehicles as partial plates at end of video.
    This ensures no vehicles are lost when the video ends."""
    global vehicle_plate_history, total_plates_by_vehicle, total_partial_plates
    
    saved_count = 0
    for vehicle_id, history in vehicle_plate_history.items():
        # Skip if already saved as stable plate
        if vehicle_id in total_plates_by_vehicle:
            continue
        
        # Skip if already saved as partial plate
        if vehicle_id in total_partial_plates:
            continue
        
        if history:
            counter = Counter(history)
            best_plate, count = counter.most_common(1)[0]
            if len(best_plate) >= MIN_PLATE_LENGTH:
                total_partial_plates[vehicle_id] = (best_plate, count)
                saved_count += 1
    
    if saved_count > 0:
        print(f"[FINAL] Saved {saved_count} remaining vehicles as partial plates")

def bus_call(bus, message, loop):
    """Handle GStreamer bus messages."""
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[ERROR] {err}: {debug}")
        loop.quit()
    return True

# ==============================================================================
# Plate-Vehicle Association Components (from plate_association module)
# ==============================================================================
spatial_grid = SpatialGrid(cell_size=64)
plate_scorer = PlateVehicleScorer()

# ==============================================================================
# READ ONCE, SKIP LATER - GPU Optimization (from plate_association.skip_logic)
# ==============================================================================
# NOTE: min_confident_readings should be >= MIN_VOTES_FOR_STABLE
# Stable = 5 consistent readings, Completed = stable + 4 more confirmations
skip_manager = SkipLogicManager(
    min_confident_readings=4,       # Additional readings needed AFTER stable (total: 5+4=9)
    min_plate_length=MIN_PLATE_LENGTH,  # Same as stabilization check
    max_frames_missing=60           # Cleanup threshold (lower list)
)

# High-density traffic heuristics (filter vehicles when too many in frame)
# Enable/disable with command line flag --no-heuristics
ENABLE_HEURISTICS = True
heuristics_manager = HighDensityHeuristics(
    frame_width=1920,   # Will be updated from config
    frame_height=1080
)
# Note: SGIE min size (50) is configured in config_infer_secondary_yolo11.txt

# Create the pre-SGIE probe function
# Pass heuristics_manager for worst-case high-density filtering (actual GPU savings!)
pre_sgie_probe = create_pre_sgie_probe(
    skip_manager, 
    heuristics_manager=heuristics_manager if ENABLE_HEURISTICS else None,
    completed_vehicles_set=total_vehicles_completed_ever,
    vehicles_with_stable_plate_ref=vehicle_stable_plates
)

# Cached vehicle labels from PGIE (primary model) - loaded from labelfile
_vehicle_labels_cache = None

def _get_vehicle_labels():
    """Load vehicle class labels from PGIE config labelfile (labels_vehicles.txt)."""
    global _vehicle_labels_cache
    if _vehicle_labels_cache is not None:
        return _vehicle_labels_cache
    labels_path = os.path.join(os.path.dirname(PGIE_CONFIG), "labels_vehicles.txt")
    try:
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                _vehicle_labels_cache = [line.strip() for line in f if line.strip()]
        else:
            _vehicle_labels_cache = []
    except Exception:
        _vehicle_labels_cache = []
    return _vehicle_labels_cache

def _get_vehicle_type(obj_meta):
    """
    Get vehicle type string from primary model (PGIE) output.
    Uses obj_label if set, else class_id + labels file.
    """
    try:
        # obj_label is populated by DeepStream from the labelfile
        if hasattr(obj_meta, "obj_label") and obj_meta.obj_label:
            label = obj_meta.obj_label
            if isinstance(label, bytes):
                label = label.decode("utf-8", errors="replace").strip()
            elif isinstance(label, str):
                label = label.strip()
            if label:
                return label
    except Exception:
        pass
    # Fallback: use class_id with labels file
    try:
        labels = _get_vehicle_labels()
        if labels and 0 <= obj_meta.class_id < len(labels):
            return labels[obj_meta.class_id]
    except Exception:
        pass
    return "vehicle"


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe function using modular plate-vehicle association.
    Uses SpatialGrid for O(1) lookup and PlateVehicleScorer for multi-factor matching.
    
    Integrates with "Read Once, Skip Later" optimization:
    - Restores bboxes that were shrunk by pre_sgie_probe
    - Marks vehicles as completed after confident readings
    - Displays stored plates for completed vehicles
    """
    global frame_count, start_time
    import time
    
    if start_time is None:
        start_time = time.time()
    
    frame_count += 1
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_num = frame_meta.frame_num
        except StopIteration:
            break
        
        # Clear spatial grid for this frame
        spatial_grid.clear()
        plates_to_process = []
        vehicle_types_this_frame = {}  # vehicle_id -> vehicle_type (for plate format validation)
        
        # Single pass: build spatial grid, collect plates, restore completed bboxes
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                # Vehicles (gie-unique-id=1): Add to spatial grid
                if obj_meta.unique_component_id == 1:
                    vehicle_id = obj_meta.object_id
                    vehicle_type = _get_vehicle_type(obj_meta)
                    
                    # Check if this vehicle was SKIPPED by heuristics or PARKED
                    heuristics_skipped = get_heuristics_skipped(frame_num)
                    is_skipped_by_heuristics = vehicle_id in heuristics_skipped
                    parked_vehicles = get_parked_vehicles(frame_num)
                    is_parked = vehicle_id in parked_vehicles
                    
                    # Skip logic handling (only when enabled)
                    if ENABLE_SKIP_LOGIC:
                        # Restore bbox if it was shrunk by pre_sgie_probe
                        skip_manager.restore_bbox(obj_meta, frame_num)
                        
                        # VISUAL: Parked = YELLOW | Heuristics = BLUE | Completed = GREEN | Stable (not completed) = ORANGE | No stable = default
                        has_stable = vehicle_id in vehicle_stable_plates or vehicle_id in vehicle_locked_plates
                        if is_parked:
                            obj_meta.text_params.display_text = f"#{vehicle_id} [PARKED] ({vehicle_type})"
                            obj_meta.text_params.set_bg_clr = 1
                            obj_meta.text_params.text_bg_clr.red = 1.0
                            obj_meta.text_params.text_bg_clr.green = 1.0
                            obj_meta.text_params.text_bg_clr.blue = 0.0
                            obj_meta.text_params.text_bg_clr.alpha = 0.8
                            obj_meta.rect_params.border_color.red = 1.0
                            obj_meta.rect_params.border_color.green = 1.0
                            obj_meta.rect_params.border_color.blue = 0.0
                            obj_meta.rect_params.border_color.alpha = 1.0
                            obj_meta.rect_params.border_width = 4
                        elif is_skipped_by_heuristics:
                            obj_meta.text_params.display_text = f"[SKIP] #{vehicle_id} ({vehicle_type})"
                            obj_meta.text_params.set_bg_clr = 1
                            obj_meta.text_params.text_bg_clr.red = 0.0
                            obj_meta.text_params.text_bg_clr.green = 0.0
                            obj_meta.text_params.text_bg_clr.blue = 1.0
                            obj_meta.text_params.text_bg_clr.alpha = 1.0
                            # BLUE border (very visible)
                            obj_meta.rect_params.border_color.red = 0.0
                            obj_meta.rect_params.border_color.green = 0.0
                            obj_meta.rect_params.border_color.blue = 1.0
                            obj_meta.rect_params.border_color.alpha = 1.0
                            obj_meta.rect_params.border_width = 4  # Thicker border
                        # For COMPLETED vehicles: GREEN background + GREEN border + locked plate
                        elif skip_manager.is_completed(vehicle_id):
                            # Use LOCKED plate (confirmed text) for completed vehicles
                            stored_plate = vehicle_locked_plates.get(vehicle_id, vehicle_stable_plates.get(vehicle_id, ""))
                            if stored_plate:
                                obj_meta.text_params.display_text = f"#{vehicle_id} {stored_plate} ({vehicle_type})"
                                obj_meta.text_params.set_bg_clr = 1
                                obj_meta.text_params.text_bg_clr.red = 0.0
                                obj_meta.text_params.text_bg_clr.green = 0.6
                                obj_meta.text_params.text_bg_clr.blue = 0.0
                                obj_meta.text_params.text_bg_clr.alpha = 0.8
                            else:
                                obj_meta.text_params.display_text = f"#{vehicle_id} ({vehicle_type})"
                                obj_meta.text_params.set_bg_clr = 1
                            # GREEN border (completed)
                            obj_meta.rect_params.border_color.red = 0.0
                            obj_meta.rect_params.border_color.green = 0.6
                            obj_meta.rect_params.border_color.blue = 0.0
                            obj_meta.rect_params.border_color.alpha = 1.0
                            obj_meta.rect_params.border_width = 4
                        elif has_stable:
                            # Stable plate (has text) but not yet completed - ORANGE border
                            stored_plate = vehicle_locked_plates.get(vehicle_id, vehicle_stable_plates.get(vehicle_id, ""))
                            obj_meta.text_params.display_text = f"#{vehicle_id} {stored_plate} ({vehicle_type})"
                            obj_meta.text_params.set_bg_clr = 1
                            obj_meta.text_params.text_bg_clr.red = 1.0
                            obj_meta.text_params.text_bg_clr.green = 0.5
                            obj_meta.text_params.text_bg_clr.blue = 0.0
                            obj_meta.text_params.text_bg_clr.alpha = 0.8
                            obj_meta.rect_params.border_color.red = 1.0
                            obj_meta.rect_params.border_color.green = 0.5
                            obj_meta.rect_params.border_color.blue = 0.0
                            obj_meta.rect_params.border_color.alpha = 1.0
                            obj_meta.rect_params.border_width = 4
                        else:
                            # No stable plate - failed or in progress (default border)
                            obj_meta.text_params.display_text = f"#{vehicle_id} ({vehicle_type})"
                            obj_meta.text_params.set_bg_clr = 1
                    else:
                        # Skip logic disabled - show vehicle ID
                        obj_meta.text_params.display_text = f"#{vehicle_id} ({vehicle_type})"
                        obj_meta.text_params.set_bg_clr = 1
                    
                    spatial_grid.add_vehicle(vehicle_id, obj_meta.rect_params)
                    vehicle_types_this_frame[vehicle_id] = vehicle_type
                    vehicle_type_by_id[vehicle_id] = vehicle_type
                
                # Plates (gie-unique-id=2): Collect for processing
                elif obj_meta.unique_component_id == 2:
                    obj_meta.text_params.display_text = " "
                    obj_meta.text_params.set_bg_clr = 0
                    plates_to_process.append(obj_meta)
                
            except Exception:
                pass
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        prioritized_vehicle_ids = None  # Filtering done in pre-SGIE probe
        
        # Duplicate plate resolution: one plate text per vehicle; prefer existing owner
        plate_assignments = []  # (plate_meta, plate_text, parent_id, score)
        
        # First pass: collect plate-to-vehicle assignments (prefer pipeline parent)
        for plate_meta in plates_to_process:
            try:
                parent_id = -1
                score = 0.0
                # Use pipeline parent when available (correct association from SGIE crop)
                if hasattr(plate_meta, 'parent') and plate_meta.parent is not None:
                    try:
                        parent_id = plate_meta.parent.object_id
                        score = 1.0  # Trust pipeline when parent is set
                    except Exception:
                        pass
                # Fallback to spatial lookup when parent is None
                if parent_id < 0:
                    plate_rect = plate_meta.rect_params
                    plate_cx = plate_rect.left + plate_rect.width / 2
                    plate_cy = plate_rect.top + plate_rect.height / 2
                    candidates = spatial_grid.get_candidate_vehicles(plate_cx, plate_cy)
                    if candidates:
                        parent_id, score = plate_scorer.find_best_vehicle(
                            plate_rect, candidates, min_score=0.1
                        )
                
                if prioritized_vehicle_ids is not None and parent_id >= 0:
                    if parent_id not in prioritized_vehicle_ids:
                        continue
                
                # Get LPR text from classifier (gie-unique-id=3)
                plate_text = None
                l_cls = plate_meta.classifier_meta_list
                while l_cls is not None:
                    try:
                        cls_meta = pyds.NvDsClassifierMeta.cast(l_cls.data)
                        if cls_meta.unique_component_id == 3:
                            l_lbl = cls_meta.label_info_list
                            if l_lbl is not None:
                                try:
                                    lbl = pyds.NvDsLabelInfo.cast(l_lbl.data)
                                    if lbl.result_label:
                                        plate_text = lbl.result_label
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    try:
                        l_cls = l_cls.next
                    except StopIteration:
                        break
                
                if plate_text and len(plate_text) >= MIN_PLATE_LENGTH and parent_id >= 0:
                    # Format validation: non-Moto vehicles require 2L+3D+2L
                    vehicle_type = vehicle_types_this_frame.get(parent_id, "")
                    if not is_valid_plate_format(plate_text, vehicle_type):
                        continue  # Reject invalid format
                    plate_assignments.append((plate_meta, plate_text, parent_id, score))
            except Exception:
                pass
        
        # Resolve duplicate plate text: same physical plate can't be on two vehicles.
        # Keep only the assignment with best effective score per plate text.
        # Effective score: prefer vehicle that ALREADY owns this plate (prevents stealing).
        # If vehicle has this plate in stable/locked history, add 1000 to score.
        def _effective_score(plate_text, parent_id, spatial_score):
            if parent_id < 0:
                return spatial_score
            existing = vehicle_stable_plates.get(parent_id) or vehicle_locked_plates.get(parent_id)
            if existing and existing == plate_text:
                return 1000.0 + spatial_score  # Strongly prefer existing owner
            return spatial_score
        
        best_by_text = {}  # plate_text -> (plate_meta, parent_id, effective_score)
        for plate_meta, plate_text, parent_id, score in plate_assignments:
            eff = _effective_score(plate_text, parent_id, score)
            if plate_text not in best_by_text or eff > best_by_text[plate_text][2]:
                best_by_text[plate_text] = (plate_meta, parent_id, eff)
        
        # Explicitly suppress loser plates - don't show LPR text on duplicate detections
        winner_plate_metas = {best_by_text[pt][0] for pt in best_by_text}
        for plate_meta in plates_to_process:
            if plate_meta not in winner_plate_metas:
                plate_meta.text_params.display_text = " "
                plate_meta.text_params.set_bg_clr = 0
        
        # Second pass: process only the winning assignments
        for plate_text, (plate_meta, parent_id, score) in best_by_text.items():
            try:
                # Check if this vehicle is already completed (plate already known)
                if ENABLE_SKIP_LOGIC and skip_manager.is_completed(parent_id):
                    display_text = vehicle_stable_plates.get(parent_id, "")
                else:
                    tracking_id = parent_id
                    if tracking_id != 18446744073709551615 and tracking_id < 1000:
                        vehicle_total_readings[tracking_id] += 1
                        stable = get_stable_plate_for_vehicle(tracking_id, plate_text, frame_count)
                        if stable:
                            display_text = stable
                            if ENABLE_SKIP_LOGIC and tracking_id in vehicle_stable_plates:
                                if tracking_id not in vehicle_locked_plates:
                                    vehicle_locked_plates[tracking_id] = stable
                                locked_text = vehicle_locked_plates[tracking_id]
                                if plate_text == locked_text:
                                    vehicle_completion_count[tracking_id] += 1
                                    if vehicle_completion_count[tracking_id] >= skip_manager.min_confident_readings:
                                        if tracking_id not in total_vehicles_completed_ever:
                                            skip_manager.completed_vehicles.add(tracking_id)
                                            total_vehicles_completed_ever.add(tracking_id)
                        else:
                            display_text = plate_text
                    else:
                        text_based_id = hash(plate_text) % 100000 + 100000
                        stable = get_stable_plate_for_vehicle(text_based_id, plate_text, frame_count)
                        display_text = stable if stable else plate_text
                
                if display_text and len(display_text) > 0:
                    plate_meta.text_params.display_text = display_text
                    plate_meta.text_params.set_bg_clr = 1
            except Exception:
                pass
        
        # Cleanup old tracks periodically (saves partial plates for vehicles that left)
        if frame_count % 60 == 0:
            cleanup_old_vehicles(frame_count)
            # Show cumulative stats (unique plates = unique vehicles with plates)
            total_plates = len(total_plates_by_vehicle)
            if ENABLE_SKIP_LOGIC:
                skip_manager.cleanup(frame_count)
                cleanup_parked_counts(set(vehicle_types_this_frame.keys()))
                total_completed = len(total_vehicles_completed_ever)
                print(f"[Info]{frame_count%TOTAL_FRAME}/{TOTAL_FRAME} | Plates found: {total_plates} | Vehicles completed: {total_completed}")
            if ENABLE_HEURISTICS:
                heuristics_manager.cleanup(set(vehicle_types_this_frame.keys()))
            if not ENABLE_SKIP_LOGIC:
                print(f"[Info]{frame_count%TOTAL_FRAME}/{TOTAL_FRAME} | Plates found: {total_plates}")
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK


def create_element(factory_name, element_name, exit_on_fail=True):
    """Create a GStreamer element with error handling."""
    element = Gst.ElementFactory.make(factory_name, element_name)
    if not element:
        if exit_on_fail:
            print(f"[ERROR] Unable to create element: {factory_name} ({element_name})")
            sys.exit(1)
        else:
            return None
    return element


def check_gstreamer_registry():
    """Check and update GStreamer plugin registry."""
    # Force registry update silently
    registry = Gst.Registry.get()
    registry.scan_path("/opt/nvidia/deepstream/deepstream/lib/gst-plugins")
    registry.scan_path("/opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins")
    registry.scan_path("/usr/lib/aarch64-linux-gnu/gstreamer-1.0")
    registry.scan_path("/usr/lib/x86_64-linux-gnu/gstreamer-1.0")


def create_encoder(element_name):
    """
    Create an H.264 encoder element.
    Original config uses enc-type=1 (software), so prefer software encoder.
    """
    check_gstreamer_registry()
    
    # Try software encoder first (matches original config enc-type=1)
    encoder_options = [
        "x264enc",        # Software encoder (preferred - matches enc-type=1)
        "avenc_h264",     # FFmpeg/libav encoder
        "nvv4l2h264enc",  # Jetson hardware encoder (fallback)
        "nvh264enc",      # NVIDIA GPU encoder (dGPU)
    ]
    
    for enc_factory in encoder_options:
        encoder = Gst.ElementFactory.make(enc_factory, element_name)
        if encoder:
            return encoder, enc_factory
    
    print("[ERROR] No H.264 encoder available!")
    sys.exit(1)




def main():
    """Main function to set up and run the DeepStream ALPR pipeline."""
    
    # Parse command line arguments
    args = parse_arguments()
    apply_arguments(args)
    
    print("[ALPR] Starting DeepStream ALPR Pipeline...")
    print(f"[ALPR] Mode: {'LIVE' if LIVE_MODE else 'FILE'}")
    print(f"[ALPR] Output: {'RTSP streaming' if ENABLE_RTSP else 'File'}")
    print(f"[ALPR] Resolution: {CFG['muxer_width']}x{CFG['muxer_height']} (padding={'ON' if CFG['enable_padding'] else 'OFF'})")
    
    # Change to config directory (config files use relative paths for models)
    if os.path.exists(CONFIG_DIR):
        os.chdir(CONFIG_DIR)
        print(f"[ALPR] Working directory: {os.getcwd()}")
    
    # Initialize GStreamer
    Gst.init(None)
    
    # Create Pipeline
    pipeline = Gst.Pipeline()
    if not pipeline:
        print("[ERROR] Unable to create Pipeline")
        sys.exit(1)
    pipeline.set_name("pipeline")
    
    # Source
    source = create_element("filesrc", "file-source")
    source.set_property("location", INPUT_VIDEO)
    
    # Demuxer
    demuxer = create_element("qtdemux", "demuxer")
    
    # H264 Parser
    h264parser = create_element("h264parse", "h264-parser")
    
    # Decoder (nvv4l2decoder for Jetson)
    decoder = create_element("nvv4l2decoder", "nvv4l2-decoder")
    try:
        decoder.set_property("cudadec-memtype", 2)
    except:
        pass
    
    # Stream Muxer
    streammux = create_element("nvstreammux", "stream-muxer")
    streammux.set_property("width", CFG['muxer_width'])
    streammux.set_property("height", CFG['muxer_height'])
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", CFG['muxer_batch_timeout'])
    streammux.set_property("live-source", 1 if LIVE_MODE else 0)
    streammux.set_property("nvbuf-memory-type", 0)
    streammux.set_property("gpu-id", 0)
    try:
        # Enable padding to preserve aspect ratio (adds black bars if needed)
        streammux.set_property("enable-padding", CFG['enable_padding'])
    except:
        pass
    
    # Primary GIE - Vehicle Detection
    pgie = create_element("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", PGIE_CONFIG)
    
    # Tracker
    tracker = create_element("nvtracker", "tracker")
    tracker.set_property("tracker-width", CFG['tracker_width'])
    tracker.set_property("tracker-height", CFG['tracker_height'])
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", TRACKER_CONFIG)
    tracker.set_property("display-tracking-id", 0)
    
    # Secondary GIE - License Plate Detection
    sgie_plate = create_element("nvinfer", "secondary-inference-plate")
    sgie_plate.set_property("config-file-path", SGIE_PLATE_DETECTOR_CONFIG)
    
    # Tertiary GIE - License Plate Recognition
    sgie_lpr = create_element("nvinfer", "secondary-inference-lpr")
    sgie_lpr.set_property("config-file-path", SGIE_LPR_CONFIG)
    
    # Queue after inference
    queue1 = create_element("queue", "queue1")
    
    # Video Converter (before OSD)
    nvvidconv1 = create_element("nvvideoconvert", "converter1")
    nvvidconv1.set_property("gpu-id", 0)
    nvvidconv1.set_property("nvbuf-memory-type", 0)
    
    # On-Screen Display
    osd = create_element("nvdsosd", "onscreen-display")
    osd.set_property("process-mode", 0)
    osd.set_property("display-text", 1)
    osd.set_property("display-bbox", 1)
    osd.set_property("gpu-id", 0)
    # Text styling (applies to all labels)
    try:
        osd.set_property("text-size", 15)
        osd.set_property("font", "Serif")
    except Exception:
        pass  # Properties might not exist in all versions
    
    # Video Converter (before encoder)
    nvvidconv2 = create_element("nvvideoconvert", "converter2")
    nvvidconv2.set_property("gpu-id", 0)
    
    # Create encoder
    encoder, encoder_type = create_encoder("encoder")
    use_software_encoder = encoder_type == "x264enc"
    
    # Additional converter for software encoder
    videoconvert_sw = None
    if use_software_encoder:
        videoconvert_sw = create_element("videoconvert", "converter-sw")
    
    # Caps filter
    capsfilter = create_element("capsfilter", "caps-filter")
    
    # Configure encoder and caps
    if encoder_type == "nvv4l2h264enc":
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        encoder.set_property("bitrate", CFG['bitrate'])
        encoder.set_property("iframeinterval", CFG['iframe_interval'])
    elif encoder_type == "x264enc":
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
        encoder.set_property("bitrate", CFG['bitrate'] // 1000)
        encoder.set_property("key-int-max", CFG['iframe_interval'])
        encoder.set_property("tune", "zerolatency")
        encoder.set_property("speed-preset", "ultrafast")
    else:
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
    
    capsfilter.set_property("caps", caps)
    
    # H264 Parser
    h264parser2 = create_element("h264parse", "h264-parser2")
    
    # Create sink based on mode (RTSP or File)
    if ENABLE_RTSP:
        # RTP payload for RTSP
        rtppay = create_element("rtph264pay", "rtppay")
        rtppay.set_property("config-interval", 1)
        rtppay.set_property("pt", 96)
        
        # UDP sink for RTSP server
        sink = create_element("udpsink", "udpsink")
        sink.set_property("host", "127.0.0.1")
        sink.set_property("port", 5400)
        sink.set_property("sync", 1 if LIVE_MODE else 0)
        sink.set_property("async", 0)
        muxer = None  # Not used in RTSP mode
    else:
        # MP4 Muxer and File Sink
        muxer = create_element("qtmux", "muxer")
        sink = create_element("filesink", "file-sink")
        sink.set_property("location", OUTPUT_VIDEO)
        sink.set_property("sync", 0)
        sink.set_property("async", 0)
        rtppay = None  # Not used in file mode
    
    # Add Elements to Pipeline
    pipeline.add(source)
    pipeline.add(demuxer)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie_plate)
    pipeline.add(sgie_lpr)
    pipeline.add(queue1)
    pipeline.add(nvvidconv1)
    pipeline.add(osd)
    pipeline.add(nvvidconv2)
    if use_software_encoder:
        pipeline.add(videoconvert_sw)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(h264parser2)
    
    if ENABLE_RTSP:
        pipeline.add(rtppay)
        pipeline.add(sink)
    else:
        pipeline.add(muxer)
        pipeline.add(sink)
    
    # Link Elements
    if not source.link(demuxer):
        print("[ERROR] Could not link source to demuxer")
        sys.exit(1)
    
    def demuxer_pad_added(demuxer, pad, data):
        pad_name = pad.get_name()
        if pad_name.startswith("video"):
            sink_pad = h264parser.get_static_pad("sink")
            if not sink_pad.is_linked():
                pad.link(sink_pad)
    
    demuxer.connect("pad-added", demuxer_pad_added, None)
    
    if not h264parser.link(decoder):
        print("[ERROR] Could not link h264parser to decoder")
        sys.exit(1)
    
    # Link decoder to streammux
    padtemplate = streammux.get_pad_template("sink_%u")
    sinkpad = streammux.request_pad(padtemplate, "sink_0", None)
    if not sinkpad:
        sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        print("[ERROR] Unable to get streammux sink pad")
        sys.exit(1)
    
    srcpad = decoder.get_static_pad("src")
    if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
        print("[ERROR] Could not link decoder to streammux")
        sys.exit(1)
    
    # Link main processing chain
    if not streammux.link(pgie):
        print("[ERROR] Could not link streammux to pgie")
        sys.exit(1)
    
    if not pgie.link(tracker):
        print("[ERROR] Could not link pgie to tracker")
        sys.exit(1)
    
    if not tracker.link(sgie_plate):
        print("[ERROR] Could not link tracker to sgie_plate")
        sys.exit(1)
    
    if not sgie_plate.link(sgie_lpr):
        print("[ERROR] Could not link sgie_plate to sgie_lpr")
        sys.exit(1)
    
    if not sgie_lpr.link(queue1):
        print("[ERROR] Could not link sgie_lpr to queue1")
        sys.exit(1)
    
    if not queue1.link(nvvidconv1):
        print("[ERROR] Could not link queue1 to nvvidconv1")
        sys.exit(1)
    
    if not nvvidconv1.link(osd):
        print("[ERROR] Could not link nvvidconv1 to osd")
        sys.exit(1)
    
    # Link encoding chain
    if not osd.link(nvvidconv2):
        print("[ERROR] Could not link osd to nvvidconv2")
        sys.exit(1)
    
    if use_software_encoder:
        if not nvvidconv2.link(videoconvert_sw):
            print("[ERROR] Could not link nvvidconv2 to videoconvert_sw")
            sys.exit(1)
        if not videoconvert_sw.link(capsfilter):
            print("[ERROR] Could not link videoconvert_sw to capsfilter")
            sys.exit(1)
    else:
        if not nvvidconv2.link(capsfilter):
            print("[ERROR] Could not link nvvidconv2 to capsfilter")
            sys.exit(1)
    
    if not capsfilter.link(encoder):
        print("[ERROR] Could not link capsfilter to encoder")
        sys.exit(1)
    
    if not encoder.link(h264parser2):
        print("[ERROR] Could not link encoder to h264parser2")
        sys.exit(1)
    
    # Link output based on mode
    if ENABLE_RTSP:
        if not h264parser2.link(rtppay):
            print("[ERROR] Could not link h264parser2 to rtppay")
            sys.exit(1)
        if not rtppay.link(sink):
            print("[ERROR] Could not link rtppay to sink")
            sys.exit(1)
    else:
        if not h264parser2.link(muxer):
            print("[ERROR] Could not link h264parser2 to muxer")
            sys.exit(1)
        if not muxer.link(sink):
            print("[ERROR] Could not link muxer to sink")
            sys.exit(1)
    
    # Add Pre-SGIE Probe (for "Read Once, Skip Later" optimization)
    # This probe runs BEFORE secondary GIE to skip already-completed vehicles
    if ENABLE_SKIP_LOGIC:
        tracker_srcpad = tracker.get_static_pad("src")
        if tracker_srcpad:
            tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, pre_sgie_probe, 0)
            print("[ALPR] GPU optimization ENABLED: Read Once, Skip Later")
    else:
        print("[ALPR] GPU optimization DISABLED (--no-skip mode for comparison)")
    
    # Add timing probes (--time) for inference latency measurement
    if ENABLE_TIMING:
        pgie_sink = pgie.get_static_pad("sink")
        pgie_src = pgie.get_static_pad("src")
        sgie_plate_sink = sgie_plate.get_static_pad("sink")
        sgie_plate_src = sgie_plate.get_static_pad("src")
        sgie_lpr_sink = sgie_lpr.get_static_pad("sink")
        sgie_lpr_src = sgie_lpr.get_static_pad("src")
        if all([pgie_sink, pgie_src, sgie_plate_sink, sgie_plate_src, sgie_lpr_sink, sgie_lpr_src]):
            pgie_sink.add_probe(Gst.PadProbeType.BUFFER, _make_timing_sink_probe('pgie'), 0)
            pgie_src.add_probe(Gst.PadProbeType.BUFFER, _make_timing_src_probe('pgie'), 0)
            sgie_plate_sink.add_probe(Gst.PadProbeType.BUFFER, _make_timing_sink_probe('sgie_plate'), 0)
            sgie_plate_src.add_probe(Gst.PadProbeType.BUFFER, _make_timing_src_probe('sgie_plate'), 0)
            sgie_lpr_sink.add_probe(Gst.PadProbeType.BUFFER, _make_timing_sink_probe('sgie_lpr'), 0)
            sgie_lpr_src.add_probe(Gst.PadProbeType.BUFFER, _make_timing_src_probe('sgie_lpr'), 0)
            print("[ALPR] Timing ENABLED: measuring inference latency")
    
    # Add Main Probe (after OSD, for display and completion tracking)
    osdsinkpad = osd.get_static_pad("sink")
    if not osdsinkpad:
        print("[ERROR] Unable to get OSD sink pad")
        sys.exit(1)
    
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # Start RTSP Server if enabled
    if ENABLE_RTSP:
        rtsp_server = GstRtspServer.RTSPServer.new()
        rtsp_server.set_service(str(RTSP_PORT))
        
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(
            '( udpsrc name=pay0 port=5400 buffer-size=524288 '
            'caps="application/x-rtp, media=video, clock-rate=90000, '
            'encoding-name=H264, payload=96" )'
        )
        factory.set_shared(True)
        
        mounts = rtsp_server.get_mount_points()
        mounts.add_factory(f"/{RTSP_STREAM_NAME}", factory)
        rtsp_server.attach(None)
    
    # Print info
    print(f"[ALPR] Input:  {INPUT_VIDEO}")
    if ENABLE_RTSP:
        print(f"[RTSP] Stream URL: rtsp://<your-ip>:{RTSP_PORT}/{RTSP_STREAM_NAME}")
        print(f"[RTSP] To view: ffplay rtsp://localhost:{RTSP_PORT}/{RTSP_STREAM_NAME}")
    else:
        print(f"[ALPR] Output: {OUTPUT_VIDEO}")
    
    # Create GLib MainLoop
    loop = GLib.MainLoop()
    
    # Add bus watch
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Start playing
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Unable to set pipeline to PLAYING state")
        sys.exit(1)
    
    print("[ALPR] Processing... (Press Ctrl+C to stop)")
    
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("\n[CLEANUP] Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        
        if frame_count > 0 and start_time:
            import time
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # FINAL CLEANUP: Save all remaining vehicles as partial plates
            # This ensures vehicles still in frame at end of video are captured
            final_cleanup_save_all()
            
            print("\n" + "="*60)
            print("PERFORMANCE SUMMARY")
            print("="*60)
            print(f"Total frames:     {frame_count}")
            print(f"Processing time:  {elapsed:.2f}s")
            print(f"Average FPS:      {fps:.2f}")
            
            if ENABLE_TIMING:
                timing_stats = _get_timing_stats()
                print("-"*60)
                print("INFERENCE LATENCY (--time)")
                print("-"*60)
                print(f"  PGIE (Vehicle Detection):     {timing_stats['pgie']:.2f} ms")
                print(f"  SGIE Plate (Plate Detection): {timing_stats['sgie_plate']:.2f} ms")
                print(f"  SGIE LPR (Plate Recognition): {timing_stats['sgie_lpr']:.2f} ms")
                total_latency = timing_stats['pgie'] + timing_stats['sgie_plate'] + timing_stats['sgie_lpr']
                print(f"  Total inference latency:      {total_latency:.2f} ms")
            
            # Count totals - consolidate by plate TEXT to handle tracker issues
            # Group plates by text
            all_plate_texts = set()
            for plate_text in total_plates_by_vehicle.values():
                all_plate_texts.add(plate_text)
            for vehicle_id, (plate_text, count) in total_partial_plates.items():
                all_plate_texts.add(plate_text)
            
            stable_text_count = len(set(total_plates_by_vehicle.values()))
            partial_texts = set(pt for pt, c in total_partial_plates.values())
            # Only count partial texts that aren't also in stable
            partial_only = partial_texts - set(total_plates_by_vehicle.values())
            
            print(f"Stable plates (high confidence):   {stable_text_count}")
            print(f"Partial plates (brief appearance): {len(partial_only)}")
            print(f"Total unique plate texts:          {len(all_plate_texts)}")
            
            # Consolidate plates by TEXT (not vehicle ID) to handle tracker issues
            # Group stable plates by text
            stable_by_text = {}
            for vehicle_id, plate_text in total_plates_by_vehicle.items():
                if plate_text not in stable_by_text:
                    stable_by_text[plate_text] = []
                stable_by_text[plate_text].append(vehicle_id)
            
            # Group partial plates by text
            partial_by_text = {}
            for vehicle_id, (plate_text, count) in total_partial_plates.items():
                if plate_text not in partial_by_text:
                    partial_by_text[plate_text] = {'count': 0, 'vehicles': []}
                partial_by_text[plate_text]['count'] += count
                partial_by_text[plate_text]['vehicles'].append(vehicle_id)
            
            # Show stable plates (high confidence) - DEDUPLICATED by plate text
            # Same plate text with multiple vehicle IDs = tracker ID switches (same car)
            if stable_by_text:
                print("-"*60)
                print("STABLE PLATES (>= 5 consistent readings):")
                for plate_text, vehicle_ids in sorted(stable_by_text.items()):
                    # Aggregate stats across all vehicle IDs for this plate (from stable vehicles)
                    total_reads = sum(vehicle_total_readings.get(vid, 0) for vid in vehicle_ids)
                    
                    # ALSO add readings from partial vehicles with same plate text
                    # (tracker switched but didn't get enough readings to become stable again)
                    if plate_text in partial_by_text:
                        total_reads += partial_by_text[plate_text]['count']
                        # Also include those vehicle IDs in the list
                        vehicle_ids = vehicle_ids + partial_by_text[plate_text]['vehicles']
                    
                    any_completed = any(vid in total_vehicles_completed_ever for vid in vehicle_ids)
                    
                    # Determine status
                    if any_completed:
                        status = "[COMPLETED]"
                    else:
                        # Check if any are locked
                        locked_count = sum(1 for vid in vehicle_ids if vid in vehicle_locked_plates)
                        if locked_count > 0:
                            status = "[LOCKED]"
                        else:
                            status = "[STABLE]"
                    
                    # Format vehicle type (deduplicated display)
                    type_str = vehicle_type_by_id.get(vehicle_ids[0], "vehicle")
                    if len(vehicle_ids) > 1:
                        type_str += " (tracker switched)"
                    print(f"  {plate_text} ({type_str}, {total_reads} readings) {status}")
            
            # Show partial plates (brief appearances) - EXCLUDE plates already in stable
            # Filter out any plate text that already appears in stable section
            partial_only = {k: v for k, v in partial_by_text.items() if k not in stable_by_text}
            
            if partial_only:
                print("-"*60)
                print("PARTIAL PLATES (brief appearance, lower confidence):")
                for plate_text, data in sorted(partial_only.items()):
                    total_readings = data['count']
                    vehicle_ids = data['vehicles']
                    type_str = vehicle_type_by_id.get(vehicle_ids[0], "vehicle")
                    if len(vehicle_ids) > 1:
                        type_str += " (tracker switched)"
                    print(f"  {plate_text} ({type_str}, {total_readings} readings)")
            
            # Calculate unique plates (by text, not by vehicle)
            # partial_only already excludes stable plates, so this is the true unique count
            all_unique_plates = set(stable_by_text.keys()) | set(partial_only.keys())
            
            if ENABLE_SKIP_LOGIC:
                stats = skip_manager.get_stats()
                print("-"*60)
                print("GPU OPTIMIZATION (Read Once, Skip Later): ENABLED")
                print(f"  Vehicles completed (total): {len(total_vehicles_completed_ever)}")
                print(f"  Total skipped:              {stats['total_skipped']}")
                print(f"  Total processed:            {stats['total_processed']}")
                if stats['total_skipped'] + stats['total_processed'] > 0:
                    print(f"  Skip ratio:                 {stats['skip_ratio']:.1%}")
            else:
                print("-"*60)
                print("GPU OPTIMIZATION: DISABLED (comparison mode)")
            
            if ENABLE_HEURISTICS:
                h_stats = heuristics_manager.get_stats()
                print("-"*60)
                print("HIGH-DENSITY HEURISTICS: ENABLED")
                print(f"  Density threshold:          {heuristics_manager.HIGH_DENSITY_THRESHOLD} vehicles")
                print(f"  Max process per frame:      {heuristics_manager.MAX_PROCESS_PER_FRAME}")
                print(f"  High-density activations:   {h_stats['high_density_activations']}")
                print(f"  Total vehicles filtered:    {h_stats['total_filtered']}")
            else:
                print("-"*60)
                print("HIGH-DENSITY HEURISTICS: DISABLED")
            print("="*60)


if __name__ == "__main__":
    main()