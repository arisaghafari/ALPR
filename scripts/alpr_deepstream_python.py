import sys
import os

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

# ==============================================================================
# Configuration - Adjust paths as needed
# ==============================================================================
CONFIG_DIR = "/opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project"

INPUT_VIDEO = f"{CONFIG_DIR}/sample.mp4"
OUTPUT_VIDEO = f"{CONFIG_DIR}/output_video_python.mp4"

PGIE_CONFIG = f"{CONFIG_DIR}/DeepStream-Yolo/config_infer_primary_yolo11.txt"
SGIE_PLATE_DETECTOR_CONFIG = f"{CONFIG_DIR}/DeepStream-Yolo/config_infer_secondary_yolo11.txt"
SGIE_LPR_CONFIG = f"{CONFIG_DIR}/DeepStream-Yolo/config_infer_tertiary_lprnet.txt"
TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvSORT.yml"

# Path to main DeepStream app config (to read settings from)
APP_CONFIG = f"{CONFIG_DIR}/DeepStream-Yolo/deepstream_app_config.txt"


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
    values = {
        'muxer_width': 1280,
        'muxer_height': 720,
        'muxer_batch_timeout': 40000,
        'tracker_width': 640,
        'tracker_height': 384,
        'bitrate': 4000000,
        'iframe_interval': 30,
    }
    
    if config:
        # Read from streammux section
        if 'streammux' in config:
            values['muxer_width'] = int(config['streammux'].get('width', 1280))
            values['muxer_height'] = int(config['streammux'].get('height', 720))
            values['muxer_batch_timeout'] = int(config['streammux'].get('batched-push-timeout', 40000))
        
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

# Store plate recognition history PER VEHICLE (using vehicle's track ID)
# Key: vehicle_track_id (from parent object)
# Value: list of recent plate texts for this specific vehicle
vehicle_plate_history = defaultdict(list)

# Store the stable/final plate text PER VEHICLE
# Key: vehicle_track_id
# Value: stable plate text for this vehicle
vehicle_stable_plates = {}

# Track last frame each vehicle was seen (for cleanup)
vehicle_last_seen = {}

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
    global vehicle_plate_history, vehicle_stable_plates, vehicle_last_seen
    
    # Update last seen frame
    vehicle_last_seen[vehicle_id] = current_frame
    
    # Add new recognition to this vehicle's history
    if new_plate_text:
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
    
    # Only update stable plate if we have enough consistent votes
    if count >= MIN_VOTES_FOR_STABLE:
        vehicle_stable_plates[vehicle_id] = most_common_plate
        return most_common_plate
    elif vehicle_id in vehicle_stable_plates:
        # Keep previous stable plate if not enough new votes
        return vehicle_stable_plates[vehicle_id]
    else:
        # Return most common even if below threshold
        return most_common_plate

def cleanup_old_vehicles(current_frame, max_frames_missing=90):
    """Remove history for vehicles that haven't been seen recently."""
    global vehicle_plate_history, vehicle_stable_plates, vehicle_last_seen
    
    # Find vehicles not seen recently
    vehicles_to_remove = []
    for vehicle_id, last_frame in vehicle_last_seen.items():
        if current_frame - last_frame > max_frames_missing:
            vehicles_to_remove.append(vehicle_id)
    
    # Remove old vehicle data
    for vehicle_id in vehicles_to_remove:
        vehicle_plate_history.pop(vehicle_id, None)
        vehicle_stable_plates.pop(vehicle_id, None)
        vehicle_last_seen.pop(vehicle_id, None)




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


# Grid cell size for spatial indexing (pixels)
GRID_CELL_SIZE = 64


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Optimized probe using grid-based spatial indexing for O(1) vehicle lookup.
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
        except StopIteration:
            break
        
        # Spatial grid: key = (grid_x, grid_y), value = (vehicle_id, rect)
        # Each vehicle occupies multiple grid cells based on its bounding box
        vehicle_grid = {}
        plates_to_process = []
        
        # Single pass: build grid and collect plates
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                # Vehicles (gie-unique-id=1): Add to spatial grid
                if obj_meta.unique_component_id == 1:
                    obj_meta.text_params.display_text = f"{obj_meta.object_id}"
                    rect = obj_meta.rect_params
                    vehicle_id = obj_meta.object_id
                    
                    # Register vehicle in all grid cells it occupies
                    x1 = int(rect.left) // GRID_CELL_SIZE
                    y1 = int(rect.top) // GRID_CELL_SIZE
                    x2 = int(rect.left + rect.width) // GRID_CELL_SIZE
                    y2 = int(rect.top + rect.height) // GRID_CELL_SIZE
                    
                    for gx in range(x1, x2 + 1):
                        for gy in range(y1, y2 + 1):
                            cell_key = (gx, gy)
                            if cell_key not in vehicle_grid:
                                vehicle_grid[cell_key] = []
                            vehicle_grid[cell_key].append((vehicle_id, rect))
                
                # Plates (gie-unique-id=2): Collect for processing
                elif obj_meta.unique_component_id == 2:
                    plates_to_process.append(obj_meta)
                
            except:
                pass
            
            try:
                l_obj = l_obj.next
            except:
                break
        
        # Process plates with O(1) grid lookup
        for plate_meta in plates_to_process:
            try:
                # Try parent metadata first (fastest path)
                parent_id = 0
                if plate_meta.parent:
                    parent_id = plate_meta.parent.object_id
                    if parent_id == 18446744073709551615:  # Invalid UINT64_MAX
                        parent_id = 0
                
                # Fallback: O(1) grid lookup with bottom-edge matching
                # Plates are at the BOTTOM of vehicles, so match to vehicle
                # whose bottom edge is closest to the plate
                if parent_id == 0 and vehicle_grid:
                    rect = plate_meta.rect_params
                    plate_cx = rect.left + rect.width / 2
                    plate_cy = rect.top + rect.height / 2
                    cell_key = (int(plate_cx) // GRID_CELL_SIZE, int(plate_cy) // GRID_CELL_SIZE)
                    
                    best_match = 0
                    best_distance = float('inf')
                    
                    if cell_key in vehicle_grid:
                        for vehicle_id, v_rect in vehicle_grid[cell_key]:
                            # Check if plate center is inside this vehicle
                            if (v_rect.left <= plate_cx <= v_rect.left + v_rect.width and
                                v_rect.top <= plate_cy <= v_rect.top + v_rect.height):
                                # Distance from plate to vehicle's BOTTOM edge
                                # Plate should be near the bottom of its parent vehicle
                                vehicle_bottom = v_rect.top + v_rect.height
                                distance_to_bottom = abs(vehicle_bottom - plate_cy)
                                
                                # Keep the vehicle whose bottom is closest to plate
                                if distance_to_bottom < best_distance:
                                    best_distance = distance_to_bottom
                                    best_match = vehicle_id
                    
                    parent_id = best_match
                
                # Get LPR text from classifier (gie-unique-id=3)
                plate_text = ""
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
                                except:
                                    pass
                    except:
                        pass
                    try:
                        l_cls = l_cls.next
                    except:
                        break
                
                # Set display text
                if parent_id > 0:
                    stable = get_stable_plate_for_vehicle(parent_id, plate_text, frame_count)
                    if stable:
                        plate_meta.text_params.display_text = stable + f"_{parent_id}"
                elif plate_text:
                    plate_meta.text_params.display_text = plate_text + f"_{parent_id}"
                
            except:
                pass
        
        # Cleanup old tracks periodically
        if frame_count % 150 == 0:
            cleanup_old_vehicles(frame_count)
        
        try:
            l_frame = l_frame.next
        except:
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
    
    print("[ALPR] Starting DeepStream ALPR Pipeline...")
    
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
    # Match original config: cudadec-memtype=2
    try:
        decoder.set_property("cudadec-memtype", 2)
    except:
        pass
    
    # Stream Muxer (matching original config)
    streammux = create_element("nvstreammux", "stream-muxer")
    streammux.set_property("width", CFG['muxer_width'])
    streammux.set_property("height", CFG['muxer_height'])
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", CFG['muxer_batch_timeout'])
    streammux.set_property("live-source", 0)
    streammux.set_property("nvbuf-memory-type", 0)
    streammux.set_property("gpu-id", 0)
    try:
        streammux.set_property("enable-padding", 0)
    except:
        pass
    
    # Primary GIE - Vehicle Detection (YOLO11)
    # All settings come from config file (gpu-id, batch-size, etc.)
    pgie = create_element("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", PGIE_CONFIG)
    
    # Tracker
    tracker = create_element("nvtracker", "tracker")
    tracker.set_property("tracker-width", CFG['tracker_width'])
    tracker.set_property("tracker-height", CFG['tracker_height'])
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", TRACKER_CONFIG)
    tracker.set_property("display-tracking-id", 0)
    
    # Secondary GIE - License Plate Detection (YOLO11)
    # All settings come from config file
    sgie_plate = create_element("nvinfer", "secondary-inference-plate")
    sgie_plate.set_property("config-file-path", SGIE_PLATE_DETECTOR_CONFIG)
    
    # Tertiary GIE - License Plate Recognition (LPRNet)
    # All settings come from config file
    sgie_lpr = create_element("nvinfer", "secondary-inference-lpr")
    sgie_lpr.set_property("config-file-path", SGIE_LPR_CONFIG)
    
    # Queue after inference to help with buffer management
    queue1 = create_element("queue", "queue1")
    
    # Video Converter (before OSD)
    nvvidconv1 = create_element("nvvideoconvert", "converter1")
    nvvidconv1.set_property("gpu-id", 0)
    nvvidconv1.set_property("nvbuf-memory-type", 0)
    
    # On-Screen Display (matching original config)
    osd = create_element("nvdsosd", "onscreen-display")
    osd.set_property("process-mode", 0)  # CPU mode (from config)
    osd.set_property("display-text", 1)
    osd.set_property("display-bbox", 1)
    osd.set_property("gpu-id", 0)
    
    # Video Converter (before encoder)
    nvvidconv2 = create_element("nvvideoconvert", "converter2")
    nvvidconv2.set_property("gpu-id", 0)
    
    # Additional converter for software encoder (NVMM -> system memory)
    # This will be used only if we fall back to software encoder
    videoconvert_sw = None  # Will be created if needed
    
    # Caps filter for encoder input
    capsfilter = create_element("capsfilter", "caps-filter")
    
    # Create encoder (try multiple options)
    encoder, encoder_type = create_encoder("encoder")
    use_software_encoder = encoder_type == "x264enc"
    
    # If using software encoder, need additional converter to go from NVMM to system memory
    if use_software_encoder:
        videoconvert_sw = create_element("videoconvert", "converter-sw")
    
    # Set encoder properties based on type and configure caps
    if encoder_type == "nvv4l2h264enc":
        # Jetson hardware encoder - needs NVMM memory
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        encoder.set_property("bitrate", CFG['bitrate'])
        encoder.set_property("iframeinterval", CFG['iframe_interval'])
    elif encoder_type == "x264enc":
        # Software encoder - needs regular memory (not NVMM)
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
        encoder.set_property("bitrate", CFG['bitrate'] // 1000)  # x264enc uses kbps
        encoder.set_property("key-int-max", CFG['iframe_interval'])
        encoder.set_property("tune", "zerolatency")
        encoder.set_property("speed-preset", "ultrafast")
    elif encoder_type == "omxh264enc":
        # OMX encoder
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        encoder.set_property("bitrate", CFG['bitrate'])
        encoder.set_property("iframeinterval", CFG['iframe_interval'])
    else:
        # Default caps
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
    
    capsfilter.set_property("caps", caps)
    
    # H264 Parser (for muxer)
    h264parser2 = create_element("h264parse", "h264-parser2")
    
    # MP4 Muxer
    muxer = create_element("qtmux", "muxer")
    
    # File Sink
    sink = create_element("filesink", "file-sink")
    sink.set_property("location", OUTPUT_VIDEO)
    sink.set_property("sync", 0)
    sink.set_property("async", 0)  # Important for file output
    
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
    pipeline.add(muxer)
    pipeline.add(sink)
    
    # Link Elements
    
    # Link source -> demuxer
    if not source.link(demuxer):
        print("[ERROR] Could not link source to demuxer")
        sys.exit(1)
    
    # Demuxer -> h264parser will be linked dynamically (pad-added callback)
    def demuxer_pad_added(demuxer, pad, data):
        """Callback for dynamic pad linking from demuxer."""
        pad_name = pad.get_name()
        if pad_name.startswith("video"):
            sink_pad = h264parser.get_static_pad("sink")
            if not sink_pad.is_linked():
                pad.link(sink_pad)
    
    demuxer.connect("pad-added", demuxer_pad_added, None)
    
    # Link h264parser -> decoder
    if not h264parser.link(decoder):
        print("[ERROR] Could not link h264parser to decoder")
        sys.exit(1)
    
    # Get streammux sink pad and link decoder to it
    # Use request_pad instead of deprecated get_request_pad
    padtemplate = streammux.get_pad_template("sink_%u")
    sinkpad = streammux.request_pad(padtemplate, "sink_0", None)
    if not sinkpad:
        # Fallback to old method
        sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        print("[ERROR] Unable to get streammux sink pad")
        sys.exit(1)
    
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        print("[ERROR] Unable to get decoder source pad")
        sys.exit(1)
    
    if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
        print("[ERROR] Could not link decoder to streammux")
        sys.exit(1)
    
    # Link the main processing chain
    # streammux -> pgie -> tracker -> sgie_plate -> sgie_lpr -> nvvidconv1 -> osd
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
    # Hardware: osd -> nvvidconv2 -> capsfilter -> encoder -> h264parser2 -> muxer -> sink
    # Software: osd -> nvvidconv2 -> videoconvert_sw -> capsfilter -> encoder -> ...
    if not osd.link(nvvidconv2):
        print("[ERROR] Could not link osd to nvvidconv2")
        sys.exit(1)
    
    if use_software_encoder:
        # Software encoder path: need videoconvert to convert from NVMM
        if not nvvidconv2.link(videoconvert_sw):
            print("[ERROR] Could not link nvvidconv2 to videoconvert_sw")
            sys.exit(1)
        if not videoconvert_sw.link(capsfilter):
            print("[ERROR] Could not link videoconvert_sw to capsfilter")
            sys.exit(1)
    else:
        # Hardware encoder path
        if not nvvidconv2.link(capsfilter):
            print("[ERROR] Could not link nvvidconv2 to capsfilter")
            sys.exit(1)
    
    if not capsfilter.link(encoder):
        print("[ERROR] Could not link capsfilter to encoder")
        sys.exit(1)
    
    if not encoder.link(h264parser2):
        print("[ERROR] Could not link encoder to h264parser2")
        sys.exit(1)
    
    if not h264parser2.link(muxer):
        print("[ERROR] Could not link h264parser2 to muxer")
        sys.exit(1)
    
    if not muxer.link(sink):
        print("[ERROR] Could not link muxer to sink")
        sys.exit(1)
    
    # Add Probe for Metadata Access
    
    osdsinkpad = osd.get_static_pad("sink")
    if not osdsinkpad:
        print("[ERROR] Unable to get OSD sink pad")
        sys.exit(1)
    
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # Start Pipeline
    print(f"[ALPR] Input:  {INPUT_VIDEO}")
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
        # Cleanup
        print("\n[CLEANUP] Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        
        # Print final stats
        if frame_count > 0 and start_time:
            import time
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[DONE] Processed {frame_count} frames in {elapsed:.2f}s (Avg FPS: {fps:.2f})")
            print(f"[DONE] Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
