#!/usr/bin/env python3

"""
ALPR DeepStream Pipeline - Python Binding Implementation
=========================================================
This script implements an Automatic License Plate Recognition (ALPR) pipeline
using NVIDIA DeepStream SDK 7.1 Python bindings for Jetson Orin.

Pipeline Architecture:
  filesrc -> h264parse -> nvv4l2decoder -> nvstreammux 
  -> nvinfer (YOLO11 Vehicle Detection)
  -> nvtracker (NvSORT)
  -> nvinfer (YOLO11 License Plate Detection)
  -> nvinfer (LPRNet Character Recognition)
  -> nvvideoconvert -> nvdsosd -> nvv4l2h264enc -> h264parse -> mp4mux -> filesink

Author: Converted from deepstream-app config files
Date: 2026
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds
import ctypes
from datetime import datetime

# ============================================================================
# Configuration Constants
# ============================================================================

# Paths - Adjust these to match your DeepStream container paths
PROJECT_PATH = "/opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project"

# Input/Output
INPUT_FILE = f"{PROJECT_PATH}/sample.mp4"
OUTPUT_FILE = f"{PROJECT_PATH}/output_alpr.mp4"

# Config files for inference engines
PGIE_CONFIG_FILE = f"{PROJECT_PATH}/DeepStream-Yolo/config_infer_primary_python.txt"
SGIE_PLATE_DETECTOR_CONFIG = f"{PROJECT_PATH}/DeepStream-Yolo/config_infer_secondar_python.txt"
TGIE_PLATE_RECOGNIZER_CONFIG = f"{PROJECT_PATH}/DeepStream-Yolo/config_infer_tertiary_python.txt"

# Tracker config
TRACKER_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvSORT.yml"
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

# Stream settings
MUXER_WIDTH = 1280
MUXER_HEIGHT = 720
MUXER_BATCH_TIMEOUT_USEC = 40000
MUXER_BATCH_SIZE = 1

# Tracker settings
TRACKER_WIDTH = 640
TRACKER_HEIGHT = 384

# OSD settings
OSD_PROCESS_MODE = 0  # 0=CPU, 1=GPU
OSD_DISPLAY_TEXT = 1
OSD_DISPLAY_BBOX = 1

# Encoder settings
BITRATE = 4000000
IFRAME_INTERVAL = 30

# Class IDs for COCO (YOLO primary model)
# person=0, bicycle=1, car=2, motorcycle=3, airplane=4, bus=5, train=6, truck=7
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Unique IDs for GIE components
PGIE_UNIQUE_ID = 1
SGIE_UNIQUE_ID = 2
TGIE_UNIQUE_ID = 3

# ============================================================================
# Global Variables for Tracking
# ============================================================================

# Store plate readings per tracked object
plate_readings = {}  # {object_id: {'plate': str, 'confidence': float, 'count': int}}


# ============================================================================
# Utility Functions
# ============================================================================

def get_label_names_from_file(filepath):
    """Read label names from file."""
    labels = []
    try:
        with open(filepath, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"[WARNING] Label file not found: {filepath}")
    return labels


def is_aarch64():
    """Check if running on ARM64 architecture (Jetson)."""
    return os.uname().machine == 'aarch64'


# ============================================================================
# Probe Callbacks
# ============================================================================

def sgie_src_pad_buffer_probe(pad, info, u_data):
    """
    Probe callback attached after tertiary inference (LPRNet).
    This extracts the license plate recognition results from classifier metadata.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("[ERROR] Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # This is a vehicle object from primary detector
            if obj_meta.unique_component_id == PGIE_UNIQUE_ID:
                vehicle_id = obj_meta.object_id
                vehicle_class = obj_meta.class_id
                
                # Check if this vehicle has license plate detections
                l_classifier = obj_meta.classifier_meta_list
                
                while l_classifier is not None:
                    try:
                        classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                    except StopIteration:
                        break
                    
                    # Check if this is from LPRNet (tertiary inference)
                    if classifier_meta.unique_component_id == TGIE_UNIQUE_ID:
                        l_label = classifier_meta.label_info_list
                        
                        while l_label is not None:
                            try:
                                label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                            except StopIteration:
                                break
                            
                            plate_text = pyds.get_string(label_info.result_label)
                            confidence = label_info.result_prob
                            
                            if plate_text and len(plate_text) > 0:
                                # Update plate readings for this vehicle
                                if vehicle_id not in plate_readings:
                                    plate_readings[vehicle_id] = {
                                        'plate': plate_text,
                                        'confidence': confidence,
                                        'count': 1
                                    }
                                else:
                                    # Keep the reading with highest confidence
                                    if confidence > plate_readings[vehicle_id]['confidence']:
                                        plate_readings[vehicle_id]['plate'] = plate_text
                                        plate_readings[vehicle_id]['confidence'] = confidence
                                    plate_readings[vehicle_id]['count'] += 1
                                
                                print(f"[PLATE] Vehicle ID: {vehicle_id}, Plate: {plate_text}, Confidence: {confidence:.2f}")
                            
                            try:
                                l_label = l_label.next
                            except StopIteration:
                                break
                    
                    try:
                        l_classifier = l_classifier.next
                    except StopIteration:
                        break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe callback attached to OSD sink pad.
    This displays detection info and adds custom text overlays.
    """
    frame_number = 0
    num_vehicles = 0
    num_plates = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("[ERROR] Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Count objects by GIE
            if obj_meta.unique_component_id == PGIE_UNIQUE_ID:
                if obj_meta.class_id in VEHICLE_CLASS_IDS:
                    num_vehicles += 1
                    
                    # Set vehicle bounding box color (blue)
                    obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.8)
                    obj_meta.rect_params.border_width = 3
                    
                    # Add plate text to vehicle label if we have a reading
                    vehicle_id = obj_meta.object_id
                    if vehicle_id in plate_readings:
                        plate_info = plate_readings[vehicle_id]
                        # Create display text with plate number
                        display_text = f"ID:{vehicle_id} [{plate_info['plate']}]"
                        obj_meta.text_params.display_text = display_text
                        obj_meta.text_params.font_params.font_name = "Serif"
                        obj_meta.text_params.font_params.font_size = 12
                        obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                        obj_meta.text_params.set_bg_clr = 1
                        obj_meta.text_params.text_bg_clr.set(0.0, 0.5, 0.0, 0.7)

            elif obj_meta.unique_component_id == SGIE_UNIQUE_ID:
                num_plates += 1
                # Set plate bounding box color (green)
                obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                obj_meta.rect_params.border_width = 2

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Add display metadata with statistics
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        
        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = (
            f"Frame: {frame_number} | Vehicles: {num_vehicles} | Plates: {num_plates} | "
            f"Total Readings: {len(plate_readings)}"
        )
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 14
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # Print frame info periodically
        if frame_number % 30 == 0:
            print(f"[INFO] Frame {frame_number}: Vehicles={num_vehicles}, Plates={num_plates}")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ============================================================================
# Pipeline Creation
# ============================================================================

def create_element(factory_name, element_name):
    """Create a GStreamer element with error checking."""
    element = Gst.ElementFactory.make(factory_name, element_name)
    if not element:
        raise RuntimeError(f"Unable to create element: {factory_name} ({element_name})")
    print(f"[OK] Created: {element_name}")
    return element


def create_encoder():
    """Create video encoder - try hardware first, fallback to software."""
    # List of encoders to try in order of preference
    encoders = [
        ("nvv4l2h264enc", "h264-encoder"),      # Jetson hardware encoder
        ("x264enc", "h264-encoder"),             # Software x264 encoder
        ("avenc_h264", "h264-encoder"),          # FFmpeg/libav encoder
        ("openh264enc", "h264-encoder"),         # OpenH264 encoder
    ]
    
    for factory, name in encoders:
        encoder = Gst.ElementFactory.make(factory, name)
        if encoder:
            print(f"[OK] Created encoder: {factory}")
            return encoder, factory
    
    raise RuntimeError("No H264 encoder available! Install x264enc or use display output.")


def bus_call(bus, message, loop):
    """Handle GStreamer bus messages."""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("\n[INFO] End-of-stream reached")
        print_final_results()
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"[WARNING] {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[ERROR] {err}: {debug}")
        loop.quit()
    elif t == Gst.MessageType.STATE_CHANGED:
        if message.src.get_name() == "pipeline":
            old_state, new_state, pending = message.parse_state_changed()
            print(f"[STATE] Pipeline: {old_state.value_nick} -> {new_state.value_nick}")
    return True


def print_final_results():
    """Print summary of all detected license plates."""
    print("\n" + "=" * 60)
    print("ALPR DETECTION SUMMARY")
    print("=" * 60)
    
    if not plate_readings:
        print("No license plates were detected.")
    else:
        print(f"Total unique vehicles with plates: {len(plate_readings)}")
        print("-" * 60)
        for vehicle_id, info in sorted(plate_readings.items()):
            print(f"  Vehicle ID: {vehicle_id:4d} | Plate: {info['plate']:12s} | "
                  f"Confidence: {info['confidence']:.2f} | Readings: {info['count']}")
    
    print("=" * 60)
    
    # Save results to file
    results_file = OUTPUT_FILE.replace('.mp4', '_results.txt')
    try:
        with open(results_file, 'w') as f:
            f.write(f"ALPR Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            for vehicle_id, info in sorted(plate_readings.items()):
                f.write(f"Vehicle {vehicle_id}: {info['plate']} (conf: {info['confidence']:.2f})\n")
        print(f"[INFO] Results saved to: {results_file}")
    except Exception as e:
        print(f"[WARNING] Could not save results: {e}")


def main(args):
    """Main function to build and run the DeepStream pipeline."""
    
    # Parse command line arguments
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE
    
    if len(args) > 1:
        input_file = args[1]
    if len(args) > 2:
        output_file = args[2]
    
    print("\n" + "=" * 60)
    print("ALPR DeepStream Pipeline")
    print("=" * 60)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print("=" * 60 + "\n")

    # Initialize GStreamer
    Gst.init(None)

    # Create the pipeline
    print("[INFO] Creating Pipeline...")
    pipeline = Gst.Pipeline()
    if not pipeline:
        raise RuntimeError("Unable to create Pipeline")

    # ========================================================================
    # SOURCE: File source with demuxer for MP4 files
    # ========================================================================
    print("[INFO] Creating source elements...")
    
    # Use uridecodebin for flexible source handling (supports MP4, RTSP, etc.)
    source = create_element("uridecodebin", "uri-decode-bin")
    source.set_property("uri", f"file://{input_file}")
    
    # ========================================================================
    # STREAMMUX: Batch frames for inference
    # ========================================================================
    print("[INFO] Creating streammux...")
    streammux = create_element("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", MUXER_BATCH_SIZE)
    streammux.set_property("width", MUXER_WIDTH)
    streammux.set_property("height", MUXER_HEIGHT)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("live-source", 0)
    
    if is_aarch64():
        streammux.set_property("nvbuf-memory-type", 0)  # NVBUF_MEM_DEFAULT for Jetson
    else:
        streammux.set_property("nvbuf-memory-type", 3)  # NVBUF_MEM_CUDA_UNIFIED for dGPU
    
    # ========================================================================
    # PRIMARY GIE: YOLO11 Vehicle Detection
    # ========================================================================
    print("[INFO] Creating primary inference (YOLO11 Vehicle Detection)...")
    pgie = create_element("nvinfer", "primary-gie")
    pgie.set_property("config-file-path", PGIE_CONFIG_FILE)
    pgie.set_property("unique-id", PGIE_UNIQUE_ID)
    # Set input tensor memory type for Jetson
    if is_aarch64():
        pgie.set_property("input-tensor-meta", False)
    
    # ========================================================================
    # TRACKER: NvSORT Multi-Object Tracker
    # ========================================================================
    print("[INFO] Creating tracker...")
    tracker = create_element("nvtracker", "tracker")
    tracker.set_property("tracker-width", TRACKER_WIDTH)
    tracker.set_property("tracker-height", TRACKER_HEIGHT)
    tracker.set_property("ll-lib-file", TRACKER_LIB)
    tracker.set_property("ll-config-file", TRACKER_CONFIG_FILE)
    tracker.set_property("display-tracking-id", 0)
    # Set GPU ID for tracker
    tracker.set_property("gpu-id", 0)
    
    # ========================================================================
    # QUEUE before SGIE: Buffer management for secondary inference
    # ========================================================================
    print("[INFO] Creating queue before secondary GIE...")
    queue_sgie = create_element("queue", "queue-sgie")
    queue_sgie.set_property("max-size-buffers", 8)
    queue_sgie.set_property("max-size-bytes", 0)
    queue_sgie.set_property("max-size-time", 0)
    
    # ========================================================================
    # SECONDARY GIE: YOLO11 License Plate Detection
    # ========================================================================
    print("[INFO] Creating secondary inference (License Plate Detection)...")
    sgie = create_element("nvinfer", "secondary-gie-plate-detector")
    sgie.set_property("config-file-path", SGIE_PLATE_DETECTOR_CONFIG)
    sgie.set_property("unique-id", SGIE_UNIQUE_ID)
    sgie.set_property("process-mode", 2)  # Secondary mode
    # batch-size is set in config file (default: 4 for multiple vehicles)
    
    # ========================================================================
    # QUEUE before TGIE: Buffer management for tertiary inference
    # ========================================================================
    print("[INFO] Creating queue before tertiary GIE...")
    queue_tgie = create_element("queue", "queue-tgie")
    queue_tgie.set_property("max-size-buffers", 8)
    queue_tgie.set_property("max-size-bytes", 0)
    queue_tgie.set_property("max-size-time", 0)
    
    # ========================================================================
    # TERTIARY GIE: LPRNet License Plate Recognition
    # ========================================================================
    print("[INFO] Creating tertiary inference (LPRNet Recognition)...")
    tgie = create_element("nvinfer", "tertiary-gie-plate-recognizer")
    tgie.set_property("config-file-path", TGIE_PLATE_RECOGNIZER_CONFIG)
    tgie.set_property("unique-id", TGIE_UNIQUE_ID)
    tgie.set_property("process-mode", 2)  # Secondary mode
    # batch-size set in config file
    
    # ========================================================================
    # QUEUE after TGIE: Stabilize buffer flow to OSD
    # ========================================================================
    print("[INFO] Creating queue after tertiary GIE...")
    queue1 = create_element("queue", "queue1")
    queue1.set_property("max-size-buffers", 8)
    queue1.set_property("max-size-bytes", 0)
    queue1.set_property("max-size-time", 0)
    
    # ========================================================================
    # VIDEO CONVERT: NV12 to RGBA for OSD
    # ========================================================================
    print("[INFO] Creating video converter...")
    nvvidconv = create_element("nvvideoconvert", "converter")
    if is_aarch64():
        nvvidconv.set_property("nvbuf-memory-type", 0)  # NVBUF_MEM_DEFAULT for Jetson
    
    # ========================================================================
    # OSD: On-Screen Display
    # ========================================================================
    print("[INFO] Creating OSD...")
    nvosd = create_element("nvdsosd", "on-screen-display")
    nvosd.set_property("process-mode", OSD_PROCESS_MODE)
    nvosd.set_property("display-text", OSD_DISPLAY_TEXT)
    nvosd.set_property("display-bbox", OSD_DISPLAY_BBOX)
    
    # ========================================================================
    # ENCODER & SINK: Output to MP4 file
    # ========================================================================
    print("[INFO] Creating encoder and sink elements...")
    
    # Video converter before encoding
    nvvidconv2 = create_element("nvvideoconvert", "converter2")
    
    # Create encoder (tries hardware first, then software fallback)
    encoder, encoder_type = create_encoder()
    
    # Caps filter - different format based on encoder type
    capsfilter = create_element("capsfilter", "caps-filter")
    if encoder_type == "nvv4l2h264enc":
        # Hardware encoder needs NVMM memory
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        encoder.set_property("bitrate", BITRATE)
        encoder.set_property("iframeinterval", IFRAME_INTERVAL)
        if is_aarch64():
            encoder.set_property("preset-level", 1)  # UltraFast preset
            encoder.set_property("insert-sps-pps", 1)
            encoder.set_property("bufapi-version", 1)
    elif encoder_type == "x264enc":
        # Software encoder needs regular memory
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
        encoder.set_property("bitrate", BITRATE // 1000)  # x264enc uses kbps
        encoder.set_property("speed-preset", "ultrafast")
        encoder.set_property("tune", "zerolatency")
    else:
        # Generic fallback
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
    
    capsfilter.set_property("caps", caps)
    
    # H264 parser
    h264parser = create_element("h264parse", "h264-parser")
    
    # MP4 muxer
    muxer = create_element("qtmux", "mp4-muxer")
    
    # File sink
    sink = create_element("filesink", "file-sink")
    sink.set_property("location", output_file)
    sink.set_property("sync", 0)  # Don't sync to clock for file output
    
    # For software encoders, we need an additional converter to copy from GPU to CPU
    videoconvert_cpu = None
    if encoder_type != "nvv4l2h264enc":
        print("[INFO] Using software encoder - adding CPU video converter...")
        videoconvert_cpu = create_element("videoconvert", "cpu-converter")
    
    # ========================================================================
    # ADD ELEMENTS TO PIPELINE
    # ========================================================================
    print("[INFO] Adding elements to pipeline...")
    pipeline.add(source)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(queue_sgie)
    pipeline.add(sgie)
    pipeline.add(queue_tgie)
    pipeline.add(tgie)
    pipeline.add(queue1)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    if videoconvert_cpu:
        pipeline.add(videoconvert_cpu)
    pipeline.add(encoder)
    pipeline.add(h264parser)
    pipeline.add(muxer)
    pipeline.add(sink)
    
    # ========================================================================
    # LINK ELEMENTS
    # ========================================================================
    print("[INFO] Linking elements...")
    
    # Source will be linked dynamically via pad-added callback
    # streammux -> pgie -> tracker -> sgie -> tgie -> nvvidconv -> nvosd -> nvvidconv2 -> capsfilter -> encoder -> h264parser -> muxer -> sink
    
    if not streammux.link(pgie):
        raise RuntimeError("Failed to link streammux to pgie")
    if not pgie.link(tracker):
        raise RuntimeError("Failed to link pgie to tracker")
    if not tracker.link(queue_sgie):
        raise RuntimeError("Failed to link tracker to queue_sgie")
    if not queue_sgie.link(sgie):
        raise RuntimeError("Failed to link queue_sgie to sgie")
    if not sgie.link(queue_tgie):
        raise RuntimeError("Failed to link sgie to queue_tgie")
    if not queue_tgie.link(tgie):
        raise RuntimeError("Failed to link queue_tgie to tgie")
    if not tgie.link(queue1):
        raise RuntimeError("Failed to link tgie to queue1")
    if not queue1.link(nvvidconv):
        raise RuntimeError("Failed to link queue1 to nvvidconv")
    if not nvvidconv.link(nvosd):
        raise RuntimeError("Failed to link nvvidconv to nvosd")
    if not nvosd.link(nvvidconv2):
        raise RuntimeError("Failed to link nvosd to nvvidconv2")
    if not nvvidconv2.link(capsfilter):
        raise RuntimeError("Failed to link nvvidconv2 to capsfilter")
    
    # Link to encoder (with CPU converter for software encoders)
    if videoconvert_cpu:
        if not capsfilter.link(videoconvert_cpu):
            raise RuntimeError("Failed to link capsfilter to cpu-converter")
        if not videoconvert_cpu.link(encoder):
            raise RuntimeError("Failed to link cpu-converter to encoder")
    else:
        if not capsfilter.link(encoder):
            raise RuntimeError("Failed to link capsfilter to encoder")
    if not encoder.link(h264parser):
        raise RuntimeError("Failed to link encoder to h264parser")
    if not h264parser.link(muxer):
        raise RuntimeError("Failed to link h264parser to muxer")
    if not muxer.link(sink):
        raise RuntimeError("Failed to link muxer to sink")
    
    # ========================================================================
    # DYNAMIC PAD HANDLING (for uridecodebin)
    # ========================================================================
    def on_pad_added(decodebin, pad, streammux):
        """Handle dynamic pad creation from uridecodebin."""
        caps = pad.get_current_caps()
        if caps is None:
            caps = pad.query_caps(None)
        
        struct = caps.get_structure(0)
        name = struct.get_name()
        
        print(f"[INFO] Pad added: {name}")
        
        if name.startswith("video"):
            sinkpad = streammux.request_pad_simple("sink_0")
            if sinkpad is None:
                print("[ERROR] Unable to get streammux sink pad")
                return
            
            if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
                print("[ERROR] Failed to link decoder to streammux")
            else:
                print("[OK] Linked decoder to streammux")
    
    source.connect("pad-added", on_pad_added, streammux)
    
    # ========================================================================
    # ADD PROBES
    # ========================================================================
    print("[INFO] Adding probe callbacks...")
    
    # Probe after tertiary inference to extract plate readings
    tgie_src_pad = tgie.get_static_pad("src")
    if tgie_src_pad:
        tgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_src_pad_buffer_probe, 0)
    
    # Probe at OSD sink for display
    osd_sink_pad = nvosd.get_static_pad("sink")
    if osd_sink_pad:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # ========================================================================
    # RUN PIPELINE
    # ========================================================================
    print("\n[INFO] Starting pipeline...")
    
    # Create event loop
    loop = GLib.MainLoop()
    
    # Add bus watch
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Start playing
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        print_final_results()
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
    finally:
        # Cleanup
        print("[INFO] Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        print("[INFO] Pipeline stopped")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     ALPR - Automatic License Plate Recognition                ║
    ║     DeepStream 7.1 Python Pipeline                            ║
    ║     For Jetson Orin                                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    sys.exit(main(sys.argv))

