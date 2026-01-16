#!/usr/bin/env python3
"""
DeepStream YOLO11 + Tracker + License Plate Detection
With custom border colors and PLATE PERSISTENCE for stability
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds


# Load labels from file
def load_labels(label_file):
    try:
        with open(label_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except:
        return None


# Load labels from files
PGIE_LABELS = load_labels("../DeepStream-Yolo/labels.txt")
SGIE_LABELS = load_labels("../DeepStream-Yolo/labels_lpd.txt")

frame_count = 0

# GIE unique IDs (must match config files)
PGIE_UNIQUE_ID = 1
SGIE_UNIQUE_ID = 2

# License plate persistence: {vehicle_tracker_id: (plate_bbox, confidence, last_seen_frame)}
plate_cache = {}
PLATE_PERSIST_FRAMES = 15  # Keep plate visible for this many frames after last detection


def osd_sink_pad_buffer_probe(pad, info, data):
    """Process detections: set colors, display text, and persist plate detections"""
    global frame_count, plate_cache
    
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

        frame_count += 1
        vehicle_count = 0
        plate_count = 0
        
        # Track which vehicles have plates this frame
        vehicles_with_plates = set()
        
        # First pass: process all objects and update plate cache
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                gie_id = obj_meta.unique_component_id
                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                tracker_id = obj_meta.object_id
                
                # PGIE objects (vehicles)
                if gie_id == PGIE_UNIQUE_ID:
                    vehicle_count += 1
                    
                    # Get label from file
                    if PGIE_LABELS and class_id < len(PGIE_LABELS):
                        label = PGIE_LABELS[class_id]
                    else:
                        label = f"Class_{class_id}"
                    
                    # RED border for vehicles
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
                    obj_meta.rect_params.border_width = 3
                    
                    # Hide vehicle labels
                    obj_meta.text_params.display_text = ""
                
                # SGIE objects (license plates)
                elif gie_id == SGIE_UNIQUE_ID:
                    plate_count += 1
                    
                    # Get parent vehicle tracker ID
                    parent_id = obj_meta.parent.object_id if obj_meta.parent else -1
                    vehicles_with_plates.add(parent_id)
                    
                    # Get label from file
                    if SGIE_LABELS and class_id < len(SGIE_LABELS):
                        label = SGIE_LABELS[class_id]
                    else:
                        label = "plate"
                    
                    # Update cache for this vehicle's plate
                    rect = obj_meta.rect_params
                    plate_cache[parent_id] = {
                        'left': rect.left,
                        'top': rect.top,
                        'width': rect.width,
                        'height': rect.height,
                        'confidence': confidence,
                        'last_frame': frame_count,
                        'label': label
                    }
                    
                    # GREEN border for license plates
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                    obj_meta.rect_params.border_width = 4
                    
                    # Show plate label with confidence
                    obj_meta.text_params.display_text = f"{label}: {confidence*100:.0f}%"
                    obj_meta.text_params.font_params.font_size = 10
                    obj_meta.text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                    obj_meta.text_params.set_bg_clr = 1
                    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

                l_obj = l_obj.next
            except StopIteration:
                break

        # Second pass: add cached plates for vehicles without current detection
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                if obj_meta.unique_component_id == PGIE_UNIQUE_ID:
                    tracker_id = obj_meta.object_id
                    
                    # Check if this vehicle has a cached plate but no current detection
                    if tracker_id not in vehicles_with_plates and tracker_id in plate_cache:
                        cached = plate_cache[tracker_id]
                        frames_since = frame_count - cached['last_frame']
                        
                        # If within persistence window, create display meta for cached plate
                        if frames_since <= PLATE_PERSIST_FRAMES:
                            # Add display meta for the cached plate
                            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                            display_meta.num_rects = 1
                            
                            # Draw cached plate rectangle (slightly faded)
                            rect = display_meta.rect_params[0]
                            rect.left = cached['left']
                            rect.top = cached['top']
                            rect.width = cached['width']
                            rect.height = cached['height']
                            rect.border_width = 3
                            
                            # Fade color based on age (green -> darker)
                            fade = 1.0 - (frames_since / PLATE_PERSIST_FRAMES) * 0.5
                            rect.border_color.set(0.0, fade, 0.0, fade)
                            rect.has_bg_color = 0
                            
                            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                            plate_count += 1
                        else:
                            # Remove from cache if too old
                            del plate_cache[tracker_id]

                l_obj = l_obj.next
            except StopIteration:
                break

        # Clean up old cache entries
        stale_ids = [tid for tid, data in plate_cache.items() 
                     if frame_count - data['last_frame'] > PLATE_PERSIST_FRAMES * 2]
        for tid in stale_ids:
            del plate_cache[tid]

        # Print stats every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {vehicle_count} vehicles, {plate_count} plates (cache: {len(plate_cache)})")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def on_message(bus, msg, loop):
    msg_type = msg.type
    if msg_type == Gst.MessageType.EOS:
        print(f"\nEnd of stream. Total frames: {frame_count}")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print(f"Error: {err.message}")
        print(f"Debug: {debug}")
        loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        err, debug = msg.parse_warning()
        print(f"Warning: {err.message}")
    return True


def main(input_video, output_video, pgie_config, sgie_config, tracker_config):
    Gst.init(None)

    print(f"Input video:    {input_video}")
    print(f"Output video:   {output_video}")
    print(f"PGIE config:    {pgie_config}")
    print(f"SGIE config:    {sgie_config}")
    print(f"Tracker config: {tracker_config}")
    print(f"Plate persistence: {PLATE_PERSIST_FRAMES} frames")
    print("-" * 60)

    # Create output directory
    output_dir = os.path.dirname(output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Pipeline: Source -> Decode -> Mux -> PGIE -> Tracker -> SGIE -> OSD -> Encode -> Save
    pipeline_str = (
        f"filesrc location={input_video} ! "
        f"qtdemux ! "
        f"h264parse ! "
        f"nvv4l2decoder ! "
        f"m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! "
        f"nvinfer name=pgie config-file-path={pgie_config} ! "
        f"nvtracker name=tracker "
        f"ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
        f"ll-config-file={tracker_config} "
        f"tracker-width=480 tracker-height=288 "
        f"user-meta-pool-size=1024 ! "
        f"nvinfer name=sgie config-file-path={sgie_config} ! "
        f"nvvideoconvert ! "
        f"nvdsosd name=osd ! "
        f"nvvideoconvert ! "
        f"x264enc bitrate=4000 ! "
        f"h264parse ! "
        f"qtmux ! "
        f"filesink location={output_video}"
    )

    pipeline = Gst.parse_launch(pipeline_str)

    # Add probe to OSD to customize colors and persist plates
    osd = pipeline.get_by_name("osd")
    osd_sink_pad = osd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    # Message handling
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    # Start
    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopped by user")

    pipeline.set_state(Gst.State.NULL)
    print(f"Output saved to: {output_video}")


if __name__ == "__main__":
    # Default paths
    INPUT_VIDEO = "/opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project/sample.mp4"
    OUTPUT_VIDEO = "output/output_alpr.mp4"
    PGIE_CONFIG = "../DeepStream-Yolo/config_infer_primary_yolo11.txt"
    SGIE_CONFIG = "../DeepStream-Yolo/config_infer_secondary_yolo11.txt"
    TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

    if len(sys.argv) >= 2:
        INPUT_VIDEO = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_VIDEO = sys.argv[2]
    if len(sys.argv) >= 4:
        PGIE_CONFIG = sys.argv[3]
    if len(sys.argv) >= 5:
        SGIE_CONFIG = sys.argv[4]
    if len(sys.argv) >= 6:
        TRACKER_CONFIG = sys.argv[5]

    main(INPUT_VIDEO, OUTPUT_VIDEO, PGIE_CONFIG, SGIE_CONFIG, TRACKER_CONFIG)