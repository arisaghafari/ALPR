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


PGIE_LABELS = load_labels("../DeepStream-Yolo/labels.txt")
SGIE_LABELS = load_labels("../DeepStream-Yolo/labels_lpd.txt")

frame_count = 0
PGIE_UNIQUE_ID = 1
SGIE_UNIQUE_ID = 2

# Plate persistence cache: {vehicle_tracker_id: {'frame': last_frame, 'confidence': conf}}
plate_cache = {}
PLATE_PERSIST_FRAMES = 20  # Keep plate info for 20 frames


def osd_sink_pad_buffer_probe(pad, info, data):
    """Process detections: set colors and track plates"""
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
        cached_plates = 0
        
        # Track vehicles with plates this frame
        vehicles_with_plates = set()
        
        # Process all objects
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                gie_id = obj_meta.unique_component_id
                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                tracker_id = obj_meta.object_id
                
                # PGIE objects (vehicles) - RED border
                if gie_id == PGIE_UNIQUE_ID:
                    vehicle_count += 1
                    
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
                    obj_meta.rect_params.border_width = 3
                    obj_meta.text_params.display_text = ""
                    
                    # Check if this vehicle has a recent plate in cache
                    if tracker_id in plate_cache:
                        cache_age = frame_count - plate_cache[tracker_id]['frame']
                        if cache_age <= PLATE_PERSIST_FRAMES:
                            cached_plates += 1
                
                # SGIE objects (license plates) - GREEN border
                elif gie_id == SGIE_UNIQUE_ID:
                    plate_count += 1
                    
                    # Get parent vehicle tracker ID
                    parent_id = obj_meta.parent.object_id if obj_meta.parent else -1
                    vehicles_with_plates.add(parent_id)
                    
                    # Update cache
                    plate_cache[parent_id] = {
                        'frame': frame_count,
                        'confidence': confidence,
                        'left': obj_meta.rect_params.left,
                        'top': obj_meta.rect_params.top,
                        'width': obj_meta.rect_params.width,
                        'height': obj_meta.rect_params.height
                    }
                    
                    # Get label
                    if SGIE_LABELS and class_id < len(SGIE_LABELS):
                        label = SGIE_LABELS[class_id]
                    else:
                        label = "LP"
                    
                    # GREEN border
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                    obj_meta.rect_params.border_width = 4
                    
                    # Show confidence
                    obj_meta.text_params.display_text = f"{label}: {confidence*100:.0f}%"
                    obj_meta.text_params.font_params.font_size = 10
                    obj_meta.text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                    obj_meta.text_params.set_bg_clr = 1
                    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

                l_obj = l_obj.next
            except StopIteration:
                break
        
        # Clean up old cache entries
        stale_ids = [tid for tid, data in plate_cache.items() 
                     if frame_count - data['frame'] > PLATE_PERSIST_FRAMES * 2]
        for tid in stale_ids:
            del plate_cache[tid]

        # Print stats every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {vehicle_count} vehicles, {plate_count} plates, {cached_plates} cached")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def on_message(bus, msg, loop):
    if msg.type == Gst.MessageType.EOS:
        print(f"\nEnd of stream. Total frames: {frame_count}")
        loop.quit()
    elif msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print(f"Error: {err.message}")
        loop.quit()
    return True


def main(input_video, output_video, pgie_config, sgie_config, tracker_config):
    Gst.init(None)

    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print(f"Plate persistence: {PLATE_PERSIST_FRAMES} frames")
    print("-" * 50)

    output_dir = os.path.dirname(output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pipeline_str = (
        f"filesrc location={input_video} ! "
        f"qtdemux ! "
        f"h264parse ! "
        f"nvv4l2decoder ! "
        f"m.sink_0 nvstreammux name=m batch-size=1 width=960 height=520 nvbuf-memory-type=0 ! "
        f"nvinfer name=pgie config-file-path={pgie_config} ! "
        f"nvtracker name=tracker "
        f"ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
        f"ll-config-file={tracker_config} "
        f"tracker-width=320 tracker-height=192 ! "
        f"nvinfer name=sgie config-file-path={sgie_config} ! "
        f"nvvideoconvert ! "
        f"nvdsosd name=osd ! "
        f"nvvideoconvert ! "
        f"x264enc bitrate=4000000 ! "
        f"h264parse ! "
        f"qtmux ! "
        f"filesink location={output_video}"
    )

    pipeline = Gst.parse_launch(pipeline_str)

    osd = pipeline.get_by_name("osd")
    osd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopped")

    pipeline.set_state(Gst.State.NULL)
    print(f"Output saved to: {output_video}")


if __name__ == "__main__":
    INPUT_VIDEO = "/opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project/sample.mp4"
    OUTPUT_VIDEO = "output/output_alpr.mp4"
    PGIE_CONFIG = "../DeepStream-Yolo/config_infer_primary_yolo11.txt"
    SGIE_CONFIG = "../DeepStream-Yolo/config_infer_secondary_yolo11.txt"
    TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

    if len(sys.argv) >= 2:
        INPUT_VIDEO = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_VIDEO = sys.argv[2]

    main(INPUT_VIDEO, OUTPUT_VIDEO, PGIE_CONFIG, SGIE_CONFIG, TRACKER_CONFIG)
