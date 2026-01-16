#!/usr/bin/env python3
"""
DeepStream YOLO11 - Detection with Confidence Scores + Save Output
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

# Load labels
def load_labels(label_file):
    try:
        with open(label_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except:
        return None

LABELS = load_labels("../DeepStream-Yolo/labels.txt")
#LABELS = [
#    "license_plate",
#    "car",
#    "truck",
#    "motorcycle"
#]
frame_count = 0

def osd_sink_pad_buffer_probe(pad, info, data):
    """Add confidence score to each detected object"""
    global frame_count
    
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
        obj_count = 0
        
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                
                # Get bounding box
                rect = obj_meta.rect_params
                x = int(rect.left)
                y = int(rect.top)
                w = int(rect.width)
                h = int(rect.height)

                # Get label name
                if LABELS and class_id < len(LABELS):
                    label = LABELS[class_id]
                else:
                    label = f"Class_{class_id}"

                # Set display text with confidence
                display_text = f"{label}: {confidence*100:.1f}%"
                obj_meta.text_params.display_text = display_text
                
                # Style the text
                obj_meta.text_params.font_params.font_size = 12
                obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)  # White
                obj_meta.text_params.set_bg_clr = 1
                obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)  # Semi-transparent black
                
                obj_count += 1

                l_obj = l_obj.next
            except StopIteration:
                break

        # Print info every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {obj_count} objects detected")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def on_message(bus, msg, loop):
    msg_type = msg.type
    if msg_type == Gst.MessageType.EOS:
        print(f"\nEnd of stream. Total frames processed: {frame_count}")
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


def main(input_video, output_video, config_file):
    Gst.init(None)

    print(f"Input video:  {input_video}")
    print(f"Output video: {output_video}")
    print(f"Config file:  {config_file}")
    print("-" * 50)

    # Build pipeline
    pipeline_str = f"""
        filesrc location={input_video} !
        qtdemux name=demux !
        h264parse !
        nvv4l2decoder !
        m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 !
        nvinfer config-file-path={config_file} !
        nvvideoconvert !
        nvdsosd name=osd !
        nvvideoconvert !
        x264enc bitrate=4000 !
        h264parse !
        qtmux !
        filesink location={output_video}
    """

    pipeline = Gst.parse_launch(pipeline_str)

    # Add probe to OSD to modify display text
    osd = pipeline.get_by_name("osd")
    osd_sink_pad = osd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    # Setup message handling
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    # Start pipeline
    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Cleanup
    pipeline.set_state(Gst.State.NULL)
    print(f"Output saved to: {output_video}")


if __name__ == "__main__":
    # Default values
    INPUT_VIDEO = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
    OUTPUT_VIDEO = "output/output_with_detections_confidence.mp4"
    CONFIG_FILE = "../DeepStream-Yolo/config_infer_primary_yolo11.txt"

    # Parse arguments
    if len(sys.argv) >= 2:
        INPUT_VIDEO = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_VIDEO = sys.argv[2]
    if len(sys.argv) >= 4:
        CONFIG_FILE = sys.argv[3]

    main(INPUT_VIDEO, OUTPUT_VIDEO, CONFIG_FILE)
