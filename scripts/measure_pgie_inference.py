#!/usr/bin/env python3
"""
Measure YOLO Primary Inference Time in DeepStream Pipeline
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
import numpy as np

# Storage for timing
times = []
t_start = 0

def before_inference(pad, info, data):
    """Called BEFORE inference"""
    global t_start
    t_start = time.perf_counter()
    return Gst.PadProbeReturn.OK

def after_inference(pad, info, data):
    """Called AFTER inference"""
    global t_start, times
    t_end = time.perf_counter()
    times.append((t_end - t_start) * 1000)  # ms
    
    if len(times) % 100 == 0:
        arr = np.array(times[-100:])
        print(f"[{len(times)} frames] Inference: {np.mean(arr):.2f} ms | FPS: {1000/np.mean(arr):.1f}")
    
    return Gst.PadProbeReturn.OK

def on_message(bus, msg, loop):
    if msg.type == Gst.MessageType.EOS:
        loop.quit()
    elif msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print(f"Error: {err.message}")
        loop.quit()
    return True

def main(video_path, config_path):
    Gst.init(None)
    
    pipeline = Gst.parse_launch(f"""
        filesrc location={video_path} !
        qtdemux ! h264parse ! nvv4l2decoder !
        m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 !
        nvinfer name=pgie config-file-path={config_path} !
        fakesink sync=0
    """)
    
    # Get primary inference element and add probes
    pgie = pipeline.get_by_name("pgie")
    pgie.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, before_inference, None)
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, after_inference, None)
    
    # Run
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)
    
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    
    pipeline.set_state(Gst.State.NULL)
    
    # Final results
    if times:
        arr = np.array(times)
        print(f"\n{'='*50}")
        print(f"YOLO PRIMARY INFERENCE TIME")
        print(f"{'='*50}")
        print(f"  Frames:   {len(arr)}")
        print(f"  Mean:     {np.mean(arr):.3f} ms")
        print(f"  Median:   {np.median(arr):.3f} ms")
        print(f"  Min:      {np.min(arr):.3f} ms")
        print(f"  Max:      {np.max(arr):.3f} ms")
        print(f"  Std:      {np.std(arr):.3f} ms")
        print(f"  FPS:      {1000/np.mean(arr):.2f}")
        print(f"{'='*50}")

if __name__ == "__main__":
    # Default paths - adjust as needed
    VIDEO = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
    CONFIG = "../DeepStream-Yolo/config_infer_primary_yolo11.txt"

    if len(sys.argv) >= 2:
        VIDEO = sys.argv[1]
    if len(sys.argv) >= 3:
        CONFIG = sys.argv[2]
    
    print(f"Video: {VIDEO}")
    print(f"Config: {CONFIG}")
    main(VIDEO, CONFIG)
