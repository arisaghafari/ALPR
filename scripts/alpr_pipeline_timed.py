#!/usr/bin/env python3
"""
Simplified DeepStream ALPR Pipeline - FPS & Latency Only
"""
import sys
import os
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_VIDEO = "/opt/nvidia/deepstream/deepstream-7.1/sources/alpr_project/sample.mp4"
OUTPUT_VIDEO = "output/output_simple.mp4"
PGIE_CONFIG = "../DeepStream-Yolo/config_infer_primary_yolo11.txt"
SGIE_CONFIG = "../DeepStream-Yolo/config_infer_secondary_yolo11.txt"
TGIE_CONFIG = "../DeepStream-Yolo/config_infer_tertiary_lprnet.txt"
TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvSORT.yml"

# ============================================================================
# TIMING VARIABLES
# ============================================================================
frame_count = 0
pipeline_start_time = None

pgie_times = []
sgie_times = []
tgie_times = []
total_times = []

pgie_start = 0
sgie_start = 0
tgie_start = 0
frame_start = 0

# ============================================================================
# TIMING PROBES (Minimal - just timing, no processing)
# ============================================================================

def pgie_sink_probe(pad, info, data):
    global pgie_start, frame_start
    pgie_start = time.perf_counter()
    frame_start = pgie_start
    return Gst.PadProbeReturn.OK

def pgie_src_probe(pad, info, data):
    global pgie_times
    if pgie_start > 0:
        pgie_times.append((time.perf_counter() - pgie_start) * 1000)
    return Gst.PadProbeReturn.OK

def sgie_sink_probe(pad, info, data):
    global sgie_start
    sgie_start = time.perf_counter()
    return Gst.PadProbeReturn.OK

def sgie_src_probe(pad, info, data):
    global sgie_times
    if sgie_start > 0:
        sgie_times.append((time.perf_counter() - sgie_start) * 1000)
    return Gst.PadProbeReturn.OK

def tgie_sink_probe(pad, info, data):
    global tgie_start
    tgie_start = time.perf_counter()
    return Gst.PadProbeReturn.OK

def tgie_src_probe(pad, info, data):
    global tgie_times
    if tgie_start > 0:
        tgie_times.append((time.perf_counter() - tgie_start) * 1000)
    return Gst.PadProbeReturn.OK

def osd_probe(pad, info, data):
    global frame_count, total_times, frame_start
    frame_count += 1
    
    if frame_start > 0:
        total_times.append((time.perf_counter() - frame_start) * 1000)
    
    # Print progress every 100 frames
    if frame_count % 100 == 0:
        if total_times:
            fps = 1000 / np.mean(total_times[-100:])
            print(f"Frame {frame_count}: {fps:.1f} FPS")
    
    return Gst.PadProbeReturn.OK

# ============================================================================
# MESSAGE HANDLER
# ============================================================================

def on_message(bus, msg, loop):
    if msg.type == Gst.MessageType.EOS:
        print("\nEnd of stream.")
        loop.quit()
    elif msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print(f"Error: {err.message}")
        loop.quit()
    return True

# ============================================================================
# PRINT STATISTICS
# ============================================================================

def print_stats(name, times):
    if times:
        arr = np.array(times)
        print(f"\n{name}:")
        print(f"  Mean:    {np.mean(arr):8.2f} ms")
        print(f"  Median:  {np.median(arr):8.2f} ms")
        print(f"  Min:     {np.min(arr):8.2f} ms")
        print(f"  Max:     {np.max(arr):8.2f} ms")
        print(f"  P95:     {np.percentile(arr, 95):8.2f} ms")

def print_final_statistics():
    global pipeline_start_time
    
    total_time = time.time() - pipeline_start_time
    
    print("\n" + "=" * 60)
    print("LATENCY STATISTICS")
    print("=" * 60)
    
    print_stats("PGIE (Vehicle Detection)", pgie_times)
    print_stats("SGIE (Plate Detection)", sgie_times)
    print_stats("TGIE (LPRNet OCR)", tgie_times)
    print_stats("TOTAL Pipeline", total_times)
    
    print("\n" + "=" * 60)
    print("THROUGHPUT")
    print("=" * 60)
    print(f"\n  Frames Processed: {frame_count}")
    print(f"  Total Time:       {total_time:.2f} seconds")
    print(f"  Pipeline FPS:     {frame_count / total_time:.2f}")
    print("=" * 60)

# ============================================================================
# MAIN
# ============================================================================

def main(input_video, output_video):
    global pipeline_start_time
    
    Gst.init(None)
    
    print("=" * 60)
    print("DeepStream ALPR - FPS & Latency Measurement")
    print("=" * 60)
    print(f"Input:  {input_video}")
    print(f"Output: {output_video}")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

    pipeline_str = (
        f"filesrc location={input_video} ! "
        f"qtdemux ! h264parse ! nvv4l2decoder ! "
        f"m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=720 ! "
        f"nvinfer name=pgie config-file-path={PGIE_CONFIG} ! "
        f"nvtracker name=tracker "
        f"ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
        f"ll-config-file={TRACKER_CONFIG} tracker-width=640 tracker-height=384 "
        f"user-meta-pool-size=1024 ! "
        f"nvinfer name=sgie config-file-path={SGIE_CONFIG} ! "
        f"nvinfer name=tgie config-file-path={TGIE_CONFIG} ! "
        f"nvvideoconvert ! nvdsosd name=osd ! nvvideoconvert ! "
        f"x264enc bitrate=4000000 ! h264parse ! qtmux ! "
        f"filesink location={output_video}"
    )

    pipeline = Gst.parse_launch(pipeline_str)

    # Add timing probes
    pgie = pipeline.get_by_name("pgie")
    sgie = pipeline.get_by_name("sgie")
    tgie = pipeline.get_by_name("tgie")
    osd = pipeline.get_by_name("osd")

    pgie.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, pgie_sink_probe, None)
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, pgie_src_probe, None)
    sgie.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, sgie_sink_probe, None)
    sgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, sgie_src_probe, None)
    tgie.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, tgie_sink_probe, None)
    tgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, tgie_src_probe, None)
    osd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, osd_probe, None)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    print("\nStarting pipeline...")
    pipeline_start_time = time.time()
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopped by user")

    pipeline.set_state(Gst.State.NULL)
    print_final_statistics()
    print(f"\nOutput: {output_video}")

if __name__ == "__main__":
    input_video = INPUT_VIDEO
    output_video = OUTPUT_VIDEO
    
    if len(sys.argv) >= 2:
        input_video = sys.argv[1]
    if len(sys.argv) >= 3:
        output_video = sys.argv[2]
    
    main(input_video, output_video)
