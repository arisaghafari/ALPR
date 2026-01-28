"""
Read Once, Skip Later - GPU Optimization for High-Density Vehicle Scenarios

This module implements an algorithm to skip plate detection/recognition for 
vehicles whose plates have already been confidently read.

Algorithm Flow:
1. Track: Get Object ID from Tracker (e.g., "Car ID #55")
2. Check Memory: Have we successfully read the plate for ID #55?
   - Yes: Tell Secondary Model to skip this car (shrink bbox below min threshold)
   - No: Let Secondary Model process it
3. Update Memory: If high-confidence plate found, add to "Completed List"

This saves GPU by not running plate detection on already-completed vehicles.
"""

from collections import defaultdict
import pyds


class SkipLogicManager:
    """
    Manages the "Read Once, Skip Later" optimization.
    
    Tracks which vehicles have been confidently read and should be skipped
    in subsequent frames to save GPU resources.
    """
    
    def __init__(self, min_confident_readings=6, min_plate_length=5, 
                 max_frames_missing=90):
        """
        Initialize the skip logic manager.
        
        Args:
            min_confident_readings: Number of consistent readings before marking complete
            min_plate_length: Minimum plate text length to consider valid
            max_frames_missing: Frames before removing a vehicle from tracking
            
        Note: SGIE min object size is configured in config files:
              input-object-min-width and input-object-min-height
        """
        self.min_confident_readings = min_confident_readings
        self.min_plate_length = min_plate_length
        self.max_frames_missing = max_frames_missing
        
        # Vehicles whose plates have been confidently read
        self.completed_vehicles = set()
        
        # Store original bboxes before shrinking (to restore for display)
        # Key: (frame_number, vehicle_id), Value: (left, top, width, height)
        self.original_bboxes = {}
        
        # Counter for confident readings per vehicle
        self.confident_readings_count = defaultdict(int)
        
        # Track last frame each vehicle was seen
        self.vehicle_last_seen = {}
        
        # Statistics
        self.stats = {
            'total_skipped': 0,
            'total_processed': 0,
        }
    
    def is_completed(self, vehicle_id):
        """Check if a vehicle's plate has already been read."""
        return vehicle_id in self.completed_vehicles
    
    def get_completed_count(self):
        """Get number of completed vehicles."""
        return len(self.completed_vehicles)
    
    def shrink_bbox_for_skip(self, obj_meta, frame_num):
        """
        Shrink a vehicle's bbox so SGIE skips it.
        
        Call this in pre-SGIE probe for completed vehicles.
        Stores original bbox for later restoration.
        
        Args:
            obj_meta: NvDsObjectMeta for the vehicle
            frame_num: Current frame number
            
        Returns:
            True if bbox was shrunk, False otherwise
        """
        vehicle_id = obj_meta.object_id
        
        if vehicle_id not in self.completed_vehicles:
            return False
        
        # Store original bbox
        rect = obj_meta.rect_params
        self.original_bboxes[(frame_num, vehicle_id)] = (
            rect.left, rect.top, rect.width, rect.height
        )
        
        # Shrink bbox below SGIE's min threshold
        rect.width = 1
        rect.height = 1
        
        self.stats['total_skipped'] += 1
        return True
    
    def restore_bbox(self, obj_meta, frame_num):
        """
        Restore original bbox after SGIE processing.
        
        Call this in post-SGIE probe to restore display.
        
        Args:
            obj_meta: NvDsObjectMeta for the vehicle
            frame_num: Current frame number
            
        Returns:
            True if bbox was restored, False if no stored bbox
        """
        vehicle_id = obj_meta.object_id
        key = (frame_num, vehicle_id)
        
        if key not in self.original_bboxes:
            return False
        
        left, top, width, height = self.original_bboxes.pop(key)
        obj_meta.rect_params.left = left
        obj_meta.rect_params.top = top
        obj_meta.rect_params.width = width
        obj_meta.rect_params.height = height
        
        return True
    
    def record_reading(self, vehicle_id, plate_text, frame_num):
        """
        Record a plate reading for a vehicle.
        
        Marks vehicle as completed after enough confident readings.
        
        Args:
            vehicle_id: Vehicle's tracker ID
            plate_text: Recognized plate text
            frame_num: Current frame number
            
        Returns:
            True if vehicle was newly marked as completed
        """
        self.vehicle_last_seen[vehicle_id] = frame_num
        
        if not plate_text or len(plate_text) < self.min_plate_length:
            return False
        
        self.confident_readings_count[vehicle_id] += 1
        self.stats['total_processed'] += 1
        
        # Check if should mark as completed
        if self.confident_readings_count[vehicle_id] >= self.min_confident_readings:
            if vehicle_id not in self.completed_vehicles:
                self.completed_vehicles.add(vehicle_id)
                return True
        
        return False
    
    def cleanup(self, current_frame):
        """
        Remove tracking data for vehicles no longer in frame.
        
        Call periodically (e.g., every 150 frames).
        
        Args:
            current_frame: Current frame number
        """
        # Find vehicles to remove
        vehicles_to_remove = []
        for vehicle_id, last_frame in self.vehicle_last_seen.items():
            if current_frame - last_frame > self.max_frames_missing:
                vehicles_to_remove.append(vehicle_id)
        
        # Clean up
        for vehicle_id in vehicles_to_remove:
            self.completed_vehicles.discard(vehicle_id)
            self.confident_readings_count.pop(vehicle_id, None)
            self.vehicle_last_seen.pop(vehicle_id, None)
        
        # Clean old bbox entries
        keys_to_remove = [k for k in self.original_bboxes if current_frame - k[0] > 10]
        for key in keys_to_remove:
            self.original_bboxes.pop(key, None)
    
    def get_stats(self):
        """Get statistics about skip logic performance."""
        return {
            'completed_vehicles': len(self.completed_vehicles),
            'total_skipped': self.stats['total_skipped'],
            'total_processed': self.stats['total_processed'],
            'skip_ratio': (self.stats['total_skipped'] / 
                          max(1, self.stats['total_skipped'] + self.stats['total_processed']))
        }


def create_pre_sgie_probe(skip_manager):
    """
    Create a probe function to run BEFORE Secondary GIE.
    
    This probe shrinks bboxes of completed vehicles so SGIE skips them.
    
    Args:
        skip_manager: SkipLogicManager instance
        
    Returns:
        Probe function to attach before SGIE
    """
    def pre_sgie_probe(pad, info, u_data):
        from gi.repository import Gst
        
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
            
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    
                    # Only process vehicles (primary GIE, gie-unique-id=1)
                    if obj_meta.unique_component_id == 1:
                        skip_manager.shrink_bbox_for_skip(obj_meta, frame_num)
                    
                except Exception:
                    pass
                
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        return Gst.PadProbeReturn.OK
    
    return pre_sgie_probe

