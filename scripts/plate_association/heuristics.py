"""
Heuristics for High-Density Traffic Scenarios

When many vehicles are present, we can't process all of them efficiently.
This module provides smart prioritization and filtering.

Strategy:
1. Score each vehicle by "plate readability potential"
2. Process only top N vehicles per frame
3. Ensure fairness - don't starve vehicles that stay long
"""

from collections import defaultdict


class HighDensityHeuristics:
    """
    Manages vehicle prioritization in high-density traffic.
    
    Scoring factors:
    - Vehicle size (larger = closer = better plate visibility)
    - Position (center-bottom = optimal viewing angle)
    - Time in frame (longer = more chances, but don't starve new ones)
    - Not yet completed (prioritize unknowns)
    """
    
    # Thresholds
    HIGH_DENSITY_THRESHOLD = 10      # More than this = high density mode
    MAX_PROCESS_PER_FRAME = 8       # Max vehicles to send to SGIE in high density
    MAX_PROCESS_NORMAL = 50          # Max in normal mode
    
    # Scoring weights (must sum to 1.0)
    # Note: All vehicles in queue are non-completed (completed are pre-filtered)
    WEIGHT_SIZE = 0.45               # Larger vehicles score higher (closer = better plate visibility)
    WEIGHT_POSITION = 0.30           # Center-bottom scores higher (optimal viewing angle)
    WEIGHT_FRESHNESS = 0.25          # Vehicles waiting long get priority (fairness)
    
    # Zone definitions (as fraction of frame height)
    ZONE_A_START = 0.6               # Bottom 40% = Zone A (high priority)
    ZONE_B_START = 0.3               # Middle 30% = Zone B (medium)
    # Top 30% = Zone C (low priority, often skip)
    
    def __init__(self, frame_width=1920, frame_height=1080):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Track how long each vehicle has been waiting for processing
        self.vehicle_wait_frames = defaultdict(int)
        
        # Track frames since last processed for fairness
        self.frames_since_processed = defaultdict(int)
        
        # Statistics
        self.stats = {
            'total_filtered': 0,
            'high_density_activations': 0,
        }
    
    def get_zone(self, vehicle_top, vehicle_height):
        """
        Determine which priority zone a vehicle is in.
        
        Args:
            vehicle_top: Y coordinate of vehicle top
            vehicle_height: Vehicle bounding box height
            
        Returns:
            'A' (high), 'B' (medium), or 'C' (low)
        """
        # Use vehicle center Y
        vehicle_cy = vehicle_top + vehicle_height / 2
        relative_y = vehicle_cy / self.frame_height
        
        if relative_y >= self.ZONE_A_START:
            return 'A'  # Bottom of frame - closest, best visibility
        elif relative_y >= self.ZONE_B_START:
            return 'B'  # Middle - medium priority
        else:
            return 'C'  # Top - far away, low priority
    
    def calculate_priority_score(self, vehicle_id, bbox, max_vehicle_area=None):
        """
        Calculate priority score for a vehicle (higher = process first).
        
        Note: Only non-completed vehicles are scored (completed are pre-filtered).
        
        Args:
            vehicle_id: Vehicle tracker ID
            bbox: Vehicle bounding box (object with left, top, width, height)
            max_vehicle_area: Largest vehicle area in frame (for normalization)
            
        Returns:
            Priority score between 0.0 and 1.0
        """
        # Size score (larger = better plate visibility)
        vehicle_area = bbox.width * bbox.height
        if max_vehicle_area and max_vehicle_area > 0:
            size_score = min(1.0, vehicle_area / max_vehicle_area)
        else:
            # Normalize by typical vehicle size
            typical_area = 200 * 150  # 200x150 pixels
            size_score = min(1.0, vehicle_area / typical_area)
        
        # Position score (center-bottom = best viewing angle)
        vehicle_cx = bbox.left + bbox.width / 2
        vehicle_cy = bbox.top + bbox.height / 2
        
        # Horizontal: center is best (1.0), edges are worse (0.5)
        horizontal_center = abs(vehicle_cx - self.frame_width / 2) / (self.frame_width / 2)
        horizontal_score = 1.0 - (horizontal_center * 0.5)
        
        # Vertical: bottom is best (closer to camera)
        vertical_score = vehicle_cy / self.frame_height
        
        position_score = (horizontal_score + vertical_score) / 2
        
        # Freshness score (vehicles waiting too long get boosted for fairness)
        wait_frames = self.frames_since_processed.get(vehicle_id, 0)
        freshness_score = min(1.0, wait_frames / 30)  # Max boost after 30 frames
        
        # Weighted combination (weights sum to 1.0)
        total_score = (
            self.WEIGHT_SIZE * size_score +
            self.WEIGHT_POSITION * position_score +
            self.WEIGHT_FRESHNESS * freshness_score
        )
        
        return total_score
    
    def filter_and_prioritize(self, vehicles, completed_vehicles):
        """
        Filter and prioritize vehicles for processing.
        
        Args:
            vehicles: List of (vehicle_id, bbox) tuples
            completed_vehicles: Set of completed vehicle IDs
            
        Returns:
            List of (vehicle_id, bbox) tuples to process (sorted by priority)
        """
        num_vehicles = len(vehicles)
        
        # Determine if high density mode
        is_high_density = num_vehicles > self.HIGH_DENSITY_THRESHOLD
        
        if is_high_density:
            self.stats['high_density_activations'] += 1
            max_to_process = self.MAX_PROCESS_PER_FRAME
        else:
            max_to_process = self.MAX_PROCESS_NORMAL
        
        # If under limit, process all (but still sort by priority)
        if num_vehicles <= max_to_process:
            # Update wait counters - all will be processed
            for vid, bbox in vehicles:
                self.frames_since_processed[vid] = 0
            return vehicles
        
        # Calculate max vehicle area for normalization
        max_area = max((b.width * b.height for _, b in vehicles), default=1)
        
        # Score all vehicles (all are non-completed at this point)
        scored_vehicles = []
        for vehicle_id, bbox in vehicles:
            score = self.calculate_priority_score(vehicle_id, bbox, max_area)
            scored_vehicles.append((score, vehicle_id, bbox))
        
        # Sort by score (highest first)
        scored_vehicles.sort(reverse=True, key=lambda x: x[0])
        
        # Take top N
        selected = []
        for score, vehicle_id, bbox in scored_vehicles[:max_to_process]:
            selected.append((vehicle_id, bbox))
            self.frames_since_processed[vehicle_id] = 0
        
        # Update wait counters for vehicles NOT processed
        filtered_ids = set(vid for _, vid, _ in scored_vehicles[max_to_process:])
        for vid in filtered_ids:
            self.frames_since_processed[vid] += 1
        
        self.stats['total_filtered'] += len(filtered_ids)
        
        return selected
    
    def should_skip_zone_c(self, num_vehicles):
        """
        Decide if Zone C vehicles should be skipped entirely.
        
        In very high density, skip far-away vehicles.
        
        Args:
            num_vehicles: Total vehicles in frame
            
        Returns:
            True if Zone C should be skipped
        """
        return num_vehicles > self.HIGH_DENSITY_THRESHOLD * 1.5
    
    def get_stats(self):
        """Get heuristic statistics."""
        return {
            'total_filtered': self.stats['total_filtered'],
            'high_density_activations': self.stats['high_density_activations'],
        }
    
    def cleanup(self, active_vehicle_ids):
        """
        Remove tracking data for vehicles no longer in frame.
        
        Args:
            active_vehicle_ids: Set of currently visible vehicle IDs
        """
        # Clean up wait counters
        stale_ids = [vid for vid in self.frames_since_processed 
                     if vid not in active_vehicle_ids]
        for vid in stale_ids:
            del self.frames_since_processed[vid]


# Convenience function for quick filtering
def quick_filter_high_density(vehicles, completed_vehicles, threshold=15, max_process=10):
    """
    Quick filter for high-density scenarios without full HeuristicManager.
    
    Simple logic: If more than threshold vehicles, keep only:
    1. Uncompleted vehicles (priority)
    2. Largest vehicles (by area)
    
    Args:
        vehicles: List of (vehicle_id, bbox) tuples
        completed_vehicles: Set of completed vehicle IDs
        threshold: High density threshold
        max_process: Max vehicles to return
        
    Returns:
        Filtered list of (vehicle_id, bbox) tuples
    """
    if len(vehicles) <= threshold:
        return vehicles
    
    # Separate completed and uncompleted
    uncompleted = [(vid, bbox) for vid, bbox in vehicles 
                   if vid not in completed_vehicles]
    completed = [(vid, bbox) for vid, bbox in vehicles 
                 if vid in completed_vehicles]
    
    # Sort uncompleted by size (largest first)
    uncompleted.sort(key=lambda x: x[1].width * x[1].height, reverse=True)
    
    # Take uncompleted first, then completed if room
    result = uncompleted[:max_process]
    remaining_slots = max_process - len(result)
    
    if remaining_slots > 0:
        completed.sort(key=lambda x: x[1].width * x[1].height, reverse=True)
        result.extend(completed[:remaining_slots])
    
    return result
