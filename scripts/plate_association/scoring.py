"""
Multiple Scoring Factors for Plate-Vehicle Association

This module calculates a combined score based on:
1. Bottom-edge distance (plates are at bottom of vehicles)
2. Horizontal centering (plates are centered horizontally)
3. Size ratio (plate is 2-10% of vehicle area)
"""


class PlateVehicleScorer:
    """
    Calculate association score between a plate and a vehicle
    using multiple geometric factors.
    """
    
    # Default weights (can be adjusted)
    WEIGHT_BOTTOM = 0.50      # 50% importance
    WEIGHT_HORIZONTAL = 0.30  # 30% importance
    WEIGHT_SIZE = 0.20        # 20% importance
    
    # Size ratio bounds (plate area / vehicle area)
    MIN_SIZE_RATIO = 0.02  # 2%
    MAX_SIZE_RATIO = 0.10  # 10%
    
    def __init__(self, weight_bottom=None, weight_horizontal=None, weight_size=None):
        """
        Initialize scorer with optional custom weights.
        
        Args:
            weight_bottom: Weight for bottom-edge factor (default 0.50)
            weight_horizontal: Weight for horizontal centering (default 0.30)
            weight_size: Weight for size ratio (default 0.20)
        """
        if weight_bottom is not None:
            self.WEIGHT_BOTTOM = weight_bottom
        if weight_horizontal is not None:
            self.WEIGHT_HORIZONTAL = weight_horizontal
        if weight_size is not None:
            self.WEIGHT_SIZE = weight_size
    
    def score_bottom_edge(self, plate_cy, vehicle_top, vehicle_height):
        """
        Score based on plate's proximity to vehicle's bottom edge.
        
        Plates are typically at the bottom of vehicles (bumper area).
        Score is 1.0 when plate is at bottom, decreases towards top.
        
        Args:
            plate_cy: Plate center Y coordinate
            vehicle_top: Vehicle bounding box top Y
            vehicle_height: Vehicle bounding box height
        
        Returns:
            Score between 0.0 and 1.0
        """
        if vehicle_height <= 0:
            return 0.0
        
        vehicle_bottom = vehicle_top + vehicle_height
        distance = abs(vehicle_bottom - plate_cy)
        normalized = distance / vehicle_height
        
        # Score: 1.0 at bottom, 0.0 at middle or above
        score = max(0.0, 1.0 - normalized / 0.5)
        return score
    
    def score_horizontal_centering(self, plate_cx, vehicle_left, vehicle_width):
        """
        Score based on how centered the plate is horizontally in the vehicle.
        
        Plates are typically centered horizontally on vehicles.
        Score is 1.0 when perfectly centered, decreases towards edges.
        
        Args:
            plate_cx: Plate center X coordinate
            vehicle_left: Vehicle bounding box left X
            vehicle_width: Vehicle bounding box width
        
        Returns:
            Score between 0.0 and 1.0
        """
        if vehicle_width <= 0:
            return 0.0
        
        vehicle_cx = vehicle_left + vehicle_width / 2
        distance = abs(plate_cx - vehicle_cx)
        normalized = distance / (vehicle_width / 2)
        
        # Score: 1.0 at center, 0.0 at edge
        score = max(0.0, 1.0 - normalized)
        return score
    
    def score_size_ratio(self, plate_width, plate_height, vehicle_width, vehicle_height):
        """
        Score based on plate size relative to vehicle size.
        
        Typical plates are 2-10% of vehicle area. Score is highest
        in this range, decreases outside.
        
        Args:
            plate_width: Plate bounding box width
            plate_height: Plate bounding box height
            vehicle_width: Vehicle bounding box width
            vehicle_height: Vehicle bounding box height
        
        Returns:
            Score between 0.0 and 1.0
        """
        plate_area = plate_width * plate_height
        vehicle_area = vehicle_width * vehicle_height
        
        if vehicle_area <= 0:
            return 0.0
        
        ratio = plate_area / vehicle_area
        
        # Score based on ratio
        if self.MIN_SIZE_RATIO <= ratio <= self.MAX_SIZE_RATIO:
            return 1.0
        elif ratio < self.MIN_SIZE_RATIO:
            # Too small - linearly scale
            return ratio / self.MIN_SIZE_RATIO
        else:
            # Too large - linearly decrease
            return max(0.0, 1.0 - (ratio - self.MAX_SIZE_RATIO) / self.MAX_SIZE_RATIO)
    
    def is_plate_inside_vehicle(self, plate_cx, plate_cy, 
                                 vehicle_left, vehicle_top, 
                                 vehicle_width, vehicle_height):
        """
        Check if plate center is inside vehicle bounding box.
        
        Args:
            plate_cx, plate_cy: Plate center coordinates
            vehicle_left, vehicle_top: Vehicle bbox top-left
            vehicle_width, vehicle_height: Vehicle bbox dimensions
        
        Returns:
            True if plate center is inside vehicle bbox
        """
        return (vehicle_left <= plate_cx <= vehicle_left + vehicle_width and
                vehicle_top <= plate_cy <= vehicle_top + vehicle_height)
    
    def calculate_score(self, plate_rect, vehicle_rect):
        """
        Calculate combined association score between plate and vehicle.
        
        Args:
            plate_rect: Object with left, top, width, height attributes
            vehicle_rect: Object with left, top, width, height attributes
        
        Returns:
            Combined score between 0.0 and 1.0, or 0.0 if plate not inside vehicle
        """
        # Get plate center
        plate_cx = plate_rect.left + plate_rect.width / 2
        plate_cy = plate_rect.top + plate_rect.height / 2
        
        # First check: plate must be inside vehicle
        if not self.is_plate_inside_vehicle(
            plate_cx, plate_cy,
            vehicle_rect.left, vehicle_rect.top,
            vehicle_rect.width, vehicle_rect.height
        ):
            return 0.0
        
        # Calculate individual scores
        s_bottom = self.score_bottom_edge(
            plate_cy, vehicle_rect.top, vehicle_rect.height
        )
        
        s_horizontal = self.score_horizontal_centering(
            plate_cx, vehicle_rect.left, vehicle_rect.width
        )
        
        s_size = self.score_size_ratio(
            plate_rect.width, plate_rect.height,
            vehicle_rect.width, vehicle_rect.height
        )
        
        # Weighted combination
        total_score = (
            self.WEIGHT_BOTTOM * s_bottom +
            self.WEIGHT_HORIZONTAL * s_horizontal +
            self.WEIGHT_SIZE * s_size
        )
        
        return total_score
    
    def find_best_vehicle(self, plate_rect, vehicles, min_score=0.0):
        """
        Find the best matching vehicle for a plate.
        
        Args:
            plate_rect: Plate bounding box (object with left, top, width, height)
            vehicles: List of (vehicle_id, vehicle_rect) tuples
            min_score: Minimum score threshold to accept a match (default 0.0)
        
        Returns:
            Tuple of (best_vehicle_id, best_score) or (0, 0.0) if no match
        """
        best_vehicle_id = 0
        best_score = 0.0
        
        for vehicle_id, vehicle_rect in vehicles:
            score = self.calculate_score(plate_rect, vehicle_rect)
            
            if score > best_score:
                best_score = score
                best_vehicle_id = vehicle_id
        
        if best_score >= min_score:
            return best_vehicle_id, best_score
        
        return 0, 0.0

