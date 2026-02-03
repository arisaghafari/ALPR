"""
Spatial Grid for O(1) Vehicle Lookup

This module provides a grid-based spatial index for efficiently
finding vehicles that might contain a given plate position.
"""


class SpatialGrid:
    """
    Grid-based spatial index for fast vehicle lookup.
    
    Divides the frame into cells and stores which vehicles
    occupy each cell. Enables O(1) lookup for candidate vehicles.
    """
    
    def __init__(self, cell_size=64):
        """
        Initialize spatial grid.
        
        Args:
            cell_size: Size of each grid cell in pixels (default 64)
        """
        self.cell_size = cell_size
        self.grid = {}  # Key: (grid_x, grid_y), Value: list of (vehicle_id, rect)
    
    def clear(self):
        """Clear all vehicles from the grid."""
        self.grid.clear()
    
    def _get_cell_key(self, x, y):
        """
        Convert pixel coordinates to grid cell key.
        
        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
        
        Returns:
            Tuple (grid_x, grid_y)
        """
        return (int(x) // self.cell_size, int(y) // self.cell_size)
    
    def add_vehicle(self, vehicle_id, rect):
        """
        Add a vehicle to the spatial grid.
        
        The vehicle is registered in all grid cells that its
        bounding box overlaps with.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            rect: Bounding box object with left, top, width, height
        """
        # Calculate grid cells this vehicle occupies
        x1 = int(rect.left) // self.cell_size
        y1 = int(rect.top) // self.cell_size
        x2 = int(rect.left + rect.width) // self.cell_size
        y2 = int(rect.top + rect.height) // self.cell_size
        
        # Register vehicle in all overlapping cells
        for gx in range(x1, x2 + 1):
            for gy in range(y1, y2 + 1):
                cell_key = (gx, gy)
                if cell_key not in self.grid:
                    self.grid[cell_key] = []
                self.grid[cell_key].append((vehicle_id, rect))
    
    def get_candidate_vehicles(self, plate_cx, plate_cy):
        """
        Get candidate vehicles that might contain a plate at given position.
        
        Args:
            plate_cx: Plate center X coordinate
            plate_cy: Plate center Y coordinate
        
        Returns:
            List of (vehicle_id, rect) tuples for vehicles in the same grid cell
        """
        cell_key = self._get_cell_key(plate_cx, plate_cy)
        return self.grid.get(cell_key, [])
    
    def get_all_vehicles_in_region(self, left, top, width, height):
        """
        Get all vehicles that overlap with a given region.
        
        Args:
            left, top: Top-left corner of region
            width, height: Dimensions of region
        
        Returns:
            List of unique (vehicle_id, rect) tuples
        """
        x1 = int(left) // self.cell_size
        y1 = int(top) // self.cell_size
        x2 = int(left + width) // self.cell_size
        y2 = int(top + height) // self.cell_size
        
        seen_ids = set()
        result = []
        
        for gx in range(x1, x2 + 1):
            for gy in range(y1, y2 + 1):
                cell_key = (gx, gy)
                for vehicle_id, rect in self.grid.get(cell_key, []):
                    if vehicle_id not in seen_ids:
                        seen_ids.add(vehicle_id)
                        result.append((vehicle_id, rect))
        
        return result
    
    def __len__(self):
        """Return number of grid cells with vehicles."""
        return len(self.grid)
    
    def get_all_vehicles(self):
        """
        Get all vehicles currently in the grid.
        
        Returns:
            List of (vehicle_id, rect) tuples for all vehicles
        """
        seen_ids = set()
        result = []
        for vehicles in self.grid.values():
            for vehicle_id, rect in vehicles:
                if vehicle_id not in seen_ids:
                    seen_ids.add(vehicle_id)
                    result.append((vehicle_id, rect))
        return result
    
    def stats(self):
        """
        Get statistics about the grid.
        
        Returns:
            Dict with cells_used, total_entries, avg_per_cell
        """
        cells_used = len(self.grid)
        total_entries = sum(len(v) for v in self.grid.values())
        avg_per_cell = total_entries / cells_used if cells_used > 0 else 0
        
        return {
            'cells_used': cells_used,
            'total_entries': total_entries,
            'avg_per_cell': avg_per_cell
        }

