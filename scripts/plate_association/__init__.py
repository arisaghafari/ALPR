"""
Plate-Vehicle Association Module

This module provides algorithms for associating detected license plates
with their parent vehicles in high-density traffic scenarios.

Components:
- scoring: Multiple scoring factors for plate-vehicle matching
- grid_lookup: Spatial grid for O(1) vehicle lookup
- temporal: Temporal consistency tracking (coming soon)
"""

from .scoring import PlateVehicleScorer
from .grid_lookup import SpatialGrid

__all__ = ['PlateVehicleScorer', 'SpatialGrid']

