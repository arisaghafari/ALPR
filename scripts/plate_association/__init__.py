"""
Plate-Vehicle Association Module

This module provides algorithms for associating detected license plates
with their parent vehicles in high-density traffic scenarios.

Components:
- scoring: Multiple scoring factors for plate-vehicle matching
- grid_lookup: Spatial grid for O(1) vehicle lookup
- skip_logic: "Read Once, Skip Later" GPU optimization
"""

from .scoring import PlateVehicleScorer
from .grid_lookup import SpatialGrid
from .skip_logic import SkipLogicManager, create_pre_sgie_probe

__all__ = [
    'PlateVehicleScorer', 
    'SpatialGrid',
    'SkipLogicManager',
    'create_pre_sgie_probe'
]
