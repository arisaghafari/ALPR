"""
Plate-Vehicle Association Module

This module provides algorithms for associating detected license plates
with their parent vehicles in high-density traffic scenarios.

Components:
- scoring: Multiple scoring factors for plate-vehicle matching
- grid_lookup: Spatial grid for O(1) vehicle lookup
- skip_logic: "Read Once, Skip Later" GPU optimization
- heuristics: Smart filtering for high-density traffic
"""

from .scoring import PlateVehicleScorer
from .grid_lookup import SpatialGrid
from .skip_logic import (
    SkipLogicManager, create_pre_sgie_probe,
    get_heuristics_skipped, is_heuristics_active
)
from .heuristics import HighDensityHeuristics, quick_filter_high_density

__all__ = [
    'PlateVehicleScorer', 
    'SpatialGrid',
    'SkipLogicManager',
    'create_pre_sgie_probe',
    'get_heuristics_skipped',
    'is_heuristics_active',
    'HighDensityHeuristics',
    'quick_filter_high_density'
]
