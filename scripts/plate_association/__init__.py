"""
Plate-Vehicle Association Module

This module provides algorithms for associating detected license plates
with their parent vehicles in high-density traffic scenarios.

Components:
- scoring: Multiple scoring factors for plate-vehicle matching
- grid_lookup: Spatial grid for O(1) vehicle lookup
- skip_logic: "Read Once, Skip Later" GPU optimization
- heuristics: Smart filtering for high-density traffic
- plate_parser: Format validation per vehicle type (cars, motorcycles, trucks, etc.)
"""

from .scoring import PlateVehicleScorer
from .grid_lookup import SpatialGrid
from .skip_logic import (
    SkipLogicManager, create_pre_sgie_probe,
    get_heuristics_skipped, is_heuristics_active,
    get_parked_vehicles, cleanup_parked_counts
)
from .heuristics import HighDensityHeuristics, quick_filter_high_density
from .plate_parser import VehiclePlateParser, get_plate_parser

__all__ = [
    'PlateVehicleScorer', 
    'SpatialGrid',
    'SkipLogicManager',
    'create_pre_sgie_probe',
    'get_heuristics_skipped',
    'is_heuristics_active',
    'get_parked_vehicles',
    'cleanup_parked_counts',
    'HighDensityHeuristics',
    'quick_filter_high_density',
    'VehiclePlateParser',
    'get_plate_parser',
]