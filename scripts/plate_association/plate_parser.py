"""
Plate format validation per vehicle type.

For Car only (for now): 2 Letters + 3 digits + 2 Letters (e.g. AB123CD)
For all others: no format constraint (length >= 4)
"""

import re

# Standard Italian-style plate: 2 letters + 3 digits + 2 letters (7 chars)
PLATE_FORMAT_REGEX = re.compile(r"^[A-Za-z]{2}[0-9]{3}[A-Za-z]{2}$")

# Vehicle types that require strict format (2L+3D+2L) - only Car for now
STRICT_FORMAT_TYPES = frozenset({"car"})


def is_valid_plate_format(plate_text: str, vehicle_type: str) -> bool:
    """
    Check if plate text matches the expected format for the vehicle type.
    
    For Car: must match 2 Letters + 3 digits + 2 Letters.
    For all others (Moto, Van, Bus, Truck, etc.): any plate with length >= 4.
    
    Args:
        plate_text: Recognized plate text (may contain spaces - stripped)
        vehicle_type: Vehicle type from PGIE (e.g. "Car", "Moto", "Van")
    
    Returns:
        True if plate format is valid for this vehicle type
    """
    if not plate_text or len(plate_text) < 4:
        return False
    
    # Strip spaces (LPR may output "AB 123 CD")
    cleaned = plate_text.replace(" ", "").strip()
    
    # Car only: strict format 2 letters + 3 digits + 2 letters
    if vehicle_type and vehicle_type.lower() in STRICT_FORMAT_TYPES:
        return bool(PLATE_FORMAT_REGEX.match(cleaned))
    
    # All others: no format constraint
    return len(cleaned) >= 4


class VehiclePlateParser:
    """Parser for vehicle-specific plate formats (extensible)."""
    
    @staticmethod
    def validate(plate_text: str, vehicle_type: str) -> bool:
        """Validate plate format for vehicle type."""
        return is_valid_plate_format(plate_text, vehicle_type)


def get_plate_parser():
    """Return the default plate parser instance."""
    return VehiclePlateParser()
