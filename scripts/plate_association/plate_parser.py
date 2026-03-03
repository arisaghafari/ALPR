"""
Plate format validation per vehicle type.

All vehicles except Moto: 2 Letters + 3 digits + 2 Letters (e.g. AB123CD)
Moto only: no format constraint (length >= 4)
"""

import re

# Standard Italian-style plate: 2 letters + 3 digits + 2 letters (7 chars)
PLATE_FORMAT_REGEX = re.compile(r"^[A-Za-z]{2}[0-9]{3}[A-Za-z]{2}$")

# Vehicle types with relaxed format (no strict 2L+3D+2L)
RELAXED_FORMAT_TYPES = frozenset({"moto"})


def is_valid_plate_format(plate_text: str, vehicle_type: str) -> bool:
    """
    Check if plate text matches the expected format for the vehicle type.

    All vehicles except Moto: must match 2 Letters + 3 digits + 2 Letters.
    Moto only: any plate with length >= 4.

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

    # Moto: relaxed format (length >= 4)
    if vehicle_type and vehicle_type.lower() in RELAXED_FORMAT_TYPES:
        return len(cleaned) >= 4

    # All others (Car, Van, Bus, Truck, etc.): strict format 2L+3D+2L
    return bool(PLATE_FORMAT_REGEX.match(cleaned))


class VehiclePlateParser:
    """Parser for vehicle-specific plate formats (extensible)."""
    
    @staticmethod
    def validate(plate_text: str, vehicle_type: str) -> bool:
        """Validate plate format for vehicle type."""
        return is_valid_plate_format(plate_text, vehicle_type)


def get_plate_parser():
    """Return the default plate parser instance."""
    return VehiclePlateParser()