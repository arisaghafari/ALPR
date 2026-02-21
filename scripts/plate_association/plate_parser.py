import re


class VehiclePlateParser:
    
    # Format: LLNNNLL = Letter Letter Number Number Number Letter Letter (7 chars)
    CAR_PATTERN = re.compile(r'^[A-Za-z]{2}\s*\d{3}\s*[A-Za-z]{2}$')
    
    ## Motorcycle: often 5-6 chars, e.g. A12BC or AB123 (varies by country)
    #MOTO_PATTERN = re.compile(r'^[A-Za-z0-9]{4,6}$')
    
    ## Truck/Bus/Van: often same as car in EU, or longer
    #TRUCK_PATTERN = re.compile(r'^[A-Za-z]{2}\s*\d{3}\s*[A-Za-z]{2}$')
    #BUS_PATTERN = re.compile(r'^[A-Za-z]{2}\s*\d{3}\s*[A-Za-z]{2}$')
    #VAN_PATTERN = re.compile(r'^[A-Za-z]{2}\s*\d{3}\s*[A-Za-z]{2}$')
    
    # Others / unknown: relaxed - at least 4 alphanumeric
    OTHERS_PATTERN = re.compile(r'^[A-Za-z0-9]{4,}$')
    
    # Map vehicle type names (from labels) to validation pattern
    # Keys are matched case-insensitively, supports prefixes (e.g. "Car" matches "Car w. trailer")
    TYPE_PATTERNS = {
        'car': CAR_PATTERN,
     #   'moto': MOTO_PATTERN,
     #   'motorcycle': MOTO_PATTERN,
     #   'bus': BUS_PATTERN,
     #   'truck': TRUCK_PATTERN,
     #   'van': VAN_PATTERN,
        'others': OTHERS_PATTERN,
    }
    
    def __init__(self):
        """Initialize with default patterns. Override TYPE_PATTERNS in subclass to customize."""
        pass
    
    def _normalize_plate(self, text):
        """Remove extra spaces, uppercase for comparison."""
        if not text:
            return ""
        return text.strip().replace(" ", "").upper()
    
    def _get_pattern_for_type(self, vehicle_type_name):
        """
        Get validation pattern for vehicle type.
        
        Args:
            vehicle_type_name: e.g. "Car", "Car w. trailer", "Moto", "Bus"
            
        Returns:
            re.Pattern or None (None = use relaxed validation)
        """
        if not vehicle_type_name:
            return self.OTHERS_PATTERN
        
        name_lower = vehicle_type_name.lower().strip()
        
        # Check prefix match (Car -> car, Car w. trailer -> car)
        for key, pattern in self.TYPE_PATTERNS.items():
            if name_lower.startswith(key) or key in name_lower:
                return pattern
        
        return self.OTHERS_PATTERN
    
    def is_valid(self, plate_text, vehicle_type_name=None):
        """
        Check if plate text is valid for the given vehicle type.
        
        Args:
            plate_text: Raw plate text from OCR (e.g. "AB123CD", "AB12 3CD")
            vehicle_type_name: Vehicle type from label (e.g. "Car", "Moto")
                             If None, uses relaxed validation (min 4 alphanumeric).
        
        Returns:
            True if plate format is valid for this vehicle type
        """
        if not plate_text or len(plate_text.strip()) < 4:
            return False
        
        normalized = self._normalize_plate(plate_text)
        if len(normalized) < 4:
            return False
        
        pattern = self._get_pattern_for_type(vehicle_type_name)
        return bool(pattern.match(normalized))
    
    def normalize_for_display(self, plate_text):
        """Normalize plate text for consistent display (e.g. AB123CD)."""
        return self._normalize_plate(plate_text) if plate_text else ""


# Singleton for easy use
_default_parser = None


def get_plate_parser():
    """Get the default VehiclePlateParser instance."""
    global _default_parser
    if _default_parser is None:
        _default_parser = VehiclePlateParser()
    return _default_parser
