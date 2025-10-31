"""Main API for finding celestial object positions."""

from typing import Any, Dict, Optional
from datetime import UTC, datetime

from .search import find_object
from .catalog import Catalog
from .coordinates import ra_dec_to_alt_az
from .solar_system import get_solar_system_position


# Global catalog instance (loaded once)
_catalog: Optional[Catalog] = None


def get_catalog() -> Catalog:
    """Get or initialize the global catalog."""
    global _catalog
    if _catalog is None:
        _catalog = Catalog()
    return _catalog


def find_celestial_angles(
    name: str,
    latitude: float,
    longitude: float,
    time: Optional[datetime] = None,
    search_threshold: int = 70
) -> Dict[str, Any]:
    """Find a celestial object and calculate its azimuth and altitude.

    Args:
        name: Name of celestial object (supports fuzzy matching)
        latitude: Observer's latitude in decimal degrees (-90 to +90)
        longitude: Observer's longitude in decimal degrees (-180 to +180)
        time: Observation time (defaults to current time if None)
        search_threshold: Minimum fuzzy match score (0-100) to accept

    Returns:
        Dictionary with:
        - 'found': bool - Whether object was found
        - 'object_name': str - Name of matched object (if found)
        - 'match_score': int - Fuzzy match score 0-100 (if found)
        - 'azimuth': float - Azimuth in degrees (if found)
        - 'altitude': float - Altitude in degrees (if found)
        - 'type': str - Object type (star, galaxy, nebula, etc.) (if found)
        - 'error': str - Error message (if not found)

    Example:
        >>> result = find_celestial_angles("polaris", 40.7, -74.0)
        >>> print(f"Azimuth: {result['azimuth']:.2f}°")
        >>> print(f"Altitude: {result['altitude']:.2f}°")

    """
    # Default to current time
    if time is None:
        time = datetime.now(UTC)

    # Load catalog
    catalog = get_catalog()

    # Search for object
    search_result = find_object(name, catalog, threshold=search_threshold)

    if search_result is None:
        return {
            'found': False,
            'error': f"No celestial object found matching '{name}'"
        }

    obj, score = search_result

    # Calculate azimuth and altitude based on object category
    if obj.category == 'solar_system':
        # Use ephemerides for solar system objects (planets, moon, sun)
        azimuth, altitude = get_solar_system_position(
            obj.name,
            latitude,
            longitude,
            time
        )
    else:
        # Use RA/Dec for fixed objects (stars, galaxies, nebulae, etc.)
        azimuth, altitude = ra_dec_to_alt_az(
            obj.ra_hours,
            obj.dec_degrees,
            latitude,
            longitude,
            time
        )

    return {
        'found': True,
        'object_name': obj.name,
        'match_score': score,
        'azimuth': azimuth,
        'altitude': altitude,
        'type': obj.type,
        'ra_hours': obj.ra_hours,
        'dec_degrees': obj.dec_degrees
    }
