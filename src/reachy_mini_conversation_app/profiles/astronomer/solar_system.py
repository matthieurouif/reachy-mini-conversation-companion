"""Solar system object position calculations."""

from typing import Tuple
from datetime import datetime

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, get_body


# Objects supported by astropy's ephemerides
SOLAR_SYSTEM_OBJECTS = {
    'sun', 'moon', 'mercury', 'venus', 'mars',
    'jupiter', 'saturn', 'uranus', 'neptune'
}


def get_solar_system_position(
    name: str,
    latitude: float,
    longitude: float,
    time: datetime
) -> Tuple[float, float]:
    """Calculate azimuth and altitude for solar system objects.

    Args:
        name: Name of the solar system object (sun, moon, planets)
        latitude: Observer's latitude in decimal degrees
        longitude: Observer's longitude in decimal degrees
        time: Observation time (datetime object)

    Returns:
        Tuple of (azimuth, altitude) in decimal degrees
        - Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
        - Altitude: 0° = horizon, 90° = zenith, negative = below horizon

    """
    # Create observer location
    observer_location = EarthLocation(
        lat=latitude * u.deg,
        lon=longitude * u.deg
    )

    # Create observation time
    obs_time = Time(time)

    # Create Alt/Az frame for this observer and time
    altaz_frame = AltAz(obstime=obs_time, location=observer_location)

    # Get body position using astropy's built-in ephemerides
    body_coord = get_body(name.lower(), obs_time, observer_location)

    # Transform to Alt/Az
    altaz_coord = body_coord.transform_to(altaz_frame)

    # Return as decimal degrees
    return (altaz_coord.az.degree, altaz_coord.alt.degree)
