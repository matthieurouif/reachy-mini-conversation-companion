"""Coordinate conversion utilities for celestial objects."""

from typing import Tuple
from datetime import datetime

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord, EarthLocation


def ra_dec_to_alt_az(
    ra_hours: float,
    dec_degrees: float,
    latitude: float,
    longitude: float,
    time: datetime
) -> Tuple[float, float]:
    """Convert Right Ascension and Declination to Altitude and Azimuth.

    Args:
        ra_hours: Right Ascension in decimal hours (0-24)
        dec_degrees: Declination in decimal degrees (-90 to +90)
        latitude: Observer's latitude in decimal degrees
        longitude: Observer's longitude in decimal degrees
        time: Observation time (datetime object)

    Returns:
        Tuple of (azimuth, altitude) in decimal degrees
        - Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
        - Altitude: 0° = horizon, 90° = zenith, negative = below horizon

    """
    # Create celestial coordinate with RA/Dec
    celestial_coord = SkyCoord(
        ra=ra_hours * u.hourangle,
        dec=dec_degrees * u.deg,
        frame='icrs'
    )

    # Create observer location
    observer_location = EarthLocation(
        lat=latitude * u.deg,
        lon=longitude * u.deg
    )

    # Create observation time
    obs_time = Time(time)

    # Transform to Alt/Az frame for this observer and time
    altaz_frame = AltAz(obstime=obs_time, location=observer_location)
    altaz_coord = celestial_coord.transform_to(altaz_frame)

    # Return as decimal degrees
    return (altaz_coord.az.degree, altaz_coord.alt.degree)
