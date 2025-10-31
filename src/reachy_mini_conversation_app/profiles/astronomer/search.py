"""Fuzzy search for celestial objects."""

from typing import Tuple, Optional

from rapidfuzz import fuzz, process

from .catalog import Catalog, CelestialObject


def find_object(query: str, catalog: Catalog, threshold: int = 70) -> Optional[Tuple[CelestialObject, int]]:
    """Find a celestial object by name using fuzzy matching.

    Args:
        query: Search query (object name)
        catalog: Catalog to search in
        threshold: Minimum similarity score (0-100) to return a match

    Returns:
        Tuple of (CelestialObject, score) if match found, None otherwise
        Score is 0-100, where 100 is perfect match

    """
    # First try exact match (case-insensitive)
    exact_match = catalog.get_by_exact_name(query)
    if exact_match:
        return (exact_match, 100)

    # Get all indexed names for fuzzy matching
    all_names = catalog.get_all_names()

    if not all_names:
        return None

    # Use fuzzy matching to find best match
    result = process.extractOne(
        query.lower(),
        all_names,
        scorer=fuzz.WRatio  # Weighted ratio works well for partial matches
    )

    if result is None:
        return None

    matched_name, score, _ = result

    # Only return if score meets threshold
    if score >= threshold:
        obj = catalog.get_by_exact_name(matched_name)
        return (obj, int(score))

    return None
