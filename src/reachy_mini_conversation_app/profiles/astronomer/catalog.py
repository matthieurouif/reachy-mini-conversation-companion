"""Celestial object catalog management."""

import csv
from typing import Dict, List, Optional
from pathlib import Path


class CelestialObject:
    """Represents a celestial object with its coordinates."""

    def __init__(self, name: str, aliases: List[str], ra_hours: float,
                 dec_degrees: float, obj_type: str, category: str = 'fixed'):
        self.name = name
        self.aliases = aliases
        self.ra_hours = ra_hours  # Right Ascension in decimal hours (0-24)
        self.dec_degrees = dec_degrees  # Declination in decimal degrees (-90 to +90)
        self.type = obj_type
        self.category = category  # 'fixed' for stars/deep sky, 'solar_system' for planets/moon/sun

    def __repr__(self):
        return f"CelestialObject('{self.name}', ra={self.ra_hours:.2f}h, dec={self.dec_degrees:.2f}°)"


class Catalog:
    """Manages the catalog of celestial objects."""

    def __init__(self, csv_path: Optional[Path] = None):
        """Initialize catalog from CSV file.

        Args:
            csv_path: Path to CSV file. If None, uses default database location.

        """
        if csv_path is None:
            # Default to database/celestial_objects.csv relative to project root
            csv_path = Path(__file__).parent / "celestial_objects.csv"

        self.objects: List[CelestialObject] = []
        self.name_index: Dict[str, CelestialObject] = {}

        self._load_catalog(csv_path)

    def _load_catalog(self, csv_path: Path):
        """Load celestial objects from CSV file."""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse aliases (pipe-separated)
                aliases = []
                if row['aliases']:
                    aliases = [a.strip() for a in row['aliases'].split('|')]

                # Get category, default to 'fixed' for backward compatibility
                category = row.get('category', 'fixed')

                obj = CelestialObject(
                    name=row['name'],
                    aliases=aliases,
                    ra_hours=float(row['ra_hours']),
                    dec_degrees=float(row['dec_degrees']),
                    obj_type=row['type'],
                    category=category
                )

                self.objects.append(obj)

                # Index by primary name (lowercase for case-insensitive lookup)
                self.name_index[obj.name.lower()] = obj

                # Index by all aliases
                for alias in aliases:
                    self.name_index[alias.lower()] = obj

    def get_by_exact_name(self, name: str) -> Optional[CelestialObject]:
        """Get object by exact name match (case-insensitive).

        Args:
            name: Object name or alias

        Returns:
            CelestialObject if found, None otherwise

        """
        return self.name_index.get(name.lower())

    def get_all_names(self) -> List[str]:
        """Get all indexed names (primary names and aliases)."""
        return list(self.name_index.keys())

    def __len__(self):
        return len(self.objects)
