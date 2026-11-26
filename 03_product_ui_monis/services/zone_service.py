"""
DRIFT — Zone Service
Architecture Layer: Spatial Reference & Zone Ingestion

Provides access to the official NYC Taxi & Limousine Commission (TLC) 263-zone
spatial taxonomy, including zone names, boroughs, polygon centroids, and
haversine distance calculations.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any


class ZoneService:
    """Manages NYC TLC taxi zones and spatial distance calculations."""

    AIRPORT_LOCATION_IDS = {
        1: "EWR (Newark Airport)",
        132: "JFK Airport",
        138: "LaGuardia Airport (LGA)"
    }

    # Pre-calculated borough fallback coordinates (lat, lon)
    BOROUGH_CENTROIDS = {
        "Manhattan": (40.7831, -73.9712),
        "Brooklyn": (40.6782, -73.9442),
        "Queens": (40.7282, -73.7949),
        "Bronx": (40.8448, -73.8648),
        "Staten Island": (40.5795, -74.1502),
        "EWR": (40.6918, -74.1740)
    }

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.data_dir = base_dir / "data"
        else:
            self.data_dir = Path(data_dir)

        self._zones: List[Dict[str, Any]] = []
        self._zone_by_id: Dict[int, Dict[str, Any]] = {}
        self._load_zones()

    def _load_zones(self) -> None:
        """Loads taxi_zone_lookup.json from project data directory."""
        candidate_paths = [
            self.data_dir / "taxi_zone_lookup.json",
            Path("data/taxi_zone_lookup.json"),
            Path("../data/taxi_zone_lookup.json")
        ]

        loaded = False
        for p in candidate_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self._zones = json.load(f)
                    self._zone_by_id = {z["LocationID"]: z for z in self._zones}
                    loaded = True
                    break
                except Exception:
                    continue

        if not loaded:
            self._create_fallback_zones()

    def _create_fallback_zones(self) -> None:
        """Provides minimal fallback zones if data file is absent."""
        defaults = [
            {"LocationID": 1, "Borough": "EWR", "Zone": "Newark Airport", "lat": 40.6918, "lon": -74.1740, "is_airport": True},
            {"LocationID": 132, "Borough": "Queens", "Zone": "JFK Airport", "lat": 40.6413, "lon": -73.7781, "is_airport": True},
            {"LocationID": 138, "Borough": "Queens", "Zone": "LaGuardia Airport", "lat": 40.7769, "lon": -73.8740, "is_airport": True},
            {"LocationID": 161, "Borough": "Manhattan", "Zone": "Midtown Center", "lat": 40.7570, "lon": -73.9800, "is_airport": False},
            {"LocationID": 162, "Borough": "Manhattan", "Zone": "Midtown East", "lat": 40.7550, "lon": -73.9720, "is_airport": False},
            {"LocationID": 230, "Borough": "Manhattan", "Zone": "Times Sq/Theatre District", "lat": 40.7590, "lon": -73.9850, "is_airport": False},
            {"LocationID": 61, "Borough": "Brooklyn", "Zone": "Crown Heights North", "lat": 40.6700, "lon": -73.9400, "is_airport": False},
            {"LocationID": 79, "Borough": "Manhattan", "Zone": "East Village", "lat": 40.7280, "lon": -73.9820, "is_airport": False},
        ]
        self._zones = defaults
        self._zone_by_id = {z["LocationID"]: z for z in defaults}

    def get_zones(self) -> List[Dict[str, Any]]:
        """Returns all loaded taxi zones."""
        return list(self._zones)

    def get_zone(self, location_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single zone by LocationID."""
        return self._zone_by_id.get(location_id)

    def get_airport_zones(self) -> List[Dict[str, Any]]:
        """Returns major airport zones (JFK 132, LGA 138, EWR 1)."""
        airports = []
        for lid in [132, 138, 1]:
            z = self.get_zone(lid)
            if z:
                airports.append(z)
        return airports

    def get_dropdown_options(self) -> List[Dict[str, Any]]:
        """
        Returns options formatted for dcc.Dropdown.
        Includes zone name, borough, and LocationID.
        """
        options = []
        for z in sorted(self._zones, key=lambda x: (x.get("Borough", ""), x.get("Zone", ""))):
            lid = z["LocationID"]
            boro = z.get("Borough", "Unknown")
            name = z.get("Zone", f"Zone {lid}")
            prefix = "✈ " if z.get("is_airport") else ""
            label = f"{prefix}{name} ({boro} #{lid})"
            options.append({"label": label, "value": lid})
        return options

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates Great-Circle distance between two coordinates in statute miles.
        Uses WGS-84 radius: 3958.8 miles.
        """
        r = 3958.8
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return round(r * c, 2)

    def calculate_distance(self, pu_location_id: int, do_location_id: int) -> float:
        """
        Calculates distance between pickup and dropoff zone centroids.
        If same zone, returns an estimated intra-zone distance of 0.85 miles.
        """
        if pu_location_id == do_location_id:
            return 0.85

        pu = self.get_zone(pu_location_id)
        do = self.get_zone(do_location_id)

        if not pu or not do:
            return 3.5  # Standard NYC trip fallback

        return max(0.5, self.haversine_distance(pu["lat"], pu["lon"], do["lat"], do["lon"]))


# Global singleton instance
zone_service = ZoneService()
