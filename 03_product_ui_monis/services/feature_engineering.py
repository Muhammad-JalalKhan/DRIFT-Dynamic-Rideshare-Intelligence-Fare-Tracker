"""
DRIFT — Feature Engineering Pipeline
Architecture Layer: Transformation & Metric Derivation

Provides feature transformations aligned with teammate Stage 4 pipelines:
- Duration estimation and average speed calculation
- Log trip distance: np.log1p(trip_distance)
- Airport trip flagging
"""

import math
from typing import Tuple, Dict, Any


def estimate_duration_minutes(
    distance_miles: float,
    time_window: str = "Midday (10AM - 4PM)",
    is_rush_hour: bool = False
) -> float:
    """
    Estimates trip duration in minutes using realistic NYC urban transit speeds:
    - Rush hour / congested city traffic: ~8-12 mph
    - Off-peak midday: ~13-16 mph
    - Late night / early morning: ~18-24 mph
    """
    d = max(0.1, distance_miles)
    window_lower = str(time_window).lower()

    if is_rush_hour or "rush" in window_lower or "peak" in window_lower:
        speed_mph = 10.0 if d < 4.0 else 14.0
    elif "night" in window_lower or "late" in window_lower or "early" in window_lower:
        speed_mph = 18.0 if d < 4.0 else 24.0
    else:
        # Standard midday / evening
        speed_mph = 13.0 if d < 4.0 else 17.0

    duration_hours = d / speed_mph
    duration_min = max(2.0, duration_hours * 60.0)
    return round(duration_min, 1)


def compute_avg_speed(distance_miles: float, duration_minutes: float) -> float:
    """Computes average speed in miles per hour: trip_distance / (duration_min / 60.0)."""
    d = max(0.05, distance_miles)
    dur = max(0.5, duration_minutes)
    speed = d / (dur / 60.0)
    return round(speed, 2)


def compute_log_trip_distance(distance_miles: float) -> float:
    """Computes log1p of trip distance, matching KMeans training feature engineering."""
    return round(math.log1p(max(0.0, distance_miles)), 4)


def is_airport_trip(pu_location_id: int, do_location_id: int) -> bool:
    """Flags trip as airport trip if either pickup or dropoff is JFK (132), LGA (138), or EWR (1)."""
    airports = {1, 132, 138}
    return (pu_location_id in airports) or (do_location_id in airports)
