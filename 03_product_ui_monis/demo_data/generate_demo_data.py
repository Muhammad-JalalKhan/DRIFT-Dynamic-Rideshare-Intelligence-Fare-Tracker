"""
DRIFT — Deterministic Synthetic Demo Dataset Generator
Generates 500 realistic, synthetic NYC-style trip records strictly for
interactive interface demonstration (Demo Mode).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def generate_demo_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # 1. Timestamps spanning 7 days (Jan 1 - Jan 7, 2024)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    hour_probs = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 00-05
        0.04, 0.07, 0.09, 0.08, 0.05, 0.05,  # 06-11
        0.06, 0.05, 0.05, 0.06, 0.08, 0.09,  # 12-17
        0.08, 0.06, 0.05, 0.04, 0.03, 0.02   # 18-23
    ])
    hour_probs /= hour_probs.sum()

    days = np.random.randint(0, 7, size=n)
    hours = np.random.choice(24, size=n, p=hour_probs)
    minutes = np.random.randint(0, 60, size=n)
    seconds = np.random.randint(0, 60, size=n)

    pickup_times = [
        start_time + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(days, hours, minutes, seconds)
    ]

    # 2. Realistic NYC Borough Distribution
    boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    pu_probs = [0.70, 0.16, 0.10, 0.03, 0.01]
    do_probs = [0.62, 0.19, 0.13, 0.04, 0.02]

    pu_boroughs = np.random.choice(boroughs, size=n, p=pu_probs)
    do_boroughs = np.random.choice(boroughs, size=n, p=do_probs)

    # 3. Passenger Counts
    p_counts = np.random.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.70, 0.15, 0.05, 0.04, 0.03, 0.03])

    # 4. Trip Distance (miles) — Gamma distribution with median ~3 miles
    distances = np.round(np.random.gamma(shape=2.5, scale=1.4, size=n) + 0.3, 2)
    distances = np.clip(distances, 0.4, 25.0)

    # 5. Trip Duration (minutes)
    speeds_mph = np.random.uniform(8.0, 18.0, size=n)
    durations = np.round((distances / speeds_mph) * 60.0 + np.random.uniform(1.0, 4.0, size=n), 1)
    durations = np.clip(durations, 3.0, 75.0)

    dropoff_times = [
        p + timedelta(minutes=float(dur))
        for p, dur in zip(pickup_times, durations)
    ]

    # 6. Fare Amount ($): Base $3.00 + Distance ($3.10/mi) + Duration ($0.45/min) + noise
    fares = np.round(3.00 + (distances * 3.10) + (durations * 0.45) + np.random.uniform(-1.5, 2.5, size=n), 2)
    fares = np.clip(fares, 5.00, 115.00)

    # 7. Zone IDs (1 to 263)
    pu_zones = np.random.randint(1, 264, size=n)
    do_zones = np.random.randint(1, 264, size=n)

    # 8. Simulated cluster assignments for Demo Mode (clearly labeled as simulation)
    clusters = []
    for dist, fare, hour in zip(distances, fares, hours):
        if dist > 10.0 or fare > 50.0:
            clusters.append(2)  # Long-distance / Airport corridor
        elif hour in [22, 23, 0, 1, 2, 3]:
            clusters.append(3)  # Nightlife / Late-night mobility
        elif dist > 4.0:
            clusters.append(1)  # Outer-borough commuter
        else:
            clusters.append(0)  # Core Midtown short hops

    # 9. Simulated anomalies (~3% unusual observations)
    is_anomaly = [
        1 if (f > 65.0 and d < 3.0) or (d > 18.0 and dur < 15.0) or (i % 35 == 0) else 0
        for i, (f, d, dur) in enumerate(zip(fares, distances, durations))
    ]

    df = pd.DataFrame({
        "VendorID": np.random.choice([1, 2], size=n),
        "tpep_pickup_datetime": [p.strftime("%Y-%m-%d %H:%M:%S") for p in pickup_times],
        "tpep_dropoff_datetime": [d.strftime("%Y-%m-%d %H:%M:%S") for d in dropoff_times],
        "passenger_count": p_counts,
        "trip_distance": distances,
        "fare_amount": fares,
        "duration_min": durations,
        "PU_borough": pu_boroughs,
        "DO_borough": do_boroughs,
        "PULocationID": pu_zones,
        "DOLocationID": do_zones,
        "hour_of_day": hours,
        "day_of_week": [p.weekday() for p in pickup_times],
        "cluster": clusters,
        "is_anomaly": is_anomaly
    })

    # Chronological sort
    df["dt"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df = df.sort_values("dt").drop(columns=["dt"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_csv = out_dir / "synthetic_demo_trips.csv"
    df = generate_demo_dataset(500, seed=42)
    df.to_csv(out_csv, index=False)
    print(f"Generated {out_csv} with {len(df)} records.")
