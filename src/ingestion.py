import argparse, json, os, urllib.request
from datetime import date
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

RAW_DIR, OUT_DIR = "data/raw", "data"
UNKNOWN_ZONES = (264, 265)
EARTH_RADIUS_MI = 3958.8

def fetch(url: str) -> str:
    """Download a small reference file once and cache it in data/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, os.path.basename(url))
    if not os.path.exists(path):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path

def month_bounds(month: str):
    """Return the inclusive start and exclusive end date bounds for a month."""
    y, m = map(int, month.split("-"))
    start = date(y, m, 1)
    end = date(y + (m // 12), 1 if m == 12 else m + 1, 1)
    return start.isoformat(), end.isoformat()

def clean_sql(src: str, month: str) -> str:
    """CTE returning only rows that pass every cleaning rule."""
    start, end = month_bounds(month)
    return f"""
    WITH base AS (
        SELECT *,
            date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 60.0 AS duration_min
        FROM read_parquet('{src}')
    ),
    clean AS (
        SELECT *,
            trip_distance / (duration_min / 60.0) AS avg_speed_mph
        FROM base
        WHERE tpep_pickup_datetime >= TIMESTAMP '{start}' 
          AND tpep_pickup_datetime < TIMESTAMP '{end}'
          AND trip_distance BETWEEN 0.1 AND 100.0
          AND fare_amount BETWEEN 2.50 AND 500.0
          AND PULocationID BETWEEN 1 AND 263 
          AND DOLocationID BETWEEN 1 AND 263
          AND duration_min BETWEEN 1.0 AND 180.0
          AND passenger_count BETWEEN 1 AND 6
    )
    SELECT * FROM clean WHERE avg_speed_mph BETWEEN 0.5 AND 90.0
    """

def ingest(con, src: str, month: str, sample: int, seed: int = 42) -> pd.DataFrame:
    cols = """
        tpep_pickup_datetime, tpep_dropoff_datetime, PULocationID, DOLocationID, RatecodeID,
        passenger_count, trip_distance, duration_min, avg_speed_mph, fare_amount, tip_amount,
        tolls_amount, total_amount
    """
    q = f"SELECT {cols} FROM ({clean_sql(src, month)}) USING SAMPLE reservoir({sample} ROWS) REPEATABLE ({seed})"
    return con.execute(q).df()

def zone_hour_demand(con, src: str, month: str) -> pd.DataFrame:
    """Demand must be counted on the FULL month, never on the sample."""
    q = f"""
    SELECT 
        PULocationID, 
        CAST(tpep_pickup_datetime AS DATE) AS date,
        hour(tpep_pickup_datetime) AS hour_of_day,
        count(*) AS pickups, 
        avg(fare_amount) AS mean_fare, 
        avg(trip_distance) AS mean_distance
    FROM ({clean_sql(src, month)}) 
    GROUP BY ALL 
    ORDER BY ALL
    """
    return con.execute(q).df()

def build_zone_table() -> pd.DataFrame:
    gdf = gpd.read_file(fetch(ZONES_URL))
    gdf = gdf.dissolve(by="LocationID", as_index=False)
    gdf["geometry"] = gdf.geometry.make_valid()
    cent = gdf.geometry.centroid
    cent = gpd.GeoSeries(cent, crs=gdf.crs).to_crs(4326)
    gdf["lat"], gdf["lon"] = cent.y.values, cent.x.values
    gdf["area_sq_mi"] = gdf.geometry.area / 27878400.0
    
    look = pd.read_csv(fetch(LOOKUP_URL))
    z = gdf[["LocationID", "lat", "lon", "area_sq_mi"]].merge(look, on="LocationID", how="left")
    z.to_csv(os.path.join(OUT_DIR, "zones.csv"), index=False)
    gdf.to_crs(4326)[["LocationID", "geometry"]].to_file(os.path.join(OUT_DIR, "taxi_zones.geojson"), driver="GeoJSON")
    return z

def haversine_mi(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin((lon2 - lon1) / 2)**2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))

def add_features(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    z = zones.set_index("LocationID")
    for side, col in (("PU", "PULocationID"), ("DO", "DOLocationID")):
        df[f"{side}_lat"] = df[col].map(z["lat"])
        df[f"{side}_lon"] = df[col].map(z["lon"])
        df[f"{side}_borough"] = df[col].map(z["Borough"])
        
    ts = pd.to_datetime(df["tpep_pickup_datetime"])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["peak_hour_flag"] = ((df["day_of_week"] < 5) & df["hour_of_day"].isin([7, 8, 9, 16, 17, 18, 19])).astype(int)
    df["night_flag"] = df["hour_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["is_same_zone"] = (df["PULocationID"] == df["DOLocationID"]).astype(int)
    df["is_airport_trip"] = (df["PULocationID"].isin([1, 132, 138]) | df["DOLocationID"].isin([1, 132, 138])).astype(int)
    
    d = haversine_mi(df["PU_lat"], df["PU_lon"], df["DO_lat"], df["DO_lon"])
    df["centroid_dist_mi"] = d
    df["detour_ratio"] = np.where(df["is_same_zone"] == 0, df["trip_distance"] / d.clip(lower=0.1), np.nan)
    df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"]
    return df

def trim_fare_per_mile(df: pd.DataFrame) -> pd.DataFrame:
    lo, hi = df["fare_per_mile"].quantile([0.01, 0.99])
    return df[df["fare_per_mile"].between(lo, hi)].copy()

CONTRACT = {
    "PULocationID": ("i", 1, 263), "DOLocationID": ("i", 1, 263),
    "trip_distance": ("f", 0.1, 100.0), "duration_min": ("f", 1.0, 180.0),
    "fare_amount": ("f", 2.50, 500.0), "avg_speed_mph": ("f", 0.5, 90.0),
    "hour_of_day": ("i", 0, 23), "day_of_week": ("i", 0, 6),
    "PU_lat": ("f", 40.4, 41.0), "PU_lon": ("f", -74.3, -73.6)
}

def validate(df: pd.DataFrame) -> dict:
    report = {"rows": len(df), "columns": len(df.columns), "nulls": {}}
    for col, (kind, lo, hi) in CONTRACT.items():
        assert col in df.columns, f"missing column {col}"
        assert df[col].dtype.kind == kind, f"{col}: expected kind {kind}, got {df[col].dtype}"
        assert df[col].between(lo, hi).all(), f"{col} outside [{lo}, {hi}]"
    report["nulls"] = {c: int(n) for c, n in df.isna().sum().items()}
    assert df[list(CONTRACT)].notna().all().all(), "nulls in contract columns"
    return report

def run(month: str, sample: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    src = TRIP_URL.format(month=month)
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except Exception:
        pass
        
    print("[1/5] Zone geometry centroids")
    zones = build_zone_table()
    print(f"[2/5] Cleaning & sampling {sample:,} rows in SQL")
    raw = ingest(con, src, month, sample)
    print("[3/5] Feature engineering")
    df = trim_fare_per_mile(add_features(raw, zones))
    print("[4/5] Full-month zone-hour demand")
    demand = zone_hour_demand(con, src, month)
    print("[5/5] Validate + export")
    report = validate(df)
    
    df.to_parquet(f"{OUT_DIR}/processed_trips.parquet", index=False)
    demand.to_parquet(f"{OUT_DIR}/zone_hour_demand.parquet", index=False)
    df.head(1000).to_csv(f"{OUT_DIR}/sample_trips.csv", index=False)
    json.dump(report, open(f"{OUT_DIR}/qa_report.json", "w"), indent=2)
    print(f"Done: {report['rows']:,} rows x {report['columns']} cols")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2024-01")
    ap.add_argument("--sample", type=int, default=200_000)
    a = ap.parse_args()
    run(a.month, a.sample)