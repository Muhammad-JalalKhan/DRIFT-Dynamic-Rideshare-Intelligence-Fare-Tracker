import pandas as pd
import duckdb
import os

def export_raw_sample(sample_size: int = 10000, output_path: str = "data/raw_sample.csv"):
    """Exports a raw, uncleaned sample dataset combined with taxi zone names

    for initial team inspection and data understanding.
    """
    print("Fetching raw sample records via DuckDB...") 
    
    # 1. Stream raw trips from cloud CDN
    con = duckdb.connect(database=':memory:')
    raw_trips = con.execute(f"""
        SELECT * 
        FROM 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet'
        LIMIT {sample_size}
    """).df()
    
    # 2. Fetch raw taxi zone lookup table
    lookup = pd.read_csv("https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv")
    
    # 3. Merge raw zone names for easy readability
    raw_sample = raw_trips.merge(
        lookup[['LocationID', 'Borough', 'Zone']], 
        left_on='PULocationID', 
        right_on='LocationID', 
        how='left'
    ).rename(columns={'Borough': 'PUBorough', 'Zone': 'PUZone'}).drop(columns=['LocationID'])
    
    raw_sample = raw_sample.merge(
        lookup[['LocationID', 'Borough', 'Zone']], 
        left_on='DOLocationID', 
        right_on='LocationID', 
        how='left'
    ).rename(columns={'Borough': 'DOBorough', 'Zone': 'DOZone'}).drop(columns=['LocationID'])
    
    # 4. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_sample.to_csv(output_path, index=False)
    print(f"Exported {len(raw_sample)} raw sample rows to '{output_path}'")

if __name__ == "__main__":
    export_raw_sample()