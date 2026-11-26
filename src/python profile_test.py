import duckdb

# 1. Connect to DuckDB in memory
con = duckdb.connect()

# Link to the official January 2024 NYC TLC Parquet file
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"

print("--- 1. CHECKING TOTAL ROW COUNT ---")
row_count = con.execute(f"SELECT count(*) FROM '{url}'").fetchone()[0]
print(f"Total rows in file: {row_count:,}\n")

print("--- 2. CHECKING COLUMNS & DATA TYPES ---")
schema_rows = con.execute(f"DESCRIBE SELECT * FROM '{url}'").fetchall()
print("column_name column_type")
for column_name, column_type, *_ in schema_rows[:10]:
    print(f"{column_name} {column_type}")
print("\n")

print("--- 3. SUMMARIZING STATISTICS & NULLS ---")
summary_rows = con.execute(f"SUMMARIZE SELECT * FROM '{url}'").fetchall()
summary_columns = [
    column[0] for column in con.description
]
summary_indexes = {
    column: summary_columns.index(column)
    for column in ["column_name", "min", "max", "null_percentage"]
}
print("column_name min max null_percentage")
for row in summary_rows[:10]:
    print(" ".join(str(row[summary_indexes[column]]) for column in summary_indexes))
