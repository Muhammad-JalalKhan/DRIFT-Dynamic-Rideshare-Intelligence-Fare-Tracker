# DRIFT Intelligence — Teammate Integration Guide & Data Contracts

> **Module:** Phase 2 — DRIFT Intelligence Dashboard & Mobility Analytics  
> **Directory:** `03_product_ui_monis/`  
> **Target Audience:** Team Data Lead (Muhammad Jalal) & Mining Lead (Anoushey)  
> **Maintainer:** Product Lead (Mohammad Monis)

---

## 1. Executive Summary

The **DRIFT Intelligence** dashboard (`/intelligence`) is built with a decoupled architecture. All components (Executive KPIs, Diurnal Demand Curves, Fare Distributions, Spatial Corridors, and Data Mining Insight Cards) are filter-responsive and live.

The dashboard enforces a strict **Data Integrity Policy**:
- **Zero Fabricated Statistics:** When teammate datasets or ML models are pending, the dashboard displays honest, transparent status indicators rather than inventing placeholder numbers.
- **Auto-Discovery:** As soon as you place your exported data files into `data/` or `03_product_ui_monis/data/`, and model artifacts into `03_product_ui_monis/saved_models/`, the dashboard automatically validates the schema and activates real calculations.

---

## 2. Supported File Locations & Detection Priority

The service layer (`services/analytics_service.py`) checks the following paths in priority order:

1. `03_product_ui_monis/data/<filename>` (local UI data cache)
2. `data/<filename>` (repository root data directory)
3. `03_product_ui_monis/saved_models/<model_name>` (trained models & evaluation JSONs)
4. `saved_models/<model_name>` (root model artifacts)

Supported formats: Apache Parquet (`.parquet`), CSV (`.csv`), Joblib (`.joblib`), Pickle (`.pkl`), JSON (`.json`).

---

## 3. Dataset Contracts (Stage 1–4 Telemetry)

### Primary Dataset File Options
- Preferred: `data/processed_trips.parquet` (Cleaned 2.8M row monthly dataset)
- Sample: `03_product_ui_monis/data/sample_trips.csv` or `data/raw_sample.csv` (10k–50k rows for rapid dev)

### Required Schema & Types

| Column Name | Type | Allowed Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `tpep_pickup_datetime` | `datetime` / `string` | `YYYY-MM-DD HH:MM:SS` | Timestamp of trip initiation |
| `fare_amount` | `float` | `2.50` to `500.00` USD | Metered base fare (excl. tip/tolls) |
| `trip_distance` | `float` | `0.1` to `100.0` miles | Trip odometer distance |
| `PU_borough` | `string` | Manhattan, Brooklyn, Queens, Bronx, Staten Island | Pickup Borough name |
| `DO_borough` | `string` | Manhattan, Brooklyn, Queens, Bronx, Staten Island | Dropoff Borough name |

### Optional / Derived Columns (Enhances Visualizations)

| Column Name | Type | Notes |
| :--- | :--- | :--- |
| `duration_min` | `float` | Trip duration in minutes (`1.0` to `180.0`). Replaces median fare in KPI Card 4. |
| `passenger_count` | `int` | `1` to `6` passengers. Powers Party Size filter. |
| `hour_of_day` | `int` | `0` to `23`. Automatically computed if missing from `tpep_pickup_datetime`. |
| `day_of_week` | `int` | `0` (Mon) to `6` (Sun). Automatically computed if missing. |
| `PULocationID` | `int` | `1` to `263` (TLC Zone ID). Used for spatial mapping. |
| `DOLocationID` | `int` | `1` to `263` (TLC Zone ID). Used for spatial mapping. |

> [!IMPORTANT]  
> **Official Taxi Zone Distinction:**  
> `PULocationID` (integers 1–263) and `PU_borough` (borough strings) are **not interchangeable**. The UI handles both: borough strings power the high-level regional filters, while zone integers feed the spatial model prediction contract. If exporting processed parquet, include both `PU_borough` and `PULocationID`.

---

## 4. Machine Learning & Data Mining Artifact Contracts

The dashboard features dedicated modules for each teammate's data mining track. Connect your outputs by saving files with the specified names:

### A. K-Means Demand Clustering (Mining Track — Stage 5)
- **Artifact:** `03_product_ui_monis/saved_models/demand_kmeans.joblib`  
  *OR* add a `cluster` column (`int`: `0, 1, 2, ...`) directly into `processed_trips.parquet`.
- **Features used for clustering:** `trip_distance`, `duration_min`, `hour_of_day`, `PU_lat`, `PU_lon`.
- **Dashboard Action:** Automatically renders cluster size distributions and behavioral centroid profiles.

### B. Isolation Forest Anomaly Detection (Mining Track — Stage 5)
- **Artifact:** `03_product_ui_monis/saved_models/isolation_forest.joblib`  
  *OR* add an `is_anomaly` column (`1` = anomalous, `0` = normal) into `processed_trips.parquet`.
- **Features inspected:** `fare_amount`, `trip_distance`, `duration_min`, `avg_speed_mph`.
- **Dashboard Action:** Automatically computes anomaly rate, flagged record totals, and normal vs anomalous comparisons.

### C. Fare Regression Evaluation (Model Evaluation — Stage 6)
- **Artifact:** `03_product_ui_monis/saved_models/regression_eval.json`
- **Expected JSON Structure:**
  ```json
  {
    "model_name": "Quantile Regressor (Alpha 0.5)",
    "mae": 1.82,
    "rmse": 3.14,
    "r2": 0.894,
    "test_samples": 565605,
    "features": ["trip_distance", "hour_of_day", "PU_borough", "DO_borough", "passenger_count"]
  }
  ```
- **Dashboard Action:** Unlocks the Model Performance KPI badges and test-set residual metrics.

### D. FP-Growth Association Rules (Pattern Mining — Stage 5)
- **Artifact:** `03_product_ui_monis/saved_models/association_rules.csv`
- **Expected Columns:** `antecedents, consequents, support, confidence, lift`
- **Dashboard Action:** Displays frequent itemsets and surge pricing triggers (e.g., `{Night, Manhattan} → High Surge`).

---

## 5. How to Run & Test the Dashboard

### 1. Launch Application (Windows PowerShell)
```powershell
# Navigate to the UI project root
cd "03_product_ui_monis"

# Run Dash application using the project virtual environment
..\.venv\Scripts\python.exe app.py
```
Access the application in your browser:
- Portal / Welcome: `http://127.0.0.1:8050/`
- DRIFT Ride (Phase 1): `http://127.0.0.1:8050/ride`
- DRIFT Intelligence (Phase 2): `http://127.0.0.1:8050/intelligence`

### 2. Run Automated Test Suite
```powershell
cd "03_product_ui_monis"

# Run Phase 1 Rider tests (6 tests)
..\.venv\Scripts\python.exe -m unittest tests/test_ui.py

# Run Phase 2 Intelligence tests (14 tests)
..\.venv\Scripts\python.exe -m unittest tests/test_intelligence.py

# Run all 20 tests together
..\.venv\Scripts\python.exe -m unittest discover tests
```

---

## 6. Testing with Synthetic Fixtures vs. Normal Runtime

- **Automated Tests:** Unit tests load a small, 12-row deterministic synthetic dataset located strictly under `03_product_ui_monis/tests/fixtures/synthetic_trips.csv`. This verifies all 8 Plotly charts, KPI mathematical formulas, multi-criteria filters, and error handlers without needing external downloads.
- **Normal Runtime Isolation:** The production dashboard service (`analytics_service`) searches only `data/` and `03_product_ui_monis/data/`. It **never** loads or displays synthetic test data in production. In the absence of real teammate files, it safely renders honest "Awaiting Dataset Snapshot" empty states.

