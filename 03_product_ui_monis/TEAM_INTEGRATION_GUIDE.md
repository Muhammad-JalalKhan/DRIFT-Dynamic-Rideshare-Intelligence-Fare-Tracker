# DRIFT — Team Integration & ML Compatibility Guide (Milestone 4)

## 1. Executive Summary & Architecture Overview

This document serves as the comprehensive integration contract between the **Product UI Layer (`03_product_ui_monis`)** and teammate work streams:
- **ML Lead (Anoushey)**: Machine learning models, clustering archetypes, anomaly detection, and association rules.
- **Data Engineering Lead (Jalal)**: Parquet ingestion, DuckDB pipelines, and spatial shapefile/GeoJSON boundaries.
- **Product UI Lead (Monis)**: Front-end architecture, user experience, live carto-dark spatial visualization, receipt fraud guard, and executive analytics dashboard.

```
                              ┌─────────────────────────────────────────┐
                              │      DRIFT Application Shell (app.py)    │
                              └───────┬─────────────────────────┬───────┘
                                      │                         │
            ┌─────────────────────────┴────────┐       ┌────────┴─────────────────────────┐
            │   Rider Experience (/ride)       │       │   Receipt Fraud Guard (/fraud)   │
            │   - 263 TLC Taxi Zones           │       │   - Isolation Forest Ingestion   │
            │   - Airport Quick Chips          │       │   - Telemetry Audit Engine       │
            │   - Mobility Cluster Badging     │       │   - Metric Explanations          │
            └─────────────────┬────────────────┘       └────────┬─────────────────────────┘
                              │                                 │
                              ▼                                 ▼
                    ┌─────────────────────────────────────────────────────┐
                    │               Decoupled Service Layer               │
                    │  - ZoneService (Centroids & Haversine Distance)     │
                    │  - FeatureEngineering (Speed, Duration, Log-Dist)   │
                    │  - AnomalyService (isoforest_bundle.pkl)            │
                    │  - ClusteringService (kmeans_bundle.pkl)            │
                    │  - RulesService (repositioning & surge JSON rules)  │
                    │  - PredictionService (Fare Regressor Adapter)       │
                    └─────────────────────────┬───────────────────────────┘
                                              │
                    ┌─────────────────────────┴───────────────────────────┐
                    │            Trained Teammate ML Artifacts            │
                    │  - 02_ml_modeling_anoushey/saved_models/            │
                    │  - 03_product_ui_monis/saved_models/                │
                    │  - data/taxi_zones.geojson & taxi_zone_lookup.json  │
                    └─────────────────────────────────────────────────────┘
```

---

## 2. Real ML Model Artifacts & Integration Status

| Component | Teammate Artifact | Serialization | Status | Input Schema |
|---|---|---|---|---|
| **Anomaly Detection** | `isoforest_bundle.pkl` | `joblib` (`sklearn==1.6.1`) | **LIVE & ACTIVE** | `['trip_distance', 'fare_amount', 'duration_min']` |
| **Mobility Clustering** | `kmeans_bundle.pkl` | `joblib` (`sklearn==1.6.1`) | **LIVE & ACTIVE** | `['log_trip_distance', 'duration_min', 'avg_speed_mph']` |
| **Repositioning Rules** | `repositioning_rules.json` | JSON (Records) | **LIVE & ACTIVE** | `antecedent, consequent, support, confidence, lift` |
| **Surge Pricing Rules** | `surge_rules.json` | JSON (Records) | **LIVE & ACTIVE** | `antecedent, consequent, support, confidence, lift` |
| **Fare Regressor** | `fare_regressor_bundle.pkl` | Pending Export | **HONEST PENDING** | `trip_distance, duration_min, passenger_count, PULocationID, DOLocationID` |

### 2.1 Anomaly Detection (`AnomalyService`)
- **Bundle File**: `saved_models/isoforest_bundle.pkl`
- **Trained Model**: `IsolationForest(n_estimators=200, max_samples=1024, contamination=0.01, random_state=42)`
- **Trained Threshold**: `0.647613235541509`
- **Decision Rule**:
  $$\text{anomaly\_score} = -\text{model.score\_samples}(X)[0]$$
  $$\text{is\_anomaly} = \text{anomaly\_score} > \text{threshold}$$
- **Receipt Fraud Guard UI**: Available at route `/fraud-guard`. Allows riders and auditors to input distance, fare paid, and duration to detect meter overcharging and gridlock metering anomalies.

### 2.2 Mobility Clustering (`ClusteringService`)
- **Bundle File**: `saved_models/kmeans_bundle.pkl`
- **Trained Model**: `KMeans(n_clusters=3, n_init=10, random_state=42)` with fitted `StandardScaler`
- **Transformation Pipeline**:
  $$\text{log\_trip\_distance} = \ln(1 + \text{trip\_distance})$$
  $$X = \text{scaler.transform}([[\text{log\_trip\_distance}, \text{duration\_min}, \text{avg\_speed\_mph}]])$$
- **Cluster Archetypes**:
  - `Cluster 2`: **City Hops** (Cyan `#06b6d4`) — Short-distance intra-borough transit in dense traffic grid.
  - `Cluster 0`: **Cross-Borough** (Purple `#8b5cf6`) — Medium-distance arterial corridors traversing bridge/tunnel connections.
  - `Cluster 1`: **Airport / Long-Haul** (Amber `#f59e0b`) — High-mileage highway corridor connecting major airports or outer borough perimeters.

### 2.3 Fare Regressor Contract (`PredictionService`)
- The UI contains an active adapter ready to ingest `fare_regressor_bundle.pkl` or `fare_regressor.joblib`.
- In the absence of teammate model training weights, the UI:
  1. Displays the status: `MODEL_AWAITING_INTEGRATION`.
  2. Previews the exact feature payload prepared for the model.
  3. Displays an honest TLC Yellow Taxi reference tariff calculation without claiming it is a machine learning prediction.

---

## 3. NYC TLC Spatial Integration (263 Zones)

- **Official GeoJSON**: `data/taxi_zones.geojson` containing dissolved polygon boundaries for all 263 TLC zones in EPSG:4326.
- **Centroid Reference**: `data/taxi_zone_lookup.json` maps every `LocationID` (1 to 263) to its official Zone name, Borough, Service Zone, and exact geometric centroid (`lat`, `lon`).
- **Airport Locations**:
  - `1`: Newark Airport (EWR) — Lat 40.6918, Lon -74.1740
  - `132`: JFK International Airport — Lat 40.6470, Lon -73.7865
  - `138`: LaGuardia Airport (LGA) — Lat 40.7769, Lon -73.8740
- **Distance Calculation**:
  Great-circle Haversine formula implemented in pure Python:
  $$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
  Where $R = 3958.8\text{ miles}$.

---

## 4. Environment & Version Compatibility

- **Joblib Deserialization**: All model bundles (`isoforest_bundle.pkl`, `kmeans_bundle.pkl`) were serialized using `joblib` and `scikit-learn==1.6.1`.
- **Note on Unpickling**: Unpickling must be done via `joblib.load()` rather than standard `pickle.load()` because internal estimators (`ExtraTreeRegressor`, `StandardScaler`) require joblib's numpy array reconstruction handlers.
- **Web Worker Concurrency**: `IsolationForest.set_params(n_jobs=1)` is automatically enforced during inference to prevent Windows subprocess loky spawning overhead on single-record web requests.

---

## 5. Mobile SaaS & Deployment Specifications

- **Touch Target Accessibility**: All buttons, dropdowns, and airport chips enforce minimum tap dimensions ($\ge 44\text{px}$ on mobile).
- **iOS Viewport Optimization**: Input elements enforce `font-size: 16px` to prevent aggressive mobile Safari auto-zooming on focus.
- **PWA Capabilities**: Web App Manifest (`assets/manifest.json`) and viewport meta tags enabled (`viewport-fit=cover`, `theme-color: #0b1120`).
- **Production Server**: Production WSGI entry point exposed at `app:server`.

---

## 6. How to Launch and Verify

### 6.1 Running the App Locally
```powershell
cd 03_product_ui_monis
..\.venv\Scripts\python.exe app.py
```
Open browser to `http://127.0.0.1:8050/`.

### 6.2 Running Production WSGI Server
```powershell
gunicorn --bind 0.0.0.0:8050 app:server
```

### 6.3 Running Automated Test Suite
```powershell
cd 03_product_ui_monis
..\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
*Current test suite status: 42/42 tests passing.*
