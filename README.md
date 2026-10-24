# DRIFT (Dynamic Rideshare Intelligence & Fare Tracker)

> **Eliminates negotiation friction, reduces unpaid driver idle time, and detects fare anomalies in real time.**

Project DRIFT is a data mining and predictive analytics pipeline built on top of the NYC Taxi & Limousine Commission (TLC) telemetry dataset. Designed as a comprehensive Knowledge Discovery in Databases (KDD) workflow, this project processes millions of monthly trip records to extract actionable urban mobility patterns, engineer dynamic surge-pricing triggers, and establish spatial behavioral clusters.

## 🏗️ Architecture & Tech Stack
* **Data Ingestion & OLAP:** DuckDB, Apache Parquet
* **Data Engineering & EDA:** Python, Pandas, Numpy, Seaborn, Scipy
* **Machine Learning:** Scikit-Learn (K-Means Clustering, FP-Growth, Quantile Regression, Isolation Forest)
* **Spatial Processing:** EPSG:2263 to EPSG:4326 Centroid Projection

## 🚀 KDD Pipeline Stages

### 1. Acquire & Integrate
Streamed high-volume Parquet trip logs (~2.9M rows/month) via DuckDB to prevent memory overhead. Merged raw telemetry with official TLC Taxi Zone shapefiles, computing geometric centroids to translate arbitrary zone IDs into exact geographic coordinates (Latitude/Longitude).

### 2. Data Quality & Auditing
Executed a strict data quality protocol without blindly destroying variance. Utilized IQR fences to investigate anomalies, deliberately retaining legitimate extreme events (e.g., airport flat-fares) to prevent model blindness. Imputed hardware-induced missing values (e.g., `passenger_count`) using statistical Mode.

### 3. Exploratory Data Analysis (EDA)
Profiled the cleaned dataset to expose spatial and temporal realities.
* **Temporal:** Mapped demand by `pickup_hour` to identify rush hour and nightlife peaks.
* **Spatial:** Applied logarithmic scaling to visualize extreme class imbalances across boroughs.
* **Hypothesis Testing:** Conducted SciPy Independent T-Tests to mathematically prove the variance between airport and standard street-hailing fares.

### 4. Transform & Reduce
Engineered the data matrix for machine learning compatibility.
* **Scaling:** Applied Z-Score Standardization to distance/duration and Min-Max Normalization to bounded counts.
* **Dimensionality Reduction:** Dropped target-leaking post-trip variables (`tip_amount`, `total_amount`). Evaluated PCA but retained original features for strict business interpretability.
* **Encoding:** Applied One-Hot Encoding to categorical variables (Time of Day, Boroughs) to create isolated mathematical "ON/OFF" switches, laying the foundation for a dynamic Surge Pricing engine.

### 5. Predictive Modeling (Active Phase)
Currently establishing baseline models for Milestone 2:
* **Association Rules (FP-Growth):** Discovering hidden categorical rules (e.g., *If Night + Manhattan → High Surge Fare*).
* **Clustering (K-Means):** Grouping trips by spatiotemporal behavior (distance, speed, time) rather than arbitrary administrative borders.

## 📂 Repository Structure
```text
├── notebooks/
│   ├── 01_eda.ipynb                 # Stage 3: Profiling & Hypothesis Testing
│   ├── 02_data_transformation.ipynb # Stage 4: Scaling, Binning & Encoding
│   └── 03_modeling_baselines.ipynb  # Stage 5: FP-Growth & K-Means Baselines
├── data/
│   └── .gitkeep                     # Note: Large .parquet files are hosted externally
└── README.md
