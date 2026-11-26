# DRIFT - Dynamic Rideshare Intelligence & Fare Tracker

End-to-end Knowledge Discovery in Databases (KDD) and rideshare analytics platform built on 2.83M NYC TLC Yellow Taxi trip records.

## Repository Structure

- Notebooks/ & src/ (Muhammad Jalal): Stages 1-4 DuckDB ingestion, cleaning, EDA, scaling & feature engineering
- 02_ml_modeling_anoushey/ (Anoushey Khushhal): Stage 5 FP-Growth rules, Isolation Forest anomaly detection, K-Means validation
- 03_product_ui_monis/ (Monis): Stage 6 Plotly Dash 4.5.1 Rider Quote & Fleet Intelligence web/mobile app
- data/ (Shared): Lightweight runtime artifacts (dashboard_sample.parquet, taxi_zone_lookup.json)
- Documentations/ & Tasks Assignments/ (Shared): KDD reports, proposal, role assignments, and peer reviews
- requirements.txt (Shared): Pinned Python dependencies (scikit-learn==1.6.1, dash==4.5.1)

## Quickstart Setup

1. Clone repo: git clone https://github.com/Muhammad-JalalKhan/DRIFT-Dynamic-Rideshare-Intelligence-Fare-Tracker.git
2. Create venv & install: pip install -r requirements.txt
3. Run app: python 03_product_ui_monis/app.py
4. Full 123 MB dataset (model_ready_trips.parquet) is hosted externally on Google Drive.
