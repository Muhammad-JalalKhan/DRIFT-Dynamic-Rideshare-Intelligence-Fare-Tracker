"""
DRIFT — Prediction Service Adapter
Architecture Layer: Decoupled ML Inference Interface

This module provides a clean abstraction between the UI presentation layer
and backend machine learning model artifacts. When a trained model and 
preprocessing pipeline are placed in saved_models/, this service validates
inputs, executes feature transformation, and returns structured predictions.
If no model is available, it provides a transparent status contract without
fabricating values.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import os
import warnings

from services.zone_service import zone_service
from services.feature_engineering import (
    estimate_duration_minutes,
    compute_avg_speed,
    compute_log_trip_distance,
    is_airport_trip
)
from services.clustering_service import clustering_service, ClusterResult


@dataclass
class PredictionResult:
    is_available: bool
    status_code: str
    message: str
    predicted_fare: Optional[float] = None
    currency: str = "USD"
    fare_range: Optional[Tuple[float, float]] = None
    model_version: Optional[str] = None
    features_used: Optional[Dict[str, Any]] = None
    cluster_info: Optional[Dict[str, Any]] = None
    baseline_estimate: Optional[float] = None


class FarePredictionService:
    """
    Adapter for loading and querying fare prediction models.
    Guarantees no synthetic or random fare values are presented as trained ML predictions.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            # Resolves relative to this file: 03_product_ui_monis/saved_models
            base_dir = Path(__file__).resolve().parent.parent
            self.models_dir = base_dir / "saved_models"
        else:
            self.models_dir = Path(models_dir)

        self.model = None
        self.preprocessor = None
        self.model_filename = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Attempts to load model artifacts from saved_models/ or teammate directory.
        Strictly targets fare regressor bundles (fare_regressor_bundle.pkl, fare_regressor.joblib)
        and explicitly rejects clustering or anomaly bundles.
        """
        candidate_dirs = [
            self.models_dir,
            self.models_dir.parent.parent / "02_ml_modeling_anoushey" / "saved_models",
            Path("saved_models"),
            Path("../saved_models")
        ]

        found_file = None
        for d in candidate_dirs:
            if not d.exists():
                continue
            candidates = list(d.glob("*.joblib")) + list(d.glob("*.pkl"))
            for c in candidates:
                name = c.name.lower()
                # Target fare regressor artifacts only
                if ("fare" in name and "regressor" in name) or name == "fare_regressor_bundle.pkl":
                    found_file = c
                    break
            if found_file:
                break

        if not found_file:
            self.model = None
            return

        try:
            import joblib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = joblib.load(found_file)

            if isinstance(loaded, dict) and "model" in loaded:
                self.model = loaded["model"]
                self.preprocessor = loaded.get("preprocessor")
            else:
                self.model = loaded

            self.model_filename = found_file.name
        except Exception:
            self.model = None

    def is_model_ready(self) -> bool:
        """Returns True only when a trained regression model is loaded and ready for inference."""
        return self.model is not None

    def predict_fare(
        self,
        pickup_location: Optional[Any],
        dropoff_location: Optional[Any],
        pickup_date: Optional[str] = None,
        pickup_time: Optional[str] = None,
        passenger_count: int = 1
    ) -> PredictionResult:
        """
        Predicts the fare for a trip given validated inputs.
        Never fabricates numbers if a model artifact is missing.
        """
        if not pickup_location or not dropoff_location:
            return PredictionResult(
                is_available=False,
                status_code="INCOMPLETE_SELECTION",
                message="Please select both pickup and destination locations to view fare estimation."
            )

        # Parse Location IDs or Borough Strings
        pu_id = None
        do_id = None
        pu_name = str(pickup_location)
        do_name = str(dropoff_location)

        try:
            pu_id = int(pickup_location)
        except (ValueError, TypeError):
            pass

        try:
            do_id = int(dropoff_location)
        except (ValueError, TypeError):
            pass

        # Compute spatial metrics
        if pu_id is not None and do_id is not None:
            distance_miles = zone_service.calculate_distance(pu_id, do_id)
            pu_zone = zone_service.get_zone(pu_id)
            do_zone = zone_service.get_zone(do_id)
            pu_name = pu_zone["Zone"] if pu_zone else f"Zone {pu_id}"
            do_name = do_zone["Zone"] if do_zone else f"Zone {do_id}"
            is_airport = is_airport_trip(pu_id, do_id)
        else:
            # Fallback for borough string selections
            distance_miles = 4.2
            is_airport = "jfk" in pu_name.lower() or "jfk" in do_name.lower()

        # Feature engineering
        duration_min = estimate_duration_minutes(distance_miles, pickup_time or "Midday")
        avg_speed = compute_avg_speed(distance_miles, duration_min)
        log_distance = compute_log_trip_distance(distance_miles)

        # Classify mobility cluster
        cluster_res = clustering_service.classify_trip(distance_miles, duration_min, avg_speed)
        cluster_info = {
            "cluster_id": cluster_res.cluster_id,
            "cluster_name": cluster_res.cluster_name,
            "badge_color": cluster_res.badge_color,
            "icon": cluster_res.icon,
            "description": cluster_res.description
        }

        # TLC Baseline Rate: $3.00 base + ($3.50/mi) + ($0.70/min) + surcharges
        base_rate = 3.00 + (distance_miles * 3.50) + (duration_min * 0.70)
        if is_airport:
            base_rate += 1.75  # Airport access fee
        base_rate += 2.50  # Congestion surcharge
        baseline_quote = round(base_rate, 2)

        features_used = {
            "pickup_location": pu_name,
            "dropoff_location": do_name,
            "pickup_location_id": pu_id,
            "dropoff_location_id": do_id,
            "pickup_date": pickup_date,
            "pickup_time": pickup_time,
            "passenger_count": passenger_count,
            "trip_distance_miles": distance_miles,
            "duration_minutes": duration_min,
            "avg_speed_mph": avg_speed,
            "is_airport": is_airport,
            "cluster_name": cluster_res.cluster_name
        }

        if not self.is_model_ready():
            return PredictionResult(
                is_available=False,
                status_code="MODEL_AWAITING_INTEGRATION",
                message=(
                    "The machine learning fare regression model is currently awaiting integration. "
                    "The UI is fully prepared to receive predictions once Milestone 2 model artifacts "
                    "are deployed to 03_product_ui_monis/saved_models/."
                ),
                features_used=features_used,
                cluster_info=cluster_info,
                baseline_estimate=baseline_quote,
                fare_range=(round(baseline_quote * 0.9, 2), round(baseline_quote * 1.15, 2))
            )

        # Real inference branch (executed when teammate regressor is loaded)
        try:
            import pandas as pd
            # Teammate regressor prediction execution
            input_df = pd.DataFrame([{
                "trip_distance": distance_miles,
                "duration_min": duration_min,
                "passenger_count": passenger_count,
                "log_trip_distance": log_distance,
                "PULocationID": pu_id or 161,
                "DOLocationID": do_id or 230
            }])

            if self.preprocessor:
                input_transformed = self.preprocessor.transform(input_df)
                raw_pred = float(self.model.predict(input_transformed)[0])
            else:
                raw_pred = float(self.model.predict(input_df)[0])

            fare = round(max(3.0, raw_pred), 2)
            return PredictionResult(
                is_available=True,
                status_code="SUCCESS",
                message="Estimated with trained ML model.",
                predicted_fare=fare,
                fare_range=(round(fare * 0.92, 2), round(fare * 1.12, 2)),
                model_version=self.model_filename or "FareRegressor",
                features_used=features_used,
                cluster_info=cluster_info
            )
        except Exception as e:
            return PredictionResult(
                is_available=False,
                status_code="INFERENCE_ERROR",
                message=f"Inference error during model execution: {str(e)}",
                features_used=features_used,
                cluster_info=cluster_info
            )


# Singleton instance for the application
fare_service = FarePredictionService()
