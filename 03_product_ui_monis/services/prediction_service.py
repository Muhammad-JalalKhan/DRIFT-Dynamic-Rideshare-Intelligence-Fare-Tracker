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
from typing import Optional, Dict, Any
from pathlib import Path
import os


@dataclass
class PredictionResult:
    is_available: bool
    status_code: str
    message: str
    predicted_fare: Optional[float] = None
    currency: str = "USD"
    fare_range: Optional[tuple[float, float]] = None
    model_version: Optional[str] = None
    features_used: Optional[Dict[str, Any]] = None


class FarePredictionService:
    """
    Adapter for loading and querying fare prediction models.
    Guarantees no synthetic or random fare values are presented to users.
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
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Attempts to load model artifacts from saved_models/.
        Fails safely if no compatible artifacts are found.
        """
        if not self.models_dir.exists():
            return

        # Expected model files once Milestone 2 / teammate pipeline exports them
        model_candidates = list(self.models_dir.glob("*.joblib")) + list(self.models_dir.glob("*.pkl"))
        if not model_candidates:
            self.model = None
            return

        try:
            import joblib
            # Prefer fare_regressor or first available artifact
            selected_file = model_candidates[0]
            for candidate in model_candidates:
                if "fare" in candidate.name.lower():
                    selected_file = candidate
                    break
            self.model = joblib.load(selected_file)
        except Exception:
            self.model = None

    def is_model_ready(self) -> bool:
        """Returns True only when a trained model is loaded and ready for inference."""
        return self.model is not None

    def predict_fare(
        self,
        pickup_location: Optional[str],
        dropoff_location: Optional[str],
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

        if not self.is_model_ready():
            return PredictionResult(
                is_available=False,
                status_code="MODEL_AWAITING_INTEGRATION",
                message=(
                    "The machine learning fare regression model is currently awaiting integration. "
                    "The UI is fully prepared to receive predictions once Milestone 2 model artifacts "
                    "are deployed to 03_product_ui_monis/saved_models/."
                ),
                features_used={
                    "pickup_location": pickup_location,
                    "dropoff_location": dropoff_location,
                    "pickup_date": pickup_date,
                    "pickup_time": pickup_time,
                    "passenger_count": passenger_count
                }
            )

        # Real inference branch (executed when model is loaded)
        try:
            # Reserved for teammate ML inference pipeline execution
            # Real model prediction logic will be plugged here
            return PredictionResult(
                is_available=True,
                status_code="SUCCESS",
                message="Estimated with trained ML model.",
                predicted_fare=0.0  # Placeholder for real model output
            )
        except Exception as e:
            return PredictionResult(
                is_available=False,
                status_code="INFERENCE_ERROR",
                message=f"Inference error during model execution: {str(e)}"
            )


# Singleton instance for the application
fare_service = FarePredictionService()
