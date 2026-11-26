"""
DRIFT — Receipt Fraud Guard & Anomaly Detection Service
Architecture Layer: ML Anomaly Inference & Diagnostic Breakdown

Integrates teammate's trained Isolation Forest bundle:
- File: isoforest_bundle.pkl
- Features: ['trip_distance', 'fare_amount', 'duration_min']
- Threshold: 0.647613235541509
- Decision rule: anomaly_score = -model.score_samples(X) > threshold
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings


@dataclass
class AnomalyResult:
    is_available: bool
    status_code: str
    message: str
    is_anomaly: Optional[bool] = None
    anomaly_score: Optional[float] = None
    threshold: Optional[float] = None
    severity: str = "normal"  # normal, moderate, critical
    diagnostic_reasons: List[str] = field(default_factory=list)
    model_version: Optional[str] = None
    features_used: Optional[Dict[str, float]] = None


class AnomalyService:
    """Manages Isolation Forest inference for rideshare receipt auditing."""

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.models_dir = base_dir / "saved_models"
        else:
            self.models_dir = Path(models_dir)

        self.model = None
        self.threshold = 0.647613235541509
        self.features = ["trip_distance", "fare_amount", "duration_min"]
        self.sklearn_version = None
        self._load_bundle()

    def _load_bundle(self) -> None:
        """Loads isoforest_bundle.pkl if available."""
        candidate_paths = [
            self.models_dir / "isoforest_bundle.pkl",
            self.models_dir.parent.parent / "02_ml_modeling_anoushey" / "saved_models" / "isoforest_bundle.pkl",
            Path("saved_models/isoforest_bundle.pkl"),
            Path("../saved_models/isoforest_bundle.pkl")
        ]

        bundle_path = None
        for p in candidate_paths:
            if p.exists():
                bundle_path = p
                break

        if not bundle_path:
            return

        try:
            import joblib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bundle = joblib.load(bundle_path)

            self.model = bundle.get("model")
            if self.model and hasattr(self.model, "set_params"):
                # Single-thread execution for deterministic and low-latency web inference
                try:
                    self.model.set_params(n_jobs=1)
                except Exception:
                    pass

            self.threshold = float(bundle.get("threshold", self.threshold))
            self.features = bundle.get("features", self.features)
            self.sklearn_version = bundle.get("sklearn_version", "1.6.1")
        except Exception:
            self.model = None

    def is_ready(self) -> bool:
        """Returns True if the Isolation Forest model is loaded and ready."""
        return self.model is not None

    def audit_receipt(
        self,
        trip_distance: float,
        fare_amount: float,
        duration_min: float
    ) -> AnomalyResult:
        """
        Audits a rideshare receipt against the trained Isolation Forest model.
        Returns detailed diagnostic breakdown and anomaly classification.
        """
        if trip_distance is None or fare_amount is None or duration_min is None:
            return AnomalyResult(
                is_available=False,
                status_code="MISSING_INPUTS",
                message="Trip distance, fare amount, and duration are required to audit receipt."
            )

        try:
            d = float(trip_distance)
            f = float(fare_amount)
            dur = float(duration_min)
        except (ValueError, TypeError):
            return AnomalyResult(
                is_available=False,
                status_code="INVALID_NUMERIC_INPUT",
                message="All audit parameters must be valid numeric values."
            )

        if d <= 0 or f <= 0 or dur <= 0:
            return AnomalyResult(
                is_available=False,
                status_code="OUT_OF_RANGE",
                message="Trip distance, fare amount, and duration must be greater than zero."
            )

        features_dict = {
            "trip_distance": round(d, 2),
            "fare_amount": round(f, 2),
            "duration_min": round(dur, 1)
        }

        if not self.is_ready():
            return AnomalyResult(
                is_available=False,
                status_code="MODEL_AWAITING_INTEGRATION",
                message=(
                    "The Isolation Forest anomaly detection model artifact (isoforest_bundle.pkl) "
                    "is not loaded. Awaiting ML model deployment."
                ),
                features_used=features_dict
            )

        try:
            import pandas as pd
            import numpy as np

            # Build feature DataFrame with exact feature names
            df_input = pd.DataFrame([[d, f, dur]], columns=self.features)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_score = -float(self.model.score_samples(df_input)[0])

            score = round(raw_score, 4)
            is_anomaly = bool(score > self.threshold)

            # Generate diagnostic reasons
            reasons: List[str] = []
            fare_per_mile = f / max(0.1, d)
            mph = d / (max(0.5, dur) / 60.0)

            if is_anomaly:
                if fare_per_mile > 15.0 and d < 4.0:
                    reasons.append(
                        f"Exorbitant cost per mile: ${fare_per_mile:.2f}/mi exceeds typical NYC bounds ($3.50–$6.50/mi)."
                    )
                elif f < 8.0 and d > 8.0:
                    reasons.append(
                        f"Abnormally low fare: ${f:.2f} for a {d:.1f}-mile route indicates missing charges or metering failure."
                    )

                if dur > 45.0 and d < 2.0:
                    reasons.append(
                        f"Severe stationary metering: {dur:.1f} minutes elapsed for only {d:.1f} miles ({mph:.1f} mph average)."
                    )
                elif mph > 75.0:
                    reasons.append(
                        f"Unrealistic average speed: {mph:.1f} mph exceeds NYC arterial/expressway limits."
                    )

                if not reasons:
                    reasons.append(
                        f"Trip feature vector occupies low-density outlier space (anomaly score {score:.4f} > threshold {self.threshold:.4f})."
                    )

                severity = "critical" if (score - self.threshold) > 0.08 else "moderate"
                message = "Unusual Fare Pattern Detected: Receipt exhibits anomalous pricing or duration telemetry."
            else:
                reasons.append(
                    f"Fare-to-distance ratio (${fare_per_mile:.2f}/mi) aligns with standard TLC rate curves."
                )
                reasons.append(
                    f"Average speed of {mph:.1f} mph is typical for metropolitan traffic conditions."
                )
                reasons.append(
                    f"Anomaly score ({score:.4f}) is safely below the anomaly threshold ({self.threshold:.4f})."
                )
                severity = "normal"
                message = "Normal Fare Pattern: Receipt verified against trained Isolation Forest baseline."

            return AnomalyResult(
                is_available=True,
                status_code="SUCCESS",
                message=message,
                is_anomaly=is_anomaly,
                anomaly_score=score,
                threshold=round(self.threshold, 4),
                severity=severity,
                diagnostic_reasons=reasons,
                model_version=f"IsolationForest (scikit-learn {self.sklearn_version})",
                features_used=features_dict
            )

        except Exception as e:
            return AnomalyResult(
                is_available=False,
                status_code="INFERENCE_ERROR",
                message=f"Error evaluating anomaly model: {str(e)}",
                features_used=features_dict
            )


# Global singleton instance
anomaly_service = AnomalyService()
