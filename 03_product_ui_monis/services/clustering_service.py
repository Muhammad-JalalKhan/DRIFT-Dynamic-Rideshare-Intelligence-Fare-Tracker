"""
DRIFT — K-Means Mobility Clustering Service
Architecture Layer: Stage 4/5 Cluster Ingestion & Classification

Integrates teammate's trained K-Means bundle:
- File: kmeans_bundle.pkl
- Features: ['log_trip_distance', 'duration_min', 'avg_speed_mph']
- Scaler: StandardScaler
- Clusters: {2: 'City Hops', 0: 'Cross-Borough', 1: 'Airport / Long-Haul'}
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import math
import warnings


@dataclass
class ClusterResult:
    is_available: bool
    status_code: str
    message: str
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    badge_color: str = "#06b6d4"
    icon: str = "bi-geo-alt"
    description: Optional[str] = None
    features_used: Optional[Dict[str, float]] = None


class ClusteringService:
    """Classifies trips into mobility archetypes using teammate's K-Means model."""

    CLUSTER_METADATA = {
        2: {
            "name": "City Hops",
            "badge_color": "#06b6d4",  # Cyan
            "icon": "bi-geo-alt",
            "description": "Short-distance intra-borough trip in dense urban street grid."
        },
        0: {
            "name": "Cross-Borough",
            "badge_color": "#8b5cf6",  # Purple
            "icon": "bi-signpost-split",
            "description": "Medium-distance corridor traversing bridge/tunnel connections."
        },
        1: {
            "name": "Airport / Long-Haul",
            "badge_color": "#f59e0b",  # Amber
            "icon": "bi-airplane",
            "description": "High-mileage highway trip connecting airport or outer-perimeter zones."
        }
    }

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.models_dir = base_dir / "saved_models"
        else:
            self.models_dir = Path(models_dir)

        self.model = None
        self.scaler = None
        self.cluster_names = {2: "City Hops", 0: "Cross-Borough", 1: "Airport / Long-Haul"}
        self.raw_features = ["log_trip_distance", "duration_min", "avg_speed_mph"]
        self._load_bundle()

    def _load_bundle(self) -> None:
        """Loads kmeans_bundle.pkl if available."""
        candidate_paths = [
            self.models_dir / "kmeans_bundle.pkl",
            self.models_dir.parent.parent / "02_ml_modeling_anoushey" / "saved_models" / "kmeans_bundle.pkl",
            Path("saved_models/kmeans_bundle.pkl"),
            Path("../saved_models/kmeans_bundle.pkl")
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

            self.model = bundle.get("kmeans")
            self.scaler = bundle.get("scaler")
            self.cluster_names = bundle.get("cluster_names", self.cluster_names)
            self.raw_features = bundle.get("raw_features", self.raw_features)
        except Exception:
            self.model = None
            self.scaler = None

    def is_ready(self) -> bool:
        """Returns True if both scaler and K-Means model are ready."""
        return self.model is not None and self.scaler is not None

    def classify_trip(
        self,
        trip_distance: float,
        duration_min: float,
        avg_speed_mph: Optional[float] = None
    ) -> ClusterResult:
        """Classifies a trip into its mobility archetype."""
        try:
            d = max(0.1, float(trip_distance))
            dur = max(0.5, float(duration_min))
            speed = float(avg_speed_mph) if avg_speed_mph is not None else (d / (dur / 60.0))
        except (ValueError, TypeError):
            return ClusterResult(
                is_available=False,
                status_code="INVALID_INPUT",
                message="Trip parameters must be valid numeric values."
            )

        log_d = math.log1p(d)
        features_dict = {
            "trip_distance": round(d, 2),
            "log_trip_distance": round(log_d, 4),
            "duration_min": round(dur, 1),
            "avg_speed_mph": round(speed, 2)
        }

        if not self.is_ready():
            # Heuristic fallback if model artifact is not loaded
            if d >= 12.0 or speed >= 25.0:
                cid = 1
            elif d >= 4.0:
                cid = 0
            else:
                cid = 2

            meta = self.CLUSTER_METADATA.get(cid, {})
            name = self.cluster_names.get(cid, meta.get("name", f"Cluster {cid}"))
            return ClusterResult(
                is_available=False,
                status_code="HEURISTIC_PREVIEW",
                message="Categorized via spatial heuristic (Awaiting K-Means model deployment).",
                cluster_id=cid,
                cluster_name=name,
                badge_color=meta.get("badge_color", "#06b6d4"),
                icon=meta.get("icon", "bi-geo-alt"),
                description=meta.get("description"),
                features_used=features_dict
            )

        try:
            import pandas as pd
            import warnings

            df_input = pd.DataFrame([[log_d, dur, speed]], columns=self.raw_features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled = self.scaler.transform(df_input)
                cid = int(self.model.predict(scaled)[0])

            meta = self.CLUSTER_METADATA.get(cid, {})
            name = self.cluster_names.get(cid, meta.get("name", f"Cluster {cid}"))

            return ClusterResult(
                is_available=True,
                status_code="SUCCESS",
                message="Classified by trained K-Means model.",
                cluster_id=cid,
                cluster_name=name,
                badge_color=meta.get("badge_color", "#06b6d4"),
                icon=meta.get("icon", "bi-geo-alt"),
                description=meta.get("description"),
                features_used=features_dict
            )

        except Exception as e:
            return ClusterResult(
                is_available=False,
                status_code="INFERENCE_ERROR",
                message=f"Error classifying cluster: {str(e)}",
                features_used=features_dict
            )


# Global singleton instance
clustering_service = ClusteringService()
