"""
DRIFT — Association Rules Service
Architecture Layer: Stage 5 FP-Growth & Rule Mining Integration

Loads and queries association rules extracted by teammate pipelines:
- repositioning_rules.json: Driver repositioning guidelines based on origin-destination affinity
- surge_rules.json: Multi-feature surge fare rules (temporal, spatial, demand)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class RulesService:
    """Manages mined association rules for dashboard intelligence."""

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.models_dir = base_dir / "saved_models"
        else:
            self.models_dir = Path(models_dir)

        self._repositioning_rules: List[Dict[str, Any]] = []
        self._surge_rules: List[Dict[str, Any]] = []
        self._load_rules()

    def _load_rules(self) -> None:
        """Loads rule JSON files from saved_models directory."""
        repo_paths = [
            self.models_dir / "repositioning_rules.json",
            self.models_dir.parent.parent / "02_ml_modeling_anoushey" / "saved_models" / "repositioning_rules.json",
            Path("saved_models/repositioning_rules.json"),
            Path("../saved_models/repositioning_rules.json")
        ]
        surge_paths = [
            self.models_dir / "surge_rules.json",
            self.models_dir.parent.parent / "02_ml_modeling_anoushey" / "saved_models" / "surge_rules.json",
            Path("saved_models/surge_rules.json"),
            Path("../saved_models/surge_rules.json")
        ]

        for p in repo_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self._repositioning_rules = json.load(f)
                    break
                except Exception:
                    continue

        for p in surge_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self._surge_rules = json.load(f)
                    break
                except Exception:
                    continue

    def has_repositioning_rules(self) -> bool:
        return len(self._repositioning_rules) > 0

    def has_surge_rules(self) -> bool:
        return len(self._surge_rules) > 0

    def get_repositioning_rules(
        self,
        borough: Optional[str] = None,
        min_lift: float = 1.0,
        limit: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top repositioning rules filtered by origin borough and minimum lift.
        """
        if not self._repositioning_rules:
            return []

        rules = self._repositioning_rules
        if borough and borough != "All Boroughs":
            b_clean = str(borough).strip().lower()
            rules = [r for r in rules if b_clean in str(r.get("antecedent", "")).lower()]

        if min_lift > 1.0:
            rules = [r for r in rules if float(r.get("lift", 0)) >= min_lift]

        sorted_rules = sorted(rules, key=lambda x: float(x.get("lift", 0)), reverse=True)
        return sorted_rules[:limit]

    def get_surge_rules(
        self,
        min_confidence: float = 0.5,
        limit: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top surge pricing rules sorted by confidence and lift.
        """
        if not self._surge_rules:
            return []

        rules = [r for r in self._surge_rules if float(r.get("confidence", 0)) >= min_confidence]
        sorted_rules = sorted(
            rules,
            key=lambda x: (float(x.get("confidence", 0)), float(x.get("lift", 0))),
            reverse=True
        )
        return sorted_rules[:limit]


# Global singleton instance
rules_service = RulesService()
