"""
DRIFT — Automated Unit and Integration Tests for Phase 1 Rider UI
"""

import sys
import unittest
from pathlib import Path

# Add 03_product_ui_monis to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from pages.ride import (
    create_nyc_map,
    ride_page,
    handle_trip_actions,
    update_map,
    handle_fare_intelligence,
    LOCATION_COORDINATES
)
from services.prediction_service import fare_service, PredictionResult
from app import app, display_page


class TestDriftPhase1(unittest.TestCase):

    def test_01_map_generation(self):
        """Test create_nyc_map handles all input permutations."""
        # Case 1: Empty
        fig_empty = create_nyc_map(None, None)
        self.assertIsNotNone(fig_empty)
        self.assertEqual(len(fig_empty.data), 1)

        # Case 2: Only pickup
        fig_pu = create_nyc_map("manhattan", None)
        self.assertEqual(len(fig_pu.data), 1)
        self.assertEqual(fig_pu.data[0].name, "Pickup")

        # Case 3: Only dropoff
        fig_do = create_nyc_map(None, "brooklyn")
        self.assertEqual(len(fig_do.data), 1)
        self.assertEqual(fig_do.data[0].name, "Destination")

        # Case 4: Distinct locations
        fig_both = create_nyc_map("manhattan", "queens")
        self.assertEqual(len(fig_both.data), 2)
        names = [trace.name for trace in fig_both.data]
        self.assertIn("Pickup", names)
        self.assertIn("Destination", names)

        # Case 5: Same location
        fig_same = create_nyc_map("brooklyn", "brooklyn")
        self.assertEqual(len(fig_same.data), 2)
        self.assertEqual(fig_same.data[0].name, "Pickup Zone")
        self.assertEqual(fig_same.data[1].name, "Destination Zone")

    def test_02_ride_page_layout(self):
        """Test ride_page layout renders all essential component IDs."""
        layout = ride_page()
        self.assertIsNotNone(layout)

        # Walk children and gather component IDs
        found_ids = set()
        def collect_ids(component):
            if hasattr(component, "id") and component.id:
                found_ids.add(component.id)
            if hasattr(component, "children"):
                children = component.children
                if isinstance(children, list):
                    for child in children:
                        collect_ids(child)
                else:
                    collect_ids(children)

        collect_ids(layout)

        expected_ids = {
            "pickup-location",
            "dropoff-location",
            "swap-locations-btn",
            "reset-trip-btn",
            "trip-date",
            "trip-time",
            "trip-passengers",
            "estimate-fare-btn",
            "fare-result-container",
            "ride-map",
            "map-status-pill",
            "same-location-alert"
        }
        for expected_id in expected_ids:
            self.assertIn(expected_id, found_ids, f"Component ID {expected_id} missing from ride_page")

    def test_03_prediction_service_contract(self):
        """Test prediction service behaves honestly when model is missing."""
        self.assertFalse(fare_service.is_model_ready())

        # Test incomplete inputs
        res_incomplete = fare_service.predict_fare(None, "queens")
        self.assertFalse(res_incomplete.is_available)
        self.assertEqual(res_incomplete.status_code, "INCOMPLETE_SELECTION")

        # Test complete inputs without model
        res_complete = fare_service.predict_fare("manhattan", "brooklyn", "2026-11-22", "08:00", 2)
        self.assertFalse(res_complete.is_available)
        self.assertEqual(res_complete.status_code, "MODEL_AWAITING_INTEGRATION")
        self.assertIsNone(res_complete.predicted_fare)
        self.assertEqual(res_complete.features_used["pickup_location"], "manhattan")
        self.assertEqual(res_complete.features_used["dropoff_location"], "brooklyn")
        self.assertEqual(res_complete.features_used["passenger_count"], 2)

    def test_04_map_callback(self):
        """Test update_map callback returns figure, status pill, and alert."""
        # Unselected
        fig, pill, alert = update_map(None, None)
        self.assertIsNotNone(fig)
        self.assertIsNone(alert)

        # Same location
        fig_same, pill_same, alert_same = update_map("manhattan", "manhattan")
        self.assertIsNotNone(alert_same)
        self.assertIn("Intra-zone trip", alert_same.children[1].children)

        # Distinct locations
        fig_distinct, pill_distinct, alert_distinct = update_map("manhattan", "brooklyn")
        self.assertIsNone(alert_distinct)

    def test_05_fare_feedback_callback(self):
        """Test fare intelligence callback states."""
        # 1. No selections
        disabled, label, feedback = handle_fare_intelligence(None, None, "2026-11-22", "08:00", 1, 0)
        self.assertTrue(disabled)
        self.assertEqual(label, "Select Locations to Estimate")

        # 2. Both selected, model awaiting integration
        disabled, label, feedback = handle_fare_intelligence("manhattan", "brooklyn", "2026-11-22", "08:00", 1, 0)
        self.assertTrue(disabled)  # Disabled until model is loaded
        self.assertEqual(label, "Model Integration Pending")

    def test_06_master_router(self):
        """Test URL router renders corresponding views."""
        welcome_view = display_page("/")
        self.assertIsNotNone(welcome_view)

        ride_view = display_page("/ride")
        self.assertIsNotNone(ride_view)

        intel_view = display_page("/intelligence")
        self.assertIsNotNone(intel_view)


if __name__ == "__main__":
    unittest.main()
