"""
DRIFT — Automated Unit and Integration Tests for Milestone 4
Team Integration, Real ML Compatibility, Receipt Fraud Guard, and Zone Services
"""

import sys
import unittest
from pathlib import Path

# Add 03_product_ui_monis to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from services.zone_service import zone_service, ZoneService
from services.feature_engineering import (
    estimate_duration_minutes,
    compute_avg_speed,
    compute_log_trip_distance,
    is_airport_trip
)
from services.anomaly_service import anomaly_service, AnomalyService, AnomalyResult
from services.clustering_service import clustering_service, ClusteringService, ClusterResult
from services.rules_service import rules_service, RulesService
from services.prediction_service import fare_service
from pages.fraud_guard import fraud_guard_page, handle_presets, audit_receipt_callback
from pages.ride import create_nyc_map, resolve_coordinates
from app import app, display_page


class TestDriftMilestone4(unittest.TestCase):

    # ==========================================
    # 1. Zone Service Tests
    # ==========================================

    def test_01_zone_service_loading(self):
        """Verifies official TLC 263-zone lookup is loaded with centroids."""
        zones = zone_service.get_zones()
        self.assertGreaterEqual(len(zones), 260)

        # Check JFK Airport (132)
        jfk = zone_service.get_zone(132)
        self.assertIsNotNone(jfk)
        self.assertEqual(jfk["Borough"], "Queens")
        self.assertTrue(jfk["is_airport"])
        self.assertAlmostEqual(jfk["lat"], 40.64, delta=0.05)
        self.assertAlmostEqual(jfk["lon"], -73.78, delta=0.05)

        # Check LGA Airport (138)
        lga = zone_service.get_zone(138)
        self.assertIsNotNone(lga)
        self.assertTrue(lga["is_airport"])

        # Check EWR Airport (1)
        ewr = zone_service.get_zone(1)
        self.assertIsNotNone(ewr)
        self.assertTrue(ewr["is_airport"])

    def test_02_haversine_distance_calculation(self):
        """Tests Great-Circle distance between JFK and Midtown Center."""
        # JFK (132) to Midtown Center (161)
        dist = zone_service.calculate_distance(132, 161)
        # JFK to Midtown is ~12-14 statute miles
        self.assertGreater(dist, 10.0)
        self.assertLess(dist, 16.0)

        # Intra-zone trip distance should return realistic minimal distance
        intra_dist = zone_service.calculate_distance(161, 161)
        self.assertAlmostEqual(intra_dist, 0.85, delta=0.1)

    def test_03_zone_dropdown_options(self):
        """Tests formatted dropdown options for Dash UI."""
        options = zone_service.get_dropdown_options()
        self.assertGreaterEqual(len(options), 260)
        first = options[0]
        self.assertIn("label", first)
        self.assertIn("value", first)
        self.assertIsInstance(first["value"], int)

    # ==========================================
    # 2. Feature Engineering Tests
    # ==========================================

    def test_04_feature_engineering_pipeline(self):
        """Verifies feature engineering math against teammate Stage 4 specs."""
        # Duration estimation
        dur_rush = estimate_duration_minutes(10.0, is_rush_hour=True)
        dur_offpeak = estimate_duration_minutes(10.0, is_rush_hour=False)
        self.assertGreater(dur_rush, dur_offpeak)

        # Average speed
        speed = compute_avg_speed(10.0, 30.0)  # 10 mi in 30 min = 20 mph
        self.assertAlmostEqual(speed, 20.0, places=1)

        # Log trip distance: np.log1p
        log_d = compute_log_trip_distance(5.0)
        import math
        self.assertAlmostEqual(log_d, math.log1p(5.0), places=3)

        # Airport trip detection
        self.assertTrue(is_airport_trip(132, 161))  # JFK -> Midtown
        self.assertTrue(is_airport_trip(161, 138))  # Midtown -> LGA
        self.assertTrue(is_airport_trip(1, 161))    # EWR -> Midtown
        self.assertFalse(is_airport_trip(161, 230)) # Midtown -> Times Square

    # ==========================================
    # 3. Anomaly Detection & Fraud Guard Tests
    # ==========================================

    def test_05_isolation_forest_inference(self):
        """Tests live inference on teammate's isoforest_bundle.pkl."""
        self.assertTrue(anomaly_service.is_ready(), "Isolation Forest model bundle should be ready")

        # Test normal trip: 2.5 mi, $14.50, 12 min
        res_norm = anomaly_service.audit_receipt(2.5, 14.50, 12.0)
        self.assertTrue(res_norm.is_available)
        self.assertEqual(res_norm.status_code, "SUCCESS")
        self.assertFalse(res_norm.is_anomaly, "Normal trip should not be flagged as anomaly")
        self.assertEqual(res_norm.severity, "normal")
        self.assertLess(res_norm.anomaly_score, res_norm.threshold)

        # Test severe anomaly: 0.5 mi, $95.00, 4 min
        res_anom = anomaly_service.audit_receipt(0.5, 95.00, 4.0)
        self.assertTrue(res_anom.is_available)
        self.assertEqual(res_anom.status_code, "SUCCESS")
        self.assertTrue(res_anom.is_anomaly, "Exorbitant fare short trip must be flagged")
        self.assertGreater(res_anom.anomaly_score, res_anom.threshold)
        self.assertGreater(len(res_anom.diagnostic_reasons), 0)

    def test_06_anomaly_service_error_handling(self):
        """Tests AnomalyService gracefully handles bad or missing inputs."""
        res_none = anomaly_service.audit_receipt(None, 14.5, 12.0)
        self.assertFalse(res_none.is_available)
        self.assertEqual(res_none.status_code, "MISSING_INPUTS")

        res_zero = anomaly_service.audit_receipt(0.0, 14.5, 12.0)
        self.assertFalse(res_zero.is_available)
        self.assertEqual(res_zero.status_code, "OUT_OF_RANGE")

    # ==========================================
    # 4. K-Means Clustering Tests
    # ==========================================

    def test_07_kmeans_clustering_inference(self):
        """Tests live classification on teammate's kmeans_bundle.pkl."""
        self.assertTrue(clustering_service.is_ready(), "KMeans model bundle should be ready")

        # City Hop trip: 1.2 mi, 8 min, 9 mph
        res_hop = clustering_service.classify_trip(1.2, 8.0, 9.0)
        self.assertTrue(res_hop.is_available)
        self.assertEqual(res_hop.cluster_id, 2)
        self.assertEqual(res_hop.cluster_name, "City Hops")
        self.assertEqual(res_hop.badge_color, "#06b6d4")

        # Airport trip: 18 mi, 45 min, 24 mph
        res_air = clustering_service.classify_trip(18.0, 45.0, 24.0)
        self.assertTrue(res_air.is_available)
        self.assertEqual(res_air.cluster_id, 1)
        self.assertEqual(res_air.cluster_name, "Airport / Long-Haul")
        self.assertEqual(res_air.badge_color, "#f59e0b")

    # ==========================================
    # 5. Association Rules Tests
    # ==========================================

    def test_08_rules_service_loading(self):
        """Verifies mined FP-Growth association rules are loaded."""
        self.assertTrue(rules_service.has_repositioning_rules())
        self.assertTrue(rules_service.has_surge_rules())

        repo_rules = rules_service.get_repositioning_rules(limit=5)
        self.assertGreater(len(repo_rules), 0)
        first_repo = repo_rules[0]
        self.assertIn("antecedent", first_repo)
        self.assertIn("consequent", first_repo)
        self.assertIn("lift", first_repo)

        surge_rules = rules_service.get_surge_rules(limit=5)
        self.assertGreater(len(surge_rules), 0)
        first_surge = surge_rules[0]
        self.assertIn("antecedent", first_surge)
        self.assertIn("confidence", first_surge)

    # ==========================================
    # 6. Fraud Guard UI & App Routing Tests
    # ==========================================

    def test_09_fraud_guard_page_layout(self):
        """Verifies Receipt Fraud Guard page layout components."""
        layout = fraud_guard_page()
        self.assertIsNotNone(layout)

        # Check preset callback
        d, f, dur = handle_presets(1, None, None, None, None)
        self.assertEqual(d, 2.5)
        self.assertEqual(f, 14.5)
        self.assertEqual(dur, 12.0)

        # Check audit callback for normal trip
        card_norm = audit_receipt_callback(1, 0, 0, 0, 0, 2.5, 14.50, 12.0)
        self.assertIsNotNone(card_norm)

        # Check audit callback for anomalous trip
        card_anom = audit_receipt_callback(1, 0, 0, 0, 0, 0.5, 95.00, 4.0)
        self.assertIsNotNone(card_anom)

    def test_10_app_navigation_routes(self):
        """Tests that all routes including /fraud-guard render HTTP-ready layouts."""
        portal = display_page("/")
        self.assertIsNotNone(portal)

        ride = display_page("/ride")
        self.assertIsNotNone(ride)

        fraud_guard = display_page("/fraud-guard")
        self.assertIsNotNone(fraud_guard)

        intel = display_page("/intelligence")
        self.assertIsNotNone(intel)

    def test_11_coordinate_resolution(self):
        """Tests coordinate resolution handles both zone IDs and legacy strings."""
        # Int Zone ID
        z132 = resolve_coordinates(132)
        self.assertIsNotNone(z132)
        self.assertEqual(z132["name"], "JFK Airport")
        self.assertTrue(z132["is_airport"])

        # String Zone ID
        z138_str = resolve_coordinates("138")
        self.assertIsNotNone(z138_str)
        self.assertEqual(z138_str["name"], "LaGuardia Airport")

        # Legacy borough string
        man = resolve_coordinates("manhattan")
        self.assertIsNotNone(man)
        self.assertEqual(man["name"], "Manhattan")


if __name__ == "__main__":
    unittest.main()
