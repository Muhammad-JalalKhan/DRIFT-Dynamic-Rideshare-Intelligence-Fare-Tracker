"""
DRIFT — Phase 2 Automated Unit and Integration Test Suite
Tests for DRIFT Intelligence Dashboard, Analytics Service, Visualizations, and Demo Mode
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure 03_product_ui_monis is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from services.analytics_service import AnalyticsService, DataStatus, KPIResult, analytics_service
from utils.analytics_helpers import (
    build_empty_chart,
    build_kpi_card,
    build_trips_over_time_chart,
    build_hourly_demand_chart,
    build_day_of_week_chart,
    build_fare_distribution_chart,
    build_fare_vs_distance_chart,
    build_fare_by_hour_chart,
    build_borough_volume_chart,
    build_top_corridors_chart,
    build_data_mining_status_card
)
from pages.intelligence import intelligence_page, update_intelligence_dashboard, reset_intelligence_filters
from app import app, display_page


class TestDriftIntelligencePhase2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a deterministic synthetic test fixture for test verification only."""
        cls.fixture_csv_path = BASE_DIR / "tests" / "fixtures" / "synthetic_trips.csv"
        cls.fixture_df = pd.DataFrame({
            "tpep_pickup_datetime": pd.to_datetime([
                "2024-01-01 08:30:00",
                "2024-01-01 12:15:00",
                "2024-01-01 17:45:00",
                "2024-01-02 21:10:00",
                "2024-01-03 08:15:00",
            ]),
            "fare_amount": [12.50, 24.00, 18.75, 45.00, 10.00],
            "trip_distance": [2.1, 5.4, 3.2, 12.0, 1.5],
            "duration_min": [10.5, 22.0, 15.0, 35.0, 8.0],
            "passenger_count": [1, 2, 1, 4, 1],
            "PU_borough": ["Manhattan", "Brooklyn", "Manhattan", "Queens", "Manhattan"],
            "DO_borough": ["Brooklyn", "Queens", "Manhattan", "Manhattan", "Bronx"],
            "hour_of_day": [8, 12, 17, 21, 8],
            "day_of_week": [0, 0, 0, 1, 2],
            "cluster": [0, 1, 0, 2, 0],
            "is_anomaly": [0, 0, 0, 1, 0]
        })

    def test_01_missing_dataset_behavior(self):
        """Test analytics service returns honest AWAITING_DATA status when paths are empty."""
        empty_service = AnalyticsService(search_paths=[Path("/non_existent_path_xyz")])
        status = empty_service.get_data_status(data_mode="real")
        self.assertFalse(status.is_loaded)
        self.assertEqual(status.status, "AWAITING_DATA")
        self.assertEqual(status.total_rows, 0)
        self.assertIn("No cleaned TLC telemetry dataset currently detected", status.message)

    def test_02_kpi_calculation_with_fixture(self):
        """Test exact mathematical correctness of KPI calculations."""
        service = AnalyticsService(search_paths=[])
        kpis = service.calculate_kpis(self.fixture_df)

        self.assertEqual(kpis.total_trips, "5")
        # Mean fare: (12.50 + 24.00 + 18.75 + 45.00 + 10.00) / 5 = 110.25 / 5 = 22.05
        self.assertEqual(kpis.avg_fare, "$22.05")
        # Mean distance: (2.1 + 5.4 + 3.2 + 12.0 + 1.5) / 5 = 24.2 / 5 = 4.84
        self.assertEqual(kpis.avg_distance, "4.84")
        # Mean duration: (10.5 + 22.0 + 15.0 + 35.0 + 8.0) / 5 = 90.5 / 5 = 18.1
        self.assertEqual(kpis.secondary_metric, "18.1")
        self.assertEqual(kpis.status, "active")

    def test_03_kpi_calculation_empty(self):
        """Test KPI calculation handles empty dataframe without division by zero."""
        service = AnalyticsService(search_paths=[])
        kpis = service.calculate_kpis(pd.DataFrame())
        self.assertEqual(kpis.total_trips, "0")
        self.assertEqual(kpis.avg_fare, "—")
        self.assertEqual(kpis.avg_distance, "—")
        self.assertEqual(kpis.status, "pending")

    def test_04_filtering_logic(self):
        """Test filtering by borough, time of day, passenger count, and dates."""
        service = AnalyticsService(search_paths=[])
        service.df = self.fixture_df.copy()

        # 1. Filter by pickup borough
        manhattan_only = service.filter_data(pu_boroughs=["Manhattan"], data_mode="real")
        self.assertEqual(len(manhattan_only), 3)

        # 2. Filter by time window (morning peak = 7-9)
        morning_peak = service.filter_data(time_of_day="morning_peak", data_mode="real")
        self.assertEqual(len(morning_peak), 2)  # Hours 8 and 8

        # 3. Filter by party size (solo = 1)
        solo_riders = service.filter_data(passenger_count=1, data_mode="real")
        self.assertEqual(len(solo_riders), 3)

        # 4. Multi-criteria filter
        multi = service.filter_data(pu_boroughs=["Manhattan"], time_of_day="morning_peak", passenger_count=1, data_mode="real")
        self.assertEqual(len(multi), 2)

    def test_05_reset_filters_callback(self):
        """Test reset filter callback returns clean neutral states."""
        pu, do, time_w, party = reset_intelligence_filters(1)
        self.assertIsNone(pu)
        self.assertIsNone(do)
        self.assertEqual(time_w, "all")
        self.assertEqual(party, "all")

    def test_06_chart_generation_with_data(self):
        """Test that all chart builders generate valid Plotly Figures with data."""
        df = self.fixture_df

        fig_trends = build_trips_over_time_chart(df)
        self.assertIsNotNone(fig_trends)
        self.assertGreater(len(fig_trends.data), 0)

        fig_hourly = build_hourly_demand_chart(df)
        self.assertIsNotNone(fig_hourly)
        self.assertGreater(len(fig_hourly.data), 0)

        fig_dow = build_day_of_week_chart(df)
        self.assertIsNotNone(fig_dow)
        self.assertGreater(len(fig_dow.data), 0)

        fig_fare_dist = build_fare_distribution_chart(df)
        self.assertIsNotNone(fig_fare_dist)

        fig_fare_dist_scatter = build_fare_vs_distance_chart(df)
        self.assertIsNotNone(fig_fare_dist_scatter)

        fig_fare_hour = build_fare_by_hour_chart(df)
        self.assertIsNotNone(fig_fare_hour)

        fig_pu = build_borough_volume_chart(df, "PU_borough", "Pickups")
        self.assertIsNotNone(fig_pu)

        fig_corridors = build_top_corridors_chart(df)
        self.assertIsNotNone(fig_corridors)

    def test_07_chart_generation_empty(self):
        """Test that all chart builders handle empty DataFrames safely with empty states."""
        empty_df = pd.DataFrame()

        fig_empty = build_empty_chart("Test", "Message")
        self.assertIsNotNone(fig_empty)

        fig_trends_empty = build_trips_over_time_chart(empty_df)
        self.assertIsNotNone(fig_trends_empty)

        fig_fare_empty = build_fare_distribution_chart(empty_df)
        self.assertIsNotNone(fig_fare_empty)

    def test_08_mining_modules_status(self):
        """Test data mining module status reporting in Real mode."""
        service = AnalyticsService(search_paths=[])
        c_status = service.get_clustering_status(data_mode="real")
        self.assertIn("available", c_status)
        self.assertIn("message", c_status)

        a_status = service.get_anomaly_status(data_mode="real")
        self.assertIn("available", a_status)

        m_status = service.get_model_eval_status(data_mode="real")
        self.assertIn("available", m_status)

        p_status = service.get_pattern_mining_status(data_mode="real")
        self.assertIn("available", p_status)

    def test_09_intelligence_page_layout(self):
        """Test that intelligence_page layout instantiates all essential component IDs."""
        layout = intelligence_page()
        self.assertIsNotNone(layout)

        found_ids = set()
        def collect_ids(comp):
            if hasattr(comp, "id") and comp.id:
                found_ids.add(comp.id)
            if hasattr(comp, "children"):
                children = comp.children
                if isinstance(children, list):
                    for child in children:
                        collect_ids(child)
                else:
                    collect_ids(children)

        collect_ids(layout)

        expected_ids = {
            "intelligence-kpi-container",
            "intel-data-source-toggle",
            "intel-demo-mode-banner",
            "intel-header-badge",
            "intel-filter-pu",
            "intel-filter-do",
            "intel-filter-time",
            "intel-filter-passengers",
            "intel-reset-filters-btn",
            "intel-filter-summary",
            "chart-trips-over-time",
            "chart-hourly-demand",
            "chart-day-of-week",
            "chart-fare-dist",
            "chart-fare-vs-distance",
            "chart-fare-by-hour",
            "chart-pickup-borough",
            "chart-dropoff-borough",
            "chart-top-corridors",
            "intel-mining-modules-container"
        }
        for exp_id in expected_ids:
            self.assertIn(exp_id, found_ids, f"Expected ID '{exp_id}' missing from intelligence_page layout")

    def test_10_app_routing(self):
        """Test master router renders /intelligence cleanly."""
        intel_view = display_page("/intelligence")
        self.assertIsNotNone(intel_view)

        ride_view = display_page("/ride")
        self.assertIsNotNone(ride_view)

        home_view = display_page("/")
        self.assertIsNotNone(home_view)

    def test_11_load_synthetic_csv_fixture(self):
        """Test loading dedicated synthetic CSV test fixture from tests/fixtures/."""
        self.assertTrue(self.fixture_csv_path.exists(), f"Synthetic fixture missing: {self.fixture_csv_path}")

        test_service = AnalyticsService(search_paths=[])
        loaded = test_service.load_dataset(str(self.fixture_csv_path))
        self.assertTrue(loaded)

        status = test_service.get_data_status(data_mode="real")
        self.assertTrue(status.is_loaded)
        self.assertEqual(status.status, "READY")
        self.assertEqual(status.total_rows, 12)

        # Test that clustering columns are detected
        c_status = test_service.get_clustering_status(data_mode="real")
        self.assertTrue(c_status["available"])
        self.assertEqual(c_status["status"], "Ready")
        self.assertIn(0, c_status["cluster_counts"])

        # Test that anomaly detection columns are detected
        a_status = test_service.get_anomaly_status(data_mode="real")
        self.assertTrue(a_status["available"])
        self.assertEqual(a_status["status"], "Ready")
        self.assertEqual(a_status["flagged_count"], 1)

    def test_12_normal_runtime_isolation(self):
        """Verify normal application runtime does NOT load synthetic test data in Real mode."""
        status = analytics_service.get_data_status(data_mode="real")
        self.assertFalse(status.is_loaded, "Production Real Data mode must NOT automatically load test data.")
        self.assertEqual(status.status, "AWAITING_DATA")
        self.assertEqual(status.total_rows, 0)

    def test_13_zero_match_filter_handling(self):
        """Test that filters yielding zero rows handle KPIs and charts without crashing."""
        test_service = AnalyticsService(search_paths=[])
        test_service.load_dataset(str(self.fixture_csv_path))

        zero_df = test_service.filter_data(pu_boroughs=["Staten Island"], time_of_day="morning_peak", data_mode="real")
        self.assertEqual(len(zero_df), 0)

        kpis = test_service.calculate_kpis(zero_df)
        self.assertEqual(kpis.total_trips, "0")
        self.assertEqual(kpis.avg_fare, "—")
        self.assertEqual(kpis.status, "pending")

        fig_trend = build_trips_over_time_chart(zero_df)
        self.assertIsNotNone(fig_trend)

        fig_fare = build_fare_distribution_chart(zero_df)
        self.assertIsNotNone(fig_fare)

    def test_14_dataset_load_error_handling(self):
        """Test error handling when loading non-existent or corrupted files."""
        err_service = AnalyticsService(search_paths=[])
        loaded = err_service.load_dataset("/invalid/path/that/does_not_exist.parquet")
        self.assertFalse(loaded)

        status = err_service.get_data_status(data_mode="real")
        self.assertFalse(status.is_loaded)

    def test_15_demo_mode_dataset_and_kpis(self):
        """Test Demo Mode produces 500 deterministic records with valid KPIs."""
        status = analytics_service.get_data_status(data_mode="demo")
        self.assertTrue(status.is_loaded)
        self.assertEqual(status.status, "DEMO_MODE")
        self.assertEqual(status.total_rows, 500)
        self.assertIn("DEMO DATA", status.message)

        demo_df = analytics_service.filter_data(data_mode="demo")
        self.assertEqual(len(demo_df), 500)

        kpis = analytics_service.calculate_kpis(demo_df)
        self.assertEqual(kpis.total_trips, "500")
        self.assertTrue(kpis.avg_fare.startswith("$"))
        self.assertTrue(float(kpis.avg_distance) > 0)
        self.assertEqual(kpis.status, "active")

    def test_16_demo_mode_filters_and_isolation(self):
        """Test Demo Mode filtering works and does not contaminate Real mode."""
        # 1. Filter demo data by Manhattan pickups
        manhattan_demo = analytics_service.filter_data(pu_boroughs=["Manhattan"], data_mode="demo")
        self.assertGreater(len(manhattan_demo), 0)
        self.assertLess(len(manhattan_demo), 500)

        # 2. Filter demo data by time
        morning_demo = analytics_service.filter_data(time_of_day="morning_peak", data_mode="demo")
        self.assertGreater(len(morning_demo), 0)
        self.assertLess(len(morning_demo), 500)

        # 3. Verify Real Data mode is unaffected (remains empty / awaiting data)
        real_status = analytics_service.get_data_status(data_mode="real")
        self.assertFalse(real_status.is_loaded)
        self.assertEqual(real_status.status, "AWAITING_DATA")

    def test_17_dashboard_callback_real_vs_demo(self):
        """Test dashboard callback output switching between Real Data and Demo Data."""
        # 1. Real Data call (default)
        out_real = update_intelligence_dashboard("real", None, None, "all", "all", "volume")
        # demo_banner is index 12 in output tuple
        banner_real = out_real[12]
        self.assertIsNone(banner_real, "Real mode must not display demo banner")

        # 2. Demo Data call
        out_demo = update_intelligence_dashboard("demo", None, None, "all", "all", "volume")
        banner_demo = out_demo[12]
        self.assertIsNotNone(banner_demo, "Demo mode must display persistent demo banner")

        # Verify charts have data in demo mode
        fig_trips_demo = out_demo[1]
        self.assertGreater(len(fig_trips_demo.data), 0, "Demo mode trips chart must contain plotted traces")

    def test_18_daily_chart_date_formatting(self):
        """Test Daily Trip Demand Volume chart aggregates by calendar date and formats cleanly."""
        df = self.fixture_df  # Has timestamps on Jan 01, Jan 02, Jan 03 with various hours/minutes
        fig = build_trips_over_time_chart(df)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

        # X-axis must be categorical to prevent continuous time interpolation
        self.assertEqual(fig.layout.xaxis.type, "category")

        # X values must be formatted date strings (e.g. "Jan 01", "Jan 02", "Jan 03")
        x_vals = list(fig.data[0].x)
        self.assertEqual(len(x_vals), 3)  # 3 distinct calendar days
        self.assertIn("Jan 01", x_vals[0])
        # Ensure no time strings or microsecond substrings exist in x labels
        for val in x_vals:
            self.assertNotIn(":", str(val), "Date labels must not contain time colons")
            self.assertNotIn("000", str(val), "Date labels must not contain microseconds")

    def test_19_small_sample_chart_rendering(self):
        """Test chart behavior with single-record and 2-record subsets."""
        # 1-record slice
        single_row = self.fixture_df.iloc[:1].copy()

        # 1. Trips over time should use Bar trace with width and integer ticks
        fig_single_trend = build_trips_over_time_chart(single_row)
        self.assertEqual(fig_single_trend.data[0].type, "bar")
        self.assertEqual(fig_single_trend.layout.yaxis.dtick, 1)
        self.assertEqual(fig_single_trend.layout.yaxis.tickformat, "d")

        # 2-record slice (both on Jan 01)
        two_rows = self.fixture_df.iloc[:2].copy()
        fig_two_trend = build_trips_over_time_chart(two_rows)
        self.assertEqual(fig_two_trend.data[0].type, "bar")
        self.assertEqual(fig_two_trend.layout.yaxis.dtick, 1)

        # Hourly chart with single record has integer y-ticks
        fig_hourly = build_hourly_demand_chart(single_row)
        self.assertEqual(fig_hourly.layout.yaxis.dtick, 1)
        self.assertEqual(fig_hourly.layout.yaxis.tickformat, "d")

        # Day of week chart with single record has integer y-ticks
        fig_dow = build_day_of_week_chart(single_row)
        self.assertEqual(fig_dow.layout.yaxis.dtick, 1)

        # Borough volume chart with single record has integer ticks and width constraint
        fig_borough = build_borough_volume_chart(single_row, "PU_borough")
        self.assertEqual(fig_borough.layout.yaxis.dtick, 1)
        self.assertIsNotNone(fig_borough.data[0].width)

        # Top corridors chart with single record has generous left margin (>= 140) and integer ticks
        fig_corridors = build_top_corridors_chart(single_row)
        self.assertEqual(fig_corridors.layout.xaxis.dtick, 1)
        self.assertGreaterEqual(fig_corridors.layout.margin.l, 140)

        # Fare vs distance scatter with single record has prominent marker (size >= 8)
        fig_scatter = build_fare_vs_distance_chart(single_row)
        self.assertGreaterEqual(fig_scatter.data[0].marker.size, 8)

        # Fare by hour with single record renders as Bar
        fig_fare_hour = build_fare_by_hour_chart(single_row)
        self.assertEqual(fig_fare_hour.data[0].type, "bar")

    def test_20_demo_mode_reset_restores_complete_dataset(self):
        """Test that Reset Filters restores all 500 demo trips and refreshes KPIs and charts."""
        # 1. Apply restrictive filter in demo mode
        filtered_df = analytics_service.filter_data(pu_boroughs=["Bronx"], time_of_day="morning_peak", data_mode="demo")
        filtered_count = len(filtered_df)
        self.assertLess(filtered_count, 500)

        # 2. Reset filters
        pu, do, time_w, party = reset_intelligence_filters(1)
        restored_df = analytics_service.filter_data(
            pu_boroughs=pu,
            do_boroughs=do,
            time_of_day=None if time_w == "all" else time_w,
            passenger_count=None if party == "all" else party,
            data_mode="demo"
        )
        self.assertEqual(len(restored_df), 500, "Reset must restore full 500 demo dataset")

        # 3. Full KPI calculation matches complete demo dataset
        kpis = analytics_service.calculate_kpis(restored_df)
        self.assertEqual(kpis.total_trips, "500")
        self.assertEqual(kpis.status, "active")

    def test_21_zero_based_bar_chart_axes(self):
        """Verify that all count-based spatial bar charts visually and explicitly begin at zero."""
        df = self.fixture_df
        # 1. Borough volume chart (Pickups)
        fig_pu = build_borough_volume_chart(df, "PU_borough")
        self.assertEqual(fig_pu.layout.yaxis.rangemode, "tozero")
        self.assertEqual(fig_pu.layout.yaxis.tick0, 0)

        # 2. Borough volume chart (Destinations)
        fig_do = build_borough_volume_chart(df, "DO_borough")
        self.assertEqual(fig_do.layout.yaxis.rangemode, "tozero")
        self.assertEqual(fig_do.layout.yaxis.tick0, 0)

        # 3. Top mobility corridors (Horizontal bar chart: numeric axis is X)
        fig_corr = build_top_corridors_chart(df)
        self.assertEqual(fig_corr.layout.xaxis.rangemode, "tozero")
        self.assertEqual(fig_corr.layout.xaxis.tick0, 0)

        # 4. Hourly demand & Day of week
        fig_hourly = build_hourly_demand_chart(df)
        self.assertEqual(fig_hourly.layout.yaxis.rangemode, "tozero")
        self.assertEqual(fig_hourly.layout.yaxis.tick0, 0)

        fig_dow = build_day_of_week_chart(df)
        self.assertEqual(fig_dow.layout.yaxis.rangemode, "tozero")
        self.assertEqual(fig_dow.layout.yaxis.tick0, 0)

    def test_22_scatter_plot_initial_range_and_reset(self):
        """Verify scatter plot default range covers the entire dataset and is not clamped to a subregion."""
        demo_df = analytics_service.filter_data(data_mode="demo")
        fig_scatter = build_fare_vs_distance_chart(demo_df)

        self.assertEqual(fig_scatter.layout.xaxis.rangemode, "tozero")
        self.assertTrue(fig_scatter.layout.xaxis.autorange)
        self.assertEqual(fig_scatter.layout.yaxis.rangemode, "tozero")
        self.assertTrue(fig_scatter.layout.yaxis.autorange)

        # Verify dataset coverage: min/max spans the full data range, not a 7.5-9.5 mi / $45-$60 sub-window
        x_pts = fig_scatter.data[0].x
        y_pts = fig_scatter.data[0].y
        self.assertLess(min(x_pts), 1.0, "Scatter x-axis data must include short trips (< 1.0 mi)")
        self.assertGreater(max(x_pts), 12.0, "Scatter x-axis data must include long trips (> 12.0 mi)")
        self.assertLess(min(y_pts), 10.0, "Scatter y-axis data must include low fares (< $10.0)")
        self.assertGreater(max(y_pts), 50.0, "Scatter y-axis data must include high fares (> $50.0)")

    def test_23_hourly_axis_bounded_0_to_23(self):
        """Verify 24-hour demand and fare charts are strictly bounded to 0-23 without exceeding range."""
        df = self.fixture_df
        fig_hourly = build_hourly_demand_chart(df)

        # Axis must be strictly bounded between -0.5 and 23.5 so 24 or -1 never appear
        self.assertEqual(list(fig_hourly.layout.xaxis.range), [-0.5, 23.5])
        self.assertEqual(list(fig_hourly.layout.xaxis.tickvals), list(range(0, 24, 2)))
        self.assertEqual(fig_hourly.layout.xaxis.ticktext[0], "00:00")
        self.assertEqual(fig_hourly.layout.xaxis.ticktext[-1], "22:00")

        # Fare by hour chart
        demo_df = analytics_service.filter_data(data_mode="demo")
        fig_fare_hour = build_fare_by_hour_chart(demo_df)
        self.assertEqual(list(fig_fare_hour.layout.xaxis.range), [-0.5, 23.5])
        self.assertEqual(list(fig_fare_hour.layout.xaxis.tickvals), list(range(0, 24, 2)))

    def test_24_empty_result_behavior_clean_empty_state(self):
        """Verify that zero-matching filter results produce clean, graceful empty states."""
        empty_df = pd.DataFrame(columns=["tpep_pickup_datetime", "fare_amount", "trip_distance", "PU_borough", "DO_borough", "hour_of_day"])

        # KPI calculation on empty dataframe
        kpis = analytics_service.calculate_kpis(empty_df)
        self.assertEqual(kpis.total_trips, "0")
        self.assertEqual(kpis.avg_fare, "—")
        self.assertEqual(kpis.status, "pending")

        # Chart builders must return empty state annotation with 'No Matching Records'
        fig_trips = build_trips_over_time_chart(empty_df)
        self.assertGreater(len(fig_trips.layout.annotations), 0)
        self.assertIn("No Matching Records", fig_trips.layout.annotations[0].text)

        fig_scatter = build_fare_vs_distance_chart(empty_df)
        self.assertGreater(len(fig_scatter.layout.annotations), 0)
        self.assertIn("No Matching Records", fig_scatter.layout.annotations[0].text)

        fig_corr = build_top_corridors_chart(empty_df)
        self.assertGreater(len(fig_corr.layout.annotations), 0)
        self.assertIn("No Matching Records", fig_corr.layout.annotations[0].text)

    def test_25_kdd_model_eval_schema_format(self):
        """Verify KDD model evaluation schema format is 'mae, rmse, r2'."""
        eval_real = analytics_service.get_model_eval_status(data_mode="real")
        self.assertEqual(eval_real["expected_schema"], "mae, rmse, r2")

        eval_demo = analytics_service.get_model_eval_status(data_mode="demo")
        self.assertEqual(eval_demo["expected_schema"], "mae, rmse, r2")


if __name__ == "__main__":
    unittest.main()
