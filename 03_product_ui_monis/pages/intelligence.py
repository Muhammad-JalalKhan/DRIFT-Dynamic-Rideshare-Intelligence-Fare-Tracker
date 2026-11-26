"""
DRIFT — Dynamic Rideshare Intelligence & Fare Tracker
Phase 2: DRIFT Intelligence — Interactive Data Mining & Mobility Analytics Dashboard
Supports Real Data (default) and Explicit Demo Data (500 deterministic synthetic trips).
"""

from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from services.analytics_service import analytics_service, DataStatus, KPIResult
from utils.analytics_helpers import (
    build_kpi_card,
    build_empty_chart,
    build_trips_over_time_chart,
    build_hourly_demand_chart,
    build_day_of_week_chart,
    build_fare_distribution_chart,
    build_fare_vs_distance_chart,
    build_fare_by_hour_chart,
    build_borough_volume_chart,
    build_top_corridors_chart,
    build_data_mining_status_card,
    THEME_CYAN,
    THEME_AMBER,
    THEME_PURPLE,
    THEME_GREEN
)

NYC_BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]


# ==============================================================================
# Helper Layout Builders
# ==============================================================================

def render_kpi_row(kpis: KPIResult) -> dbc.Row:
    """Renders the executive KPI overview cards in a responsive grid."""
    return dbc.Row(
        [
            dbc.Col(
                build_kpi_card(
                    title="Total Trips Analyzed",
                    value=kpis.total_trips,
                    unit="",
                    icon="🚕",
                    subtitle="Filter-responsive record count",
                    status=kpis.status
                ),
                xs=12, sm=6, lg=3,
                className="mb-3"
            ),
            dbc.Col(
                build_kpi_card(
                    title="Average Trip Fare",
                    value=kpis.avg_fare,
                    unit="USD",
                    icon="💵",
                    subtitle="Mean base metered fare",
                    status=kpis.status
                ),
                xs=12, sm=6, lg=3,
                className="mb-3"
            ),
            dbc.Col(
                build_kpi_card(
                    title="Average Distance",
                    value=kpis.avg_distance,
                    unit="mi",
                    icon="📍",
                    subtitle="Mean odometer distance",
                    status=kpis.status
                ),
                xs=12, sm=6, lg=3,
                className="mb-3"
            ),
            dbc.Col(
                build_kpi_card(
                    title=kpis.secondary_metric_label,
                    value=kpis.secondary_metric,
                    unit="min" if "Duration" in kpis.secondary_metric_label else "USD",
                    icon="⏱️",
                    subtitle="Empirical trip duration",
                    status=kpis.status
                ),
                xs=12, sm=6, lg=3,
                className="mb-3"
            ),
        ],
        className="g-3 mb-4"
    )


def render_mining_modules(data_mode: str = "real") -> dbc.Row:
    """Renders the 4 data mining module status and insight cards."""
    c_status = analytics_service.get_clustering_status(data_mode=data_mode)
    a_status = analytics_service.get_anomaly_status(data_mode=data_mode)
    m_status = analytics_service.get_model_eval_status(data_mode=data_mode)
    p_status = analytics_service.get_pattern_mining_status(data_mode=data_mode)

    return dbc.Row(
        [
            dbc.Col(
                build_data_mining_status_card(
                    title="K-Means Demand Clustering",
                    stage_name="Stage 5: Mining Track",
                    status_label=c_status["status"],
                    status_color=c_status["color"],
                    description=c_status["message"],
                    required_artifact=c_status.get("required_artifact", "demand_kmeans.joblib"),
                    expected_schema=c_status.get("expected_schema", "PULocationID, DOLocationID, trip_distance, hour_of_day")
                ),
                xs=12, md=6, lg=3,
                className="mb-3"
            ),
            dbc.Col(
                build_data_mining_status_card(
                    title="Isolation Forest Anomalies",
                    stage_name="Stage 5: Anomaly Track",
                    status_label=a_status["status"],
                    status_color=a_status["color"],
                    description=a_status["message"],
                    required_artifact=a_status.get("required_artifact", "isolation_forest.joblib"),
                    expected_schema=a_status.get("expected_schema", "fare_amount, trip_distance, duration_min, avg_speed_mph")
                ),
                xs=12, md=6, lg=3,
                className="mb-3"
            ),
            dbc.Col(
                build_data_mining_status_card(
                    title="Fare Regression Baselines",
                    stage_name="Stage 6: Model Evaluation",
                    status_label=m_status["status"],
                    status_color=m_status["color"],
                    description=m_status["message"],
                    required_artifact=m_status.get("required_artifact", "regression_eval.json"),
                    expected_schema=m_status.get("expected_schema", "mae, rmse, r2")
                ),
                xs=12, md=6, lg=3,
                className="mb-3"
            ),
            dbc.Col(
                build_data_mining_status_card(
                    title="FP-Growth Association Rules",
                    stage_name="Stage 5: Pattern Mining",
                    status_label=p_status["status"],
                    status_color=p_status["color"],
                    description=p_status["message"],
                    required_artifact=p_status.get("required_artifact", "association_rules.csv"),
                    expected_schema=p_status.get("expected_schema", "antecedents, consequents, support, confidence")
                ),
                xs=12, md=6, lg=3,
                className="mb-3"
            ),
        ],
        className="g-3"
    )


# ==============================================================================
# DRIFT Intelligence Master Page Layout
# ==============================================================================

def intelligence_page():
    # 1. Header Section
    header = dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Span("DRIFT", className="drift-badge-brand me-2"),
                            html.Span("MOBILITY INTELLIGENCE & PATTERN MINER", className="drift-eyebrow"),
                            html.Span(id="intel-header-badge", className="ms-2")
                        ],
                        className="d-flex align-items-center mb-1 flex-wrap"
                    ),
                    html.H2("DRIFT Intelligence", className="drift-page-title mb-1"),
                    html.P(
                        "Understand the city. Decode mobility. Discover the patterns behind every ride.",
                        className="drift-muted mb-0"
                    )
                ],
                xs=12, md=8,
                className="mb-3 mb-md-0"
            ),
            dbc.Col(
                [
                    dcc.Link(
                        dbc.Button(
                            [html.Span("🚕", className="me-2"), "Rider View"],
                            color="info",
                            size="sm",
                            className="drift-btn-outline me-2"
                        ),
                        href="/ride"
                    ),
                    dcc.Link(
                        dbc.Button(
                            [html.Span("←", className="me-2"), "Back to Portal"],
                            color="outline-light",
                            size="sm",
                            className="drift-btn-outline"
                        ),
                        href="/"
                    )
                ],
                xs=12, md=4,
                className="text-md-end align-self-center"
            )
        ],
        className="mb-3 pb-2 border-bottom border-secondary border-opacity-25"
    )

    # 2. Prominent Data Source Selector Bar (Task 1)
    source_selector_bar = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Span("DATA SOURCE:", className="drift-input-label me-3 d-inline-block"),
                            dbc.RadioItems(
                                id="intel-data-source-toggle",
                                options=[
                                    {"label": "📊 Real Data (NYC Telemetry Pipeline)", "value": "real"},
                                    {"label": "🧪 Demo Data (500 Deterministic Synthetic Trips)", "value": "demo"}
                                ],
                                value="real",
                                inline=True,
                                className="drift-source-toggle d-inline-block"
                            )
                        ],
                        xs=12, md=8,
                        className="d-flex align-items-center flex-wrap"
                    ),
                    dbc.Col(
                        [
                            html.Span(
                                "Switch to Demo Data to preview live charts without teammate files",
                                className="drift-muted small"
                            )
                        ],
                        xs=12, md=4,
                        className="text-md-end align-self-center mt-2 mt-md-0"
                    )
                ],
                className="align-items-center"
            )
        ],
        className="drift-source-bar p-3 mb-3 rounded"
    )

    # 3. Persistent Demo Banner Container (Task 1)
    demo_banner_container = html.Div(id="intel-demo-mode-banner")

    # 4. Global Filter Toolbar
    filter_bar = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Pickup Boroughs", className="drift-input-label"),
                                dcc.Dropdown(
                                    id="intel-filter-pu",
                                    options=[{"label": b, "value": b} for b in NYC_BOROUGHS],
                                    multi=True,
                                    placeholder="All Boroughs",
                                    searchable=True,
                                    clearable=True,
                                    className="drift-dropdown"
                                )
                            ],
                            xs=12, sm=6, lg=3,
                            className="mb-2"
                        ),
                        dbc.Col(
                            [
                                html.Label("Destination Boroughs", className="drift-input-label"),
                                dcc.Dropdown(
                                    id="intel-filter-do",
                                    options=[{"label": b, "value": b} for b in NYC_BOROUGHS],
                                    multi=True,
                                    placeholder="All Boroughs",
                                    searchable=True,
                                    clearable=True,
                                    className="drift-dropdown"
                                )
                            ],
                            xs=12, sm=6, lg=3,
                            className="mb-2"
                        ),
                        dbc.Col(
                            [
                                html.Label("Time of Day Window", className="drift-input-label"),
                                dcc.Dropdown(
                                    id="intel-filter-time",
                                    options=[
                                        {"label": "All Hours (24h)", "value": "all"},
                                        {"label": "Morning Peak (7 AM – 9 AM)", "value": "morning_peak"},
                                        {"label": "Midday Regular (10 AM – 3 PM)", "value": "midday"},
                                        {"label": "Evening Peak (4 PM – 7 PM)", "value": "evening_peak"},
                                        {"label": "Night & Late Hours (8 PM – 6 AM)", "value": "night"}
                                    ],
                                    value="all",
                                    clearable=False,
                                    className="drift-dropdown"
                                )
                            ],
                            xs=12, sm=6, lg=2,
                            className="mb-2"
                        ),
                        dbc.Col(
                            [
                                html.Label("Party Size", className="drift-input-label"),
                                dcc.Dropdown(
                                    id="intel-filter-passengers",
                                    options=[
                                        {"label": "All Sizes", "value": "all"},
                                        {"label": "Solo (1 Rider)", "value": "1"},
                                        {"label": "2 Riders", "value": "2"},
                                        {"label": "3–4 Riders", "value": "3"},
                                        {"label": "5+ Group", "value": "5"}
                                    ],
                                    value="all",
                                    clearable=False,
                                    className="drift-dropdown"
                                )
                            ],
                            xs=12, sm=6, lg=2,
                            className="mb-2"
                        ),
                        dbc.Col(
                            [
                                html.Label("Actions", className="drift-input-label d-none d-lg-block"),
                                html.Div(
                                    dbc.Button(
                                        "↺ Reset Filters",
                                        id="intel-reset-filters-btn",
                                        color="dark",
                                        size="sm",
                                        className="drift-btn-utility w-100",
                                        style={"height": "42px", "display": "flex", "alignItems": "center", "justifyContent": "center"}
                                    )
                                )
                            ],
                            xs=12, lg=2,
                            className="mb-2"
                        )
                    ],
                    className="align-items-end"
                ),
                html.Div(
                    id="intel-filter-summary",
                    className="small drift-muted mt-2 pt-2 border-top border-secondary border-opacity-10 d-flex justify-content-between align-items-center"
                )
            ],
            className="p-3"
        ),
        className="drift-card mb-4"
    )

    # 5. Chart Sections
    section_trends = html.Div(
        [
            html.Div(
                [
                    html.Span("01 / TEMPORAL ANALYSIS", className="drift-eyebrow"),
                    html.H4("Urban Mobility Demand Patterns", className="text-white mb-3")
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="chart-trips-over-time", config={"responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]})),
                            className="drift-card h-100"
                        ),
                        xs=12, lg=6,
                        className="mb-3"
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="chart-hourly-demand", config={"responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]})),
                            className="drift-card h-100"
                        ),
                        xs=12, lg=3,
                        className="mb-3"
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="chart-day-of-week", config={"responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]})),
                            className="drift-card h-100"
                        ),
                        xs=12, lg=3,
                        className="mb-3"
                    ),
                ],
                className="g-3 mb-4"
            )
        ]
    )

    section_fares = html.Div(
        [
            html.Div(
                [
                    html.Span("02 / FARE ECONOMICS", className="drift-eyebrow"),
                    html.H4("Fare Behavior & Dynamics", className="text-white mb-3")
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="chart-fare-dist", config={"responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]})),
                            className="drift-card h-100"
                        ),
                        xs=12, lg=4,
                        className="mb-3"
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="chart-fare-vs-distance", config={"responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]})),
                            className="drift-card h-100"
                        ),
                        xs=12, lg=4,
                        className="mb-3"
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="chart-fare-by-hour", config={"responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]})),
                            className="drift-card h-100"
                        ),
                        xs=12, lg=4,
                        className="mb-3"
                    ),
                ],
                className="g-3 mb-4"
            )
        ]
    )

    section_geo = html.Div(
        [
            html.Div(
                [
                    html.Span("03 / SPATIAL DYNAMICS", className="drift-eyebrow"),
                    html.H4("Geographic Mobility & Corridors", className="text-white mb-3")
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="chart-pickup-borough", config={"responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]})),
                            className="drift-card h-100"
                        ),
                        xs=12, lg=4,
                        className="mb-3"
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="chart-dropoff-borough", config={"responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]})),
                            className="drift-card h-100"
                        ),
                        xs=12, lg=4,
                        className="mb-3"
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="chart-top-corridors", config={"responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]})),
                            className="drift-card h-100"
                        ),
                        xs=12, lg=4,
                        className="mb-3"
                    ),
                ],
                className="g-3 mb-4"
            )
        ]
    )

    section_mining = html.Div(
        [
            html.Div(
                [
                    html.Span("04 / KDD MINING PIPELINE", className="drift-eyebrow"),
                    html.H4("Data Mining Insights & Algorithmic Modules", className="text-white mb-3")
                ]
            ),
            html.Div(id="intel-mining-modules-container")
        ],
        className="mb-4"
    )

    # Metric Selector hidden radio for callback compatibility
    metric_selector = dcc.RadioItems(
        id="intel-metric-selector",
        options=[{"label": "Volume", "value": "volume"}, {"label": "Fare", "value": "fare"}],
        value="volume",
        style={"display": "none"}
    )

    return dbc.Container(
        [
            header,
            source_selector_bar,
            demo_banner_container,
            html.Div(id="intelligence-kpi-container"),
            filter_bar,
            metric_selector,
            section_trends,
            section_fares,
            section_geo,
            section_mining
        ],
        fluid=True,
        className="drift-main"
    )


# ==============================================================================
# Dynamic Callbacks
# ==============================================================================

# Callback 1: Reset Filters (Preserves Active Data Source)
@callback(
    Output("intel-filter-pu", "value"),
    Output("intel-filter-do", "value"),
    Output("intel-filter-time", "value"),
    Output("intel-filter-passengers", "value"),
    Input("intel-reset-filters-btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_intelligence_filters(n_clicks):
    """Resets all filter selections to their default neutral states."""
    return None, None, "all", "all"


# Callback 2: Main Filter & Analytics Rendering Callback (Task 1, 3, 4)
@callback(
    Output("intelligence-kpi-container", "children"),
    Output("chart-trips-over-time", "figure"),
    Output("chart-hourly-demand", "figure"),
    Output("chart-day-of-week", "figure"),
    Output("chart-fare-dist", "figure"),
    Output("chart-fare-vs-distance", "figure"),
    Output("chart-fare-by-hour", "figure"),
    Output("chart-pickup-borough", "figure"),
    Output("chart-dropoff-borough", "figure"),
    Output("chart-top-corridors", "figure"),
    Output("intel-mining-modules-container", "children"),
    Output("intel-filter-summary", "children"),
    Output("intel-demo-mode-banner", "children"),
    Output("intel-header-badge", "children"),
    Input("intel-data-source-toggle", "value"),
    Input("intel-filter-pu", "value"),
    Input("intel-filter-do", "value"),
    Input("intel-filter-time", "value"),
    Input("intel-filter-passengers", "value"),
    Input("intel-metric-selector", "value")
)
def update_intelligence_dashboard(data_source, pu_boroughs, do_boroughs, time_window, party_size, metric):
    """
    Evaluates active filters against Real or Demo dataset.
    Renders filter-responsive charts, KPIs, and status modules.
    Never fabricates fake data in Real Data mode.
    """
    mode = "demo" if data_source == "demo" else "real"
    data_status = analytics_service.get_data_status(data_mode=mode)

    time_param = None if time_window == "all" else time_window
    p_param = None if party_size == "all" else party_size

    # 1. Header Badge & Demo Banner Construction
    if mode == "demo":
        header_badge = dbc.Badge("Demo Mode (500 Synthetic Trips)", color="warning", className="drift-badge-status")
        demo_banner = dbc.Alert(
            [
                html.Div(
                    [
                        html.Span("🧪 ", className="fs-5 me-2"),
                        html.Strong("DEMO DATA — SYNTHETIC RECORDS (Simulation Mode Active): ", className="text-warning"),
                        "Displaying 500 deterministic synthetic NYC-style trip records generated strictly for UI demonstration, chart testing, and layout verification. "
                        "These are not genuine NYC TLC telemetry records. Switch to 'Real Data' anytime to inspect actual project outputs."
                    ],
                    className="small"
                )
            ],
            color="dark",
            className="drift-demo-banner py-2 px-3 mb-3"
        )
    else:
        if data_status.status == "ERROR":
            header_badge = dbc.Badge("Dataset Load Error", color="danger", className="drift-badge-status")
        elif data_status.is_loaded:
            header_badge = dbc.Badge(f"Snapshot: {data_status.total_rows:,} Trips", color="success", className="drift-badge-status")
        else:
            header_badge = dbc.Badge("Integration-Ready (Awaiting Dataset)", color="secondary", className="drift-badge-status")
        demo_banner = None

    # 2. Analytics & Visualizations Computation
    if data_status.is_loaded:
        filtered_df = analytics_service.filter_data(
            pu_boroughs=pu_boroughs,
            do_boroughs=do_boroughs,
            time_of_day=time_param,
            passenger_count=p_param,
            data_mode=mode
        )
        kpis = analytics_service.calculate_kpis(filtered_df)

        fig_trends = build_trips_over_time_chart(filtered_df)
        fig_hourly = build_hourly_demand_chart(filtered_df)
        fig_dow = build_day_of_week_chart(filtered_df)

        fig_fare_dist = build_fare_distribution_chart(filtered_df)
        fig_fare_dist_dist = build_fare_vs_distance_chart(filtered_df)
        fig_fare_by_hour = build_fare_by_hour_chart(filtered_df)

        fig_pu_b = build_borough_volume_chart(filtered_df, "PU_borough", "Pickup Demand by Borough")
        fig_do_b = build_borough_volume_chart(filtered_df, "DO_borough", "Destination Demand by Borough")
        fig_corridors = build_top_corridors_chart(filtered_df)

        filter_rev = f"{mode}_{pu_boroughs}_{do_boroughs}_{time_param}_{p_param}"
        fig_trends.layout.uirevision = filter_rev
        fig_hourly.layout.uirevision = filter_rev
        fig_dow.layout.uirevision = filter_rev
        fig_fare_dist.layout.uirevision = filter_rev
        fig_fare_dist_dist.layout.uirevision = filter_rev
        fig_fare_by_hour.layout.uirevision = filter_rev
        fig_pu_b.layout.uirevision = filter_rev
        fig_do_b.layout.uirevision = filter_rev
        fig_corridors.layout.uirevision = filter_rev

        mode_label = "Demo Records" if mode == "demo" else "Actual Records"
        summary_text = [
            html.Span(f"Showing {len(filtered_df):,} of {data_status.total_rows:,} {mode_label}", className="fw-semibold text-info"),
            html.Span("Filters: Active" if (pu_boroughs or do_boroughs or time_param or p_param) else "Filters: Inactive (Full Dataset)", className="drift-muted")
        ]
    elif data_status.status == "ERROR":
        filtered_df = pd.DataFrame()
        kpis = analytics_service.calculate_kpis(filtered_df)

        msg = f"Data Load Error: {data_status.message}"
        fig_trends = build_empty_chart("Daily Trip Demand Volume", msg)
        fig_hourly = build_empty_chart("24-Hour Diurnal Demand Curve", msg)
        fig_dow = build_empty_chart("Demand by Day of Week", msg)

        fig_fare_dist = build_empty_chart("Fare Distribution", msg)
        fig_fare_dist_dist = build_empty_chart("Fare vs Distance Scatter", msg)
        fig_fare_by_hour = build_empty_chart("Average Fare by Hour of Day", msg)

        fig_pu_b = build_empty_chart("Pickup Demand by Borough", msg)
        fig_do_b = build_empty_chart("Destination Demand by Borough", msg)
        fig_corridors = build_empty_chart("Top Mobility Corridors", msg)

        summary_text = [
            html.Span(f"Error loading dataset: {data_status.message}", className="text-danger"),
            html.Span("Verify file format / dependencies", className="drift-muted")
        ]
    else:
        # Awaiting data state (Real Data mode with no teammate file yet)
        filtered_df = pd.DataFrame()
        kpis = analytics_service.calculate_kpis(filtered_df)

        fig_trends = build_empty_chart("Daily Trip Demand Volume", "Awaiting telemetry dataset export", required_contract="data/processed_trips.parquet")
        fig_hourly = build_empty_chart("24-Hour Diurnal Demand Curve", "Awaiting telemetry dataset export", required_contract="hour_of_day [0–23]")
        fig_dow = build_empty_chart("Demand by Day of Week", "Awaiting telemetry dataset export", required_contract="day_of_week or datetime")

        fig_fare_dist = build_empty_chart("Fare Distribution", "Awaiting telemetry dataset export", required_contract="fare_amount ($)")
        fig_fare_dist_dist = build_empty_chart("Fare vs Distance Scatter", "Awaiting telemetry dataset export", required_contract="trip_distance (mi), fare_amount ($)")
        fig_fare_by_hour = build_empty_chart("Average Fare by Hour of Day", "Awaiting telemetry dataset export", required_contract="hour_of_day, fare_amount")

        fig_pu_b = build_empty_chart("Pickup Demand by Borough", "Awaiting telemetry dataset export", required_contract="PU_borough")
        fig_do_b = build_empty_chart("Destination Demand by Borough", "Awaiting telemetry dataset export", required_contract="DO_borough")
        fig_corridors = build_empty_chart("Top Mobility Corridors", "Awaiting telemetry dataset export", required_contract="PU_borough, DO_borough")

        summary_text = [
            html.Span("Real Data: Awaiting teammate dataset in 'data/' or '03_product_ui_monis/data/'", className="text-warning"),
            html.Span("Switch to 'Demo Data' above to preview live visualizations", className="text-info")
        ]

    kpi_elements = render_kpi_row(kpis)
    mining_elements = render_mining_modules(data_mode=mode)

    return (
        kpi_elements,
        fig_trends,
        fig_hourly,
        fig_dow,
        fig_fare_dist,
        fig_fare_dist_dist,
        fig_fare_by_hour,
        fig_pu_b,
        fig_do_b,
        fig_corridors,
        mining_elements,
        summary_text,
        demo_banner,
        header_badge
    )
