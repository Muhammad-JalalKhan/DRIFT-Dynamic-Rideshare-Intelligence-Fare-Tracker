"""
DRIFT — Dynamic Rideshare Intelligence & Fare Tracker
Rider Experience (DRIFT Ride) — Page Module

Provides the interactive journey planner, Plotly NYC carto-dark map,
trip detail selectors, dynamic map marker callbacks, and an
integration-ready fare prediction status interface.
"""

from datetime import date
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from services.prediction_service import fare_service

# Temporary representative coordinates for NYC boroughs.
# These are demonstration coordinates, not official NYC taxi-zone coordinates.
LOCATION_COORDINATES = {
    "manhattan": {
        "name": "Manhattan",
        "lat": 40.7831,
        "lon": -73.9712,
        "desc": "Manhattan (Demo Centroid)"
    },
    "brooklyn": {
        "name": "Brooklyn",
        "lat": 40.6782,
        "lon": -73.9442,
        "desc": "Brooklyn (Demo Centroid)"
    },
    "queens": {
        "name": "Queens",
        "lat": 40.7282,
        "lon": -73.7949,
        "desc": "Queens (Demo Centroid)"
    }
}

LOCATION_OPTIONS = [
    {"label": "Manhattan", "value": "manhattan"},
    {"label": "Brooklyn", "value": "brooklyn"},
    {"label": "Queens", "value": "queens"}
]

TIME_OPTIONS = [
    {"label": "08:00 AM — Morning Peak", "value": "08:00"},
    {"label": "12:00 PM — Midday Regular", "value": "12:00"},
    {"label": "05:30 PM — Evening Peak", "value": "17:30"},
    {"label": "09:00 PM — Night Standard", "value": "21:00"},
    {"label": "01:30 AM — Late Night / Early AM", "value": "01:30"}
]

PASSENGER_OPTIONS = [
    {"label": f"{i} Passenger{'s' if i > 1 else ''}", "value": i}
    for i in range(1, 7)
]


# ==========================================
# Map Generation Function
# ==========================================

def create_nyc_map(pickup: str = None, dropoff: str = None):
    """
    Renders an interactive Plotly map centered on New York City.
    Dynamically plots:
      - Cyan marker (#22D3EE) for Pickup Location
      - Amber marker (#F59E0B) for Destination Location
    Handles identical locations cleanly without visual collision.
    """
    fig = go.Figure()

    has_pickup = pickup in LOCATION_COORDINATES
    has_dropoff = dropoff in LOCATION_COORDINATES
    is_same_location = has_pickup and has_dropoff and (pickup == dropoff)

    center_lat = 40.7306
    center_lon = -73.9352
    zoom_level = 9.8

    if not has_pickup and not has_dropoff:
        # Default NYC overview marker
        fig.add_trace(
            go.Scattermap(
                lat=[40.7128],
                lon=[-74.0060],
                mode="markers",
                marker={
                    "size": 10,
                    "color": "#64748B",
                    "opacity": 0.7
                },
                text=["New York City (Select Pickup & Destination)"],
                hovertemplate="<b>%{text}</b><extra></extra>",
                showlegend=False
            )
        )
    elif is_same_location:
        # Same location selected: plot distinct markers with slight visual separation
        loc = LOCATION_COORDINATES[pickup]
        center_lat = loc["lat"]
        center_lon = loc["lon"]
        zoom_level = 11.2

        # Pickup marker (Cyan)
        fig.add_trace(
            go.Scattermap(
                lat=[loc["lat"] - 0.0025],
                lon=[loc["lon"] - 0.0025],
                mode="markers+text",
                marker={
                    "size": 16,
                    "color": "#22D3EE",
                    "symbol": "circle"
                },
                text=["PU"],
                textposition="bottom center",
                textfont={"color": "#22D3EE", "size": 11, "family": "Inter, sans-serif"},
                name="Pickup Zone",
                hovertemplate=f"<b>Pickup: {loc['name']}</b><br>Intra-borough trip<extra></extra>"
            )
        )

        # Destination marker (Amber)
        fig.add_trace(
            go.Scattermap(
                lat=[loc["lat"] + 0.0025],
                lon=[loc["lon"] + 0.0025],
                mode="markers+text",
                marker={
                    "size": 16,
                    "color": "#F59E0B",
                    "symbol": "circle"
                },
                text=["DO"],
                textposition="top center",
                textfont={"color": "#F59E0B", "size": 11, "family": "Inter, sans-serif"},
                name="Destination Zone",
                hovertemplate=f"<b>Destination: {loc['name']}</b><br>Intra-borough trip<extra></extra>"
            )
        )
    else:
        # Distinct locations
        lats = []
        lons = []

        if has_pickup:
            loc_pu = LOCATION_COORDINATES[pickup]
            lats.append(loc_pu["lat"])
            lons.append(loc_pu["lon"])
            fig.add_trace(
                go.Scattermap(
                    lat=[loc_pu["lat"]],
                    lon=[loc_pu["lon"]],
                    mode="markers+text",
                    marker={
                        "size": 16,
                        "color": "#22D3EE",
                        "symbol": "circle"
                    },
                    text=["Pickup"],
                    textposition="top center",
                    textfont={"color": "#22D3EE", "size": 11, "family": "Inter, sans-serif"},
                    name="Pickup",
                    hovertemplate=f"<b>Pickup: {loc_pu['name']}</b><extra></extra>"
                )
            )

        if has_dropoff:
            loc_do = LOCATION_COORDINATES[dropoff]
            lats.append(loc_do["lat"])
            lons.append(loc_do["lon"])
            fig.add_trace(
                go.Scattermap(
                    lat=[loc_do["lat"]],
                    lon=[loc_do["lon"]],
                    mode="markers+text",
                    marker={
                        "size": 16,
                        "color": "#F59E0B",
                        "symbol": "circle"
                    },
                    text=["Destination"],
                    textposition="top center",
                    textfont={"color": "#F59E0B", "size": 11, "family": "Inter, sans-serif"},
                    name="Destination",
                    hovertemplate=f"<b>Destination: {loc_do['name']}</b><extra></extra>"
                )
            )

        if len(lats) == 2:
            center_lat = sum(lats) / 2
            center_lon = sum(lons) / 2
            zoom_level = 10.3
        elif len(lats) == 1:
            center_lat = lats[0]
            center_lon = lons[0]
            zoom_level = 11.0

    fig.update_layout(
        map={
            "style": "carto-darkmatter",
            "center": {
                "lat": center_lat,
                "lon": center_lon
            },
            "zoom": zoom_level
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        autosize=True,
        paper_bgcolor="#172338",
        plot_bgcolor="#172338",
        showlegend=False,
        uirevision="drift-nyc-map"
    )

    return fig


# ==========================================
# DRIFT — Ride Page Layout
# ==========================================

def ride_page():
    # --------------------------------------
    # Header Section
    # --------------------------------------
    header_section = dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Span("DRIFT", className="drift-badge-brand me-2"),
                            html.Span("RIDER INTELLIGENCE", className="drift-eyebrow")
                        ],
                        className="d-flex align-items-center mb-1"
                    ),
                    html.H2("DRIFT Ride", className="drift-page-title mb-1"),
                    html.P(
                        "Urban Mobility Journey Planner & Fare Intelligence",
                        className="drift-muted mb-0"
                    )
                ],
                xs=12,
                md=8,
                className="mb-3 mb-md-0"
            ),
            dbc.Col(
                [
                    dcc.Link(
                        dbc.Button(
                            [
                                html.Span("←", className="me-2"),
                                "Back to Portal"
                            ],
                            color="outline-light",
                            size="sm",
                            className="drift-btn-outline"
                        ),
                        href="/"
                    )
                ],
                xs=12,
                md=4,
                className="text-md-end align-self-center"
            )
        ],
        className="mb-4 pb-2 border-bottom border-secondary border-opacity-25"
    )

    # --------------------------------------
    # Left Column: Journey Planner & Trip Details
    # --------------------------------------
    journey_planner_card = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H4("Journey Planner", className="mb-1 text-white"),
                        html.P("Select zones to preview route on NYC network", className="drift-muted small mb-3")
                    ]
                ),

                # Pickup Dropdown
                html.Div(
                    [
                        html.Label(
                            [
                                html.Span("●", style={"color": "#22D3EE"}, className="me-2"),
                                "Pickup Location"
                            ],
                            className="drift-input-label"
                        ),
                        dcc.Dropdown(
                            id="pickup-location",
                            options=LOCATION_OPTIONS,
                            placeholder="Select pickup borough...",
                            clearable=True,
                            searchable=True,
                            className="drift-dropdown mb-3"
                        )
                    ]
                ),

                # Quick swap button
                html.Div(
                    [
                        dbc.Button(
                            "⇅ Swap Locations",
                            id="swap-locations-btn",
                            size="sm",
                            color="dark",
                            className="drift-btn-utility mb-3"
                        ),
                        dbc.Button(
                            "↺ Reset",
                            id="reset-trip-btn",
                            size="sm",
                            color="dark",
                            className="drift-btn-utility ms-2 mb-3"
                        )
                    ],
                    className="d-flex justify-content-end"
                ),

                # Destination Dropdown
                html.Div(
                    [
                        html.Label(
                            [
                                html.Span("●", style={"color": "#F59E0B"}, className="me-2"),
                                "Destination Location"
                            ],
                            className="drift-input-label"
                        ),
                        dcc.Dropdown(
                            id="dropoff-location",
                            options=LOCATION_OPTIONS,
                            placeholder="Select destination borough...",
                            clearable=True,
                            searchable=True,
                            className="drift-dropdown mb-3"
                        )
                    ]
                ),

                # Same location alert container
                html.Div(id="same-location-alert"),

                html.Hr(className="border-secondary border-opacity-25 my-3"),

                # Feature C: Trip Details
                html.Div(
                    [
                        html.H5("Trip Parameters", className="text-white mb-2"),
                        html.P(
                            "Model feature inputs (aligned with TLC pipeline)",
                            className="drift-muted small mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Pickup Date", className="drift-input-label"),
                                        dcc.DatePickerSingle(
                                            id="trip-date",
                                            date=date.today().isoformat(),
                                            display_format="YYYY-MM-DD",
                                            className="drift-date-picker w-100"
                                        )
                                    ],
                                    xs=12,
                                    sm=6,
                                    className="mb-3"
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Time of Day", className="drift-input-label"),
                                        dcc.Dropdown(
                                            id="trip-time",
                                            options=TIME_OPTIONS,
                                            value="08:00",
                                            clearable=False,
                                            searchable=False,
                                            className="drift-dropdown"
                                        )
                                    ],
                                    xs=12,
                                    sm=6,
                                    className="mb-3"
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Passenger Count", className="drift-input-label"),
                                        dcc.Dropdown(
                                            id="trip-passengers",
                                            options=PASSENGER_OPTIONS,
                                            value=1,
                                            clearable=False,
                                            searchable=False,
                                            className="drift-dropdown"
                                        )
                                    ],
                                    xs=12,
                                    className="mb-2"
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        className="drift-card mb-4"
    )

    # --------------------------------------
    # Left Column: Fare Estimation Interface
    # --------------------------------------
    fare_prediction_card = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("ESTIMATION ENGINE", className="drift-eyebrow"),
                                html.H4("Fare Intelligence", className="text-white mt-1 mb-1")
                            ]
                        ),
                        dbc.Button(
                            "Estimate Fare",
                            id="estimate-fare-btn",
                            color="info",
                            className="w-100 drift-predict-btn mt-2 mb-3",
                            disabled=True
                        )
                    ]
                ),

                # Dynamic Results & Feedback Container
                dcc.Loading(
                    id="loading-prediction",
                    type="dot",
                    color="#22D3EE",
                    children=html.Div(
                        id="fare-result-container",
                        className="drift-results-panel"
                    )
                )
            ]
        ),
        className="drift-card mb-4"
    )

    # --------------------------------------
    # Right Column: Interactive NYC Map
    # --------------------------------------
    map_card = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("NYC Mobility Network", className="text-white mb-0"),
                                html.P(
                                    "Centroid visualization of selected trip endpoints",
                                    className="drift-muted small mb-0"
                                )
                            ]
                        ),
                        html.Div(
                            id="map-status-pill",
                            className="mt-2 mt-md-0"
                        )
                    ],
                    className="d-flex flex-column flex-md-row justify-content-between align-items-md-center mb-3"
                ),

                html.Div(
                    dcc.Graph(
                        id="ride-map",
                        figure=create_nyc_map(),
                        config={
                            "displayModeBar": False,
                            "scrollZoom": True,
                            "responsive": True
                        },
                        style={"width": "100%", "height": "100%"}
                    ),
                    className="drift-map-container"
                ),

                # Map Legend & Disclaimer
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(className="drift-legend-dot legend-pickup me-1"),
                                html.Span("Pickup", className="drift-muted small me-3"),
                                html.Span(className="drift-legend-dot legend-dropoff me-1"),
                                html.Span("Destination", className="drift-muted small")
                            ],
                            className="d-flex align-items-center"
                        ),
                        html.Div(
                            html.Span(
                                "Demonstration coordinates. TLC Taxi Zone boundaries pending backend dataset.",
                                className="drift-muted-subtle small"
                            )
                        )
                    ],
                    className="d-flex flex-column flex-sm-row justify-content-between align-items-sm-center mt-3 pt-2 border-top border-secondary border-opacity-10"
                )
            ]
        ),
        className="drift-card h-100"
    )

    # --------------------------------------
    # Page Layout Assembly
    # --------------------------------------
    return dbc.Container(
        [
            header_section,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            journey_planner_card,
                            fare_prediction_card
                        ],
                        xs=12,
                        lg=5,
                        className="mb-4 mb-lg-0"
                    ),
                    dbc.Col(
                        map_card,
                        xs=12,
                        lg=7,
                        className="mb-4"
                    )
                ],
                className="g-4"
            )
        ],
        fluid=True,
        className="drift-main"
    )


# ==========================================
# Dynamic Callbacks
# ==========================================

# 1. Swap & Reset Trip Locations Callback
@callback(
    Output("pickup-location", "value"),
    Output("dropoff-location", "value"),
    Input("swap-locations-btn", "n_clicks"),
    Input("reset-trip-btn", "n_clicks"),
    State("pickup-location", "value"),
    State("dropoff-location", "value"),
    prevent_initial_call=True
)
def handle_trip_actions(swap_clicks, reset_clicks, current_pu, current_do):
    triggered_id = ctx.triggered_id
    if triggered_id == "swap-locations-btn":
        return current_do, current_pu
    elif triggered_id == "reset-trip-btn":
        return None, None
    return current_pu, current_do


# 2. Update Map Markers Callback
@callback(
    Output("ride-map", "figure"),
    Output("map-status-pill", "children"),
    Output("same-location-alert", "children"),
    Input("pickup-location", "value"),
    Input("dropoff-location", "value")
)
def update_map(pickup, dropoff):
    figure = create_nyc_map(pickup, dropoff)

    # Build status pill for map header
    has_pu = pickup in LOCATION_COORDINATES
    has_do = dropoff in LOCATION_COORDINATES

    if has_pu and has_do:
        pill = dbc.Badge("Both Zones Mapped", color="info", className="drift-badge-status")
    elif has_pu:
        pill = dbc.Badge("Pickup Set", color="dark", className="drift-badge-status text-info")
    elif has_do:
        pill = dbc.Badge("Destination Set", color="dark", className="drift-badge-status text-warning")
    else:
        pill = dbc.Badge("NYC Overview", color="secondary", className="drift-badge-status")

    # Intra-borough alert if applicable
    alert = None
    if has_pu and has_do and pickup == dropoff:
        borough_name = LOCATION_COORDINATES[pickup]["name"]
        alert = dbc.Alert(
            [
                html.Span("ℹ️ ", className="me-1"),
                html.Strong("Intra-zone trip: "),
                f"Both pickup and destination are in {borough_name}."
            ],
            color="info",
            className="drift-alert small py-2 px-3 mt-2 mb-0"
        )

    return figure, pill, alert


# 3. Dynamic Fare Feedback & Prediction Status Callback
@callback(
    Output("estimate-fare-btn", "disabled"),
    Output("estimate-fare-btn", "children"),
    Output("fare-result-container", "children"),
    Input("pickup-location", "value"),
    Input("dropoff-location", "value"),
    Input("trip-date", "date"),
    Input("trip-time", "value"),
    Input("trip-passengers", "value"),
    Input("estimate-fare-btn", "n_clicks")
)
def handle_fare_intelligence(pickup, dropoff, trip_date, trip_time, passengers, estimate_clicks):
    """
    Evaluates trip state and returns clean feedback.
    Never fabricates random or synthetic fares.
    Transparently reports model status.
    """
    has_pu = bool(pickup)
    has_do = bool(dropoff)

    # Case 1: Incomplete selection
    if not has_pu or not has_do:
        missing_text = "pickup and destination" if not has_pu and not has_do else ("destination" if not has_do else "pickup")
        feedback = html.Div(
            [
                html.Div(
                    [
                        html.Span("📍", className="fs-4 me-2"),
                        html.Div(
                            [
                                html.P(
                                    f"Please select {missing_text} zone to configure journey.",
                                    className="text-white mb-1 small fw-semibold"
                                ),
                                html.P(
                                    "Once selected, your trip parameters will be validated for fare estimation.",
                                    className="drift-muted small mb-0"
                                )
                            ]
                        )
                    ],
                    className="d-flex align-items-center"
                )
            ],
            className="drift-feedback-empty p-3 rounded"
        )
        return True, "Select Locations to Estimate", feedback

    # Both locations selected: query the prediction service
    pu_name = LOCATION_COORDINATES.get(pickup, {}).get("name", pickup.title())
    do_name = LOCATION_COORDINATES.get(dropoff, {}).get("name", dropoff.title())
    res = fare_service.predict_fare(
        pickup_location=pickup,
        dropoff_location=dropoff,
        pickup_date=trip_date,
        pickup_time=trip_time,
        passenger_count=passengers
    )

    if res.is_available and res.predicted_fare is not None:
        # ML Model Active branch (milestone 2)
        fare_display = html.Div(
            [
                html.Div(
                    [
                        html.Span("$", className="drift-fare-symbol"),
                        html.Span(f"{res.predicted_fare:.2f}", className="drift-fare-value"),
                        html.Span("USD", className="drift-fare-currency ms-2")
                    ],
                    className="d-flex align-items-baseline mb-2"
                ),
                html.P(res.message, className="text-success small mb-0")
            ],
            className="drift-fare-box p-3 rounded mb-3"
        )
        btn_label = "Update Estimate"
    else:
        # Awaiting ML Integration branch (honest and transparent)
        fare_display = html.Div(
            [
                dbc.Badge(
                    "Model Awaiting Integration",
                    color="warning",
                    className="drift-badge-warning mb-2 py-1 px-2"
                ),
                html.H5("Prediction Engine Pending", className="text-white mb-2"),
                html.P(
                    res.message,
                    className="drift-muted small mb-3"
                ),
                html.Div(
                    [
                        html.Strong("Contract Payload Ready:", className="text-info small d-block mb-1"),
                        html.Code(
                            f"PULocation: {pu_name} | DOLocation: {do_name} | Date: {trip_date} | Time: {trip_time} | Passengers: {passengers}",
                            className="drift-code-snippet"
                        )
                    ],
                    className="drift-contract-box p-2 rounded"
                )
            ],
            className="drift-fare-box-pending p-3 rounded mb-2"
        )
        btn_label = "Model Integration Pending"

    # Trip Summary Strip
    summary_strip = html.Div(
        [
            html.Div(
                [
                    html.Span("Route: ", className="drift-muted small"),
                    html.Strong(f"{pu_name} → {do_name}", className="text-white small")
                ],
                className="mb-1"
            ),
            html.Div(
                [
                    html.Span("Schedule: ", className="drift-muted small"),
                    html.Span(f"{trip_date or 'Today'} at {trip_time or 'Standard'}", className="text-light small me-3"),
                    html.Span("Party: ", className="drift-muted small"),
                    html.Span(f"{passengers} Rider{'s' if passengers > 1 else ''}", className="text-light small")
                ]
            )
        ],
        className="drift-summary-strip p-2 px-3 rounded mb-3"
    )

    full_output = html.Div([summary_strip, fare_display])
    # Button is active if model is ready, or disabled in pending state
    button_disabled = not fare_service.is_model_ready()
    return button_disabled, btn_label, full_output
