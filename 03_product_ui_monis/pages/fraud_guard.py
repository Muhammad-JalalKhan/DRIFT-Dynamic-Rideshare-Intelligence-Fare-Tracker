"""
DRIFT — Receipt Fraud Guard Page
Architecture Layer: Audit & Anomaly Detection UI (/fraud-guard)

Interactive receipt auditing tool powered by teammate's trained Isolation Forest
model (isoforest_bundle.pkl). Evaluates fare, distance, and duration telemetry
to detect fraudulent overcharges, metering glitches, and pricing anomalies.
"""

from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from services.anomaly_service import anomaly_service, AnomalyResult


def fraud_guard_page():
    """Builds the Receipt Fraud Guard layout."""

    # Header Section
    header = dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Span("STAGE 4 ML", className="drift-badge-brand me-2"),
                            html.Span("FRAUD & ANOMALY AUDIT", className="drift-eyebrow")
                        ],
                        className="d-flex align-items-center mb-1"
                    ),
                    html.H2("Receipt Fraud Guard", className="drift-page-title mb-1"),
                    html.P(
                        "Machine learning anomaly detection for rideshare receipts against NYC TLC baseline distributions",
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
                                "Rider Experience"
                            ],
                            color="outline-light",
                            size="sm",
                            className="drift-btn-outline me-2"
                        ),
                        href="/ride"
                    ),
                    dcc.Link(
                        dbc.Button(
                            "Intelligence Hub →",
                            color="outline-info",
                            size="sm",
                            className="drift-btn-outline"
                        ),
                        href="/intelligence"
                    )
                ],
                xs=12,
                md=4,
                className="text-md-end align-self-center"
            )
        ],
        className="mb-4 pb-2 border-bottom border-secondary border-opacity-25"
    )

    # Preset Quick-Load Buttons
    preset_bar = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span("Quick-Load Test Receipts:", className="drift-muted small fw-semibold me-3 mb-2 mb-md-0"),
                        html.Div(
                            [
                                dbc.Button("Normal Manhattan Hop", id="fg-preset-normal-hop", size="sm", color="dark", className="drift-btn-utility me-2 mb-2"),
                                dbc.Button("Airport Run (JFK)", id="fg-preset-airport", size="sm", color="dark", className="drift-btn-utility me-2 mb-2"),
                                dbc.Button("Overcharged Short Trip", id="fg-preset-overcharge", size="sm", color="dark", className="drift-btn-utility me-2 mb-2 text-warning"),
                                dbc.Button("Metering Glitch (Duration)", id="fg-preset-glitch", size="sm", color="dark", className="drift-btn-utility mb-2 text-danger")
                            ],
                            className="d-flex flex-wrap"
                        )
                    ],
                    className="d-flex flex-column flex-md-row align-items-md-center justify-content-between"
                )
            ]
        ),
        className="drift-card mb-4"
    )

    # Input Form Card
    form_card = dbc.Card(
        dbc.CardBody(
            [
                html.H4("Receipt Parameters", className="text-white mb-1"),
                html.P("Enter trip distance, fare paid, and elapsed journey duration", className="drift-muted small mb-4"),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Trip Distance (miles)", className="drift-input-label"),
                                dbc.Input(
                                    id="fg-distance",
                                    type="number",
                                    min=0.1,
                                    max=100.0,
                                    step=0.1,
                                    value=2.5,
                                    className="drift-form-control mb-3"
                                )
                            ],
                            xs=12,
                            md=4
                        ),
                        dbc.Col(
                            [
                                html.Label("Total Fare Paid ($ USD)", className="drift-input-label"),
                                dbc.Input(
                                    id="fg-fare",
                                    type="number",
                                    min=1.0,
                                    max=500.0,
                                    step=0.5,
                                    value=14.5,
                                    className="drift-form-control mb-3"
                                )
                            ],
                            xs=12,
                            md=4
                        ),
                        dbc.Col(
                            [
                                html.Label("Trip Duration (minutes)", className="drift-input-label"),
                                dbc.Input(
                                    id="fg-duration",
                                    type="number",
                                    min=1.0,
                                    max=300.0,
                                    step=1.0,
                                    value=12.0,
                                    className="drift-form-control mb-3"
                                )
                            ],
                            xs=12,
                            md=4
                        )
                    ]
                ),

                html.Div(
                    [
                        dbc.Button(
                            "Audit Receipt Telemetry",
                            id="fg-audit-btn",
                            color="info",
                            className="drift-predict-btn me-2"
                        ),
                        dbc.Button(
                            "Reset",
                            id="fg-reset-btn",
                            color="dark",
                            className="drift-btn-utility"
                        )
                    ],
                    className="d-flex align-items-center mt-2"
                )
            ]
        ),
        className="drift-card mb-4"
    )

    # Result Output Container
    result_container = html.Div(id="fg-result-container")

    return dbc.Container(
        [
            header,
            preset_bar,
            form_card,
            dcc.Loading(
                id="loading-fraud-guard",
                type="dot",
                color="#22D3EE",
                children=result_container
            )
        ],
        fluid=True,
        className="drift-main"
    )


# ==========================================
# Fraud Guard Callbacks
# ==========================================

@callback(
    Output("fg-distance", "value"),
    Output("fg-fare", "value"),
    Output("fg-duration", "value"),
    Input("fg-preset-normal-hop", "n_clicks"),
    Input("fg-preset-airport", "n_clicks"),
    Input("fg-preset-overcharge", "n_clicks"),
    Input("fg-preset-glitch", "n_clicks"),
    Input("fg-reset-btn", "n_clicks"),
    prevent_initial_call=True
)
def handle_presets(click_normal, click_air, click_over, click_glitch, click_reset):
    """Sets input values based on preset buttons."""
    try:
        triggered = ctx.triggered_id
    except Exception:
        triggered = None

    if triggered is None:
        if click_normal:
            triggered = "fg-preset-normal-hop"
        elif click_air:
            triggered = "fg-preset-airport"
        elif click_over:
            triggered = "fg-preset-overcharge"
        elif click_glitch:
            triggered = "fg-preset-glitch"
        elif click_reset:
            triggered = "fg-reset-btn"

    if triggered == "fg-preset-normal-hop":
        return 2.5, 14.50, 12.0
    elif triggered == "fg-preset-airport":
        return 18.2, 65.00, 42.0
    elif triggered == "fg-preset-overcharge":
        return 0.6, 85.00, 5.0
    elif triggered == "fg-preset-glitch":
        return 1.2, 55.00, 68.0
    elif triggered == "fg-reset-btn":
        return 2.5, 14.50, 12.0
    return 2.5, 14.50, 12.0


@callback(
    Output("fg-result-container", "children"),
    Input("fg-audit-btn", "n_clicks"),
    Input("fg-preset-normal-hop", "n_clicks"),
    Input("fg-preset-airport", "n_clicks"),
    Input("fg-preset-overcharge", "n_clicks"),
    Input("fg-preset-glitch", "n_clicks"),
    State("fg-distance", "value"),
    State("fg-fare", "value"),
    State("fg-duration", "value")
)
def audit_receipt_callback(audit_clicks, p1, p2, p3, p4, distance, fare, duration):
    """Evaluates receipt parameters with AnomalyService and returns verdict UI."""
    if distance is None or fare is None or duration is None:
        return dbc.Alert("Please provide valid numerical values for all trip parameters.", color="warning")

    # Run audit through AnomalyService
    result: AnomalyResult = anomaly_service.audit_receipt(distance, fare, duration)

    if not result.is_available and result.status_code == "MODEL_AWAITING_INTEGRATION":
        return dbc.Card(
            dbc.CardBody(
                [
                    dbc.Badge("Model Awaiting Integration", color="warning", className="mb-2 py-1 px-2"),
                    html.H5("Isolation Forest Model Artifact Pending", className="text-white mb-2"),
                    html.P(result.message, className="drift-muted small mb-0")
                ]
            ),
            className="drift-card p-3 border-warning border-opacity-50"
        )

    if not result.is_available:
        return dbc.Alert(result.message, color="danger")

    is_anomaly = result.is_anomaly
    score = result.anomaly_score
    thresh = result.threshold
    fare_per_mile = float(fare) / max(0.1, float(distance))
    mph = float(distance) / (max(0.5, float(duration)) / 60.0)

    # Style by anomaly state
    badge_color = "danger" if is_anomaly else "success"
    badge_text = "UNUSUAL FARE PATTERN DETECTED" if is_anomaly else "VERIFIED NORMAL FARE PATTERN"
    card_border = "border-danger border-opacity-75" if is_anomaly else "border-success border-opacity-75"
    hero_icon = "⚠️" if is_anomaly else "✅"

    # Diagnostics items
    diagnostic_items = [
        html.Li(r, className="drift-diagnostic-li mb-2 text-light")
        for r in result.diagnostic_reasons
    ]

    return dbc.Card(
        dbc.CardBody(
            [
                # Status Banner
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(hero_icon, className="fs-2 me-3"),
                                html.Div(
                                    [
                                        dbc.Badge(badge_text, color=badge_color, className="drift-badge-status mb-1"),
                                        html.H4(
                                            "Audit Status: " + ("Flagged for Review" if is_anomaly else "Standard Telemetry"),
                                            className="text-white mb-0"
                                        )
                                    ]
                                )
                            ],
                            className="d-flex align-items-center"
                        ),
                        html.Div(
                            [
                                html.Span("Decision Rule: ", className="drift-muted small"),
                                html.Code(f"score ({score:.4f}) > threshold ({thresh:.4f})", className="drift-code-snippet")
                            ],
                            className="mt-2 mt-md-0"
                        )
                    ],
                    className="d-flex flex-column flex-md-row justify-content-between align-items-md-center pb-3 mb-3 border-bottom border-secondary border-opacity-25"
                ),

                # Metric Cards
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span("ISOLATION SCORE", className="drift-metric-label"),
                                    html.H3(
                                        f"{score:.4f}",
                                        className="drift-metric-val mb-0 " + ("text-danger" if is_anomaly else "text-info")
                                    ),
                                    html.Span(f"Threshold: {thresh:.4f}", className="drift-muted small")
                                ],
                                className="drift-kpi-box p-3 rounded"
                            ),
                            xs=12,
                            sm=4,
                            className="mb-3 mb-sm-0"
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span("COST PER MILE", className="drift-metric-label"),
                                    html.H3(
                                        f"${fare_per_mile:.2f}/mi",
                                        className="drift-metric-val mb-0 " + ("text-warning" if fare_per_mile > 10.0 else "text-white")
                                    ),
                                    html.Span("NYC norm: $3.50–$6.50/mi", className="drift-muted small")
                                ],
                                className="drift-kpi-box p-3 rounded"
                            ),
                            xs=12,
                            sm=4,
                            className="mb-3 mb-sm-0"
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span("AVERAGE SPEED", className="drift-metric-label"),
                                    html.H3(f"{mph:.1f} mph", className="drift-metric-val text-white mb-0"),
                                    html.Span(f"Duration: {float(duration):.0f} min for {float(distance):.1f} mi", className="drift-muted small")
                                ],
                                className="drift-kpi-box p-3 rounded"
                            ),
                            xs=12,
                            sm=4
                        )
                    ],
                    className="mb-4"
                ),

                # Diagnostic Breakdown
                html.Div(
                    [
                        html.H5("Diagnostic Analysis", className="text-white mb-2"),
                        html.Ul(diagnostic_items, className="ps-3 mb-3")
                    ],
                    className="drift-diagnostic-panel p-3 rounded mb-3"
                ),

                # Model Provenance Footer
                html.Div(
                    [
                        html.Span("Model Engine: ", className="drift-muted small"),
                        html.Span(result.model_version or "IsolationForest (scikit-learn 1.6.1)", className="text-light small fw-semibold me-3"),
                        html.Span("Contamination: 1.0% | Estimators: 200 | Max Samples: 1024", className="drift-muted-subtle small")
                    ],
                    className="pt-2 border-top border-secondary border-opacity-10"
                )
            ]
        ),
        className=f"drift-card {card_border}"
    )
