"""
DRIFT — Dynamic Rideshare Intelligence & Fare Tracker
Application Entry Point & Master Router
"""

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from pages.ride import ride_page
from pages.intelligence import intelligence_page
from pages.fraud_guard import fraud_guard_page

# Create the application with PWA and responsive viewport meta tags
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    assets_folder="assets",
    title="DRIFT | Dynamic Rideshare Intelligence",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover"},
        {"name": "theme-color", "content": "#0b1120"},
        {"name": "apple-mobile-web-app-capable", "content": "yes"},
        {"name": "apple-mobile-web-app-status-bar-style", "content": "black-translucent"},
        {"name": "description", "content": "DRIFT — Dynamic Rideshare Intelligence & Fare Tracker: NYC Urban Mobility Platform"}
    ],
    suppress_callback_exceptions=True
)
server = app.server


# ---------- Top Navigation Bar ----------

def build_navbar():
    return dbc.Navbar(
        dbc.Container(
            [
                dcc.Link(
                    html.Div(
                        [
                            html.Span("DRIFT", className="drift-badge-brand me-2"),
                            html.Span("NYC MOBILITY INTELLIGENCE", className="drift-eyebrow d-none d-sm-inline")
                        ],
                        className="d-flex align-items-center"
                    ),
                    href="/",
                    className="text-decoration-none"
                ),
                dbc.Nav(
                    [
                        dbc.NavItem(dcc.Link("Ride", href="/ride", className="nav-link drift-nav-link px-2")),
                        dbc.NavItem(dcc.Link("Fraud Guard", href="/fraud-guard", className="nav-link drift-nav-link text-info px-2")),
                        dbc.NavItem(dcc.Link("Intelligence", href="/intelligence", className="nav-link drift-nav-link px-2")),
                    ],
                    className="ms-auto d-flex flex-row gap-1 align-items-center"
                )
            ],
            fluid=True
        ),
        color="#0f172a",
        dark=True,
        className="drift-navbar border-bottom border-secondary border-opacity-25 py-2 px-3 mb-3"
    )


# ---------- Welcome Portal Page ----------

def welcome_page():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        html.Div(
                            "DRIFT / URBAN MOBILITY INTELLIGENCE PLATFORM",
                            className="drift-eyebrow"
                        ),
                        html.H1(
                            "The Future of Rideshare Intelligence",
                            className="drift-hero-title"
                        ),
                        html.P(
                            "Data-driven fare transparency, real-time receipt fraud auditing, and urban mobility data mining for New York City.",
                            className="drift-hero-description"
                        ),
                    ],
                    lg=9,
                    className="text-center mx-auto"
                ),
                className="mb-5"
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("01 / RIDER EXPERIENCE", className="drift-eyebrow"),
                                    html.H3("DRIFT Ride", className="mt-3 text-white"),
                                    html.P(
                                        "Plan journeys across 263 NYC TLC taxi zones, preview spatial route corridors on carto-dark maps, and view transparent fare intelligence.",
                                        className="drift-description"
                                    ),
                                    dcc.Link(
                                        dbc.Button("Explore Ride →", color="info", className="mt-3 drift-predict-btn"),
                                        href="/ride"
                                    )
                                ]
                            ),
                            className="drift-feature-card h-100"
                        ),
                        md=4,
                        className="mb-4"
                    ),

                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("02 / STAGE 4 ML AUDIT", className="drift-eyebrow text-info"),
                                    html.H3("Receipt Fraud Guard", className="mt-3 text-white"),
                                    html.P(
                                        "Audit rideshare fare receipts using trained Isolation Forest anomaly detection to catch meter overcharges, gridlock anomalies, and abnormal rates.",
                                        className="drift-description"
                                    ),
                                    dcc.Link(
                                        dbc.Button("Audit Receipts →", color="outline-info", className="mt-3 drift-btn-outline"),
                                        href="/fraud-guard"
                                    )
                                ]
                            ),
                            className="drift-feature-card h-100"
                        ),
                        md=4,
                        className="mb-4"
                    ),

                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("03 / DATA MINING HUB", className="drift-eyebrow"),
                                    html.H3("DRIFT Intelligence", className="mt-3 text-white"),
                                    html.P(
                                        "Explore full KDD pipeline analytics: K-Means mobility clusters, FP-Growth association rules, borough demand corridors, and real-time filters.",
                                        className="drift-description"
                                    ),
                                    dcc.Link(
                                        dbc.Button("Explore Intelligence →", color="secondary", className="mt-3 drift-btn-utility"),
                                        href="/intelligence"
                                    )
                                ]
                            ),
                            className="drift-feature-card h-100"
                        ),
                        md=4,
                        className="mb-4"
                    )
                ],
                className="justify-content-center g-4"
            )
        ],
        fluid=True,
        className="drift-welcome"
    )


# ---------- Application Layout ----------

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        build_navbar(),
        html.Div(id="page-content")
    ]
)


# ---------- Navigation Callback ----------

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/ride":
        return ride_page()

    if pathname == "/fraud-guard":
        return fraud_guard_page()

    if pathname == "/intelligence":
        return intelligence_page()

    return welcome_page()


if __name__ == "__main__":
    import os
    debug_mode = os.environ.get("DRIFT_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(debug=debug_mode, dev_tools_ui=debug_mode)
