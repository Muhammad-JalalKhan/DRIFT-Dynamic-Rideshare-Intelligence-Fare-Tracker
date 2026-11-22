"""
DRIFT — Dynamic Rideshare Intelligence & Fare Tracker
Application Entry Point & Master Router
"""

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from pages.ride import ride_page

# Create the application
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    assets_folder="assets",
    title="DRIFT | Mobility Intelligence",
    suppress_callback_exceptions=True
)
server = app.server


# ---------- Welcome Page ----------

def welcome_page():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        html.Div(
                            "DRIFT / MOBILITY INTELLIGENCE",
                            className="drift-eyebrow"
                        ),
                        html.H1(
                            "The Future of Urban Mobility",
                            className="drift-hero-title"
                        ),
                        html.P(
                            "Smarter fares. Safer rides. "
                            "Intelligent cities.",
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
                                    html.Div(
                                        "01 / RIDER EXPERIENCE",
                                        className="drift-eyebrow"
                                    ),
                                    html.H2(
                                        "DRIFT Ride",
                                        className="mt-3"
                                    ),
                                    html.P(
                                        "Predict fares, explore trips "
                                        "and check suspicious charges.",
                                        className="drift-description"
                                    ),
                                    dcc.Link(
                                        dbc.Button(
                                            "Explore Ride →",
                                            color="info",
                                            className="mt-3"
                                        ),
                                        href="/ride"
                                    )
                                ]
                            ),
                            className="drift-feature-card"
                        ),
                        md=6,
                        className="mb-4"
                    ),

                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        "02 / DATA INTELLIGENCE",
                                        className="drift-eyebrow"
                                    ),
                                    html.H2(
                                        "DRIFT Intelligence",
                                        className="mt-3"
                                    ),
                                    html.P(
                                        "Discover mobility patterns, "
                                        "clusters and AI-driven insights.",
                                        className="drift-description"
                                    ),
                                    dcc.Link(
                                        dbc.Button(
                                            "Explore Intelligence →",
                                            color="secondary",
                                            className="mt-3"
                                        ),
                                        href="/intelligence"
                                    )
                                ]
                            ),
                            className="drift-feature-card"
                        ),
                        md=6,
                        className="mb-4"
                    )
                ],
                className="justify-content-center"
            )
        ],
        fluid=True,
        className="drift-welcome"
    )


# ---------- Intelligence Page (Placeholder) ----------

def intelligence_page():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    "02 / DATA INTELLIGENCE",
                                    className="drift-eyebrow mb-2"
                                ),
                                html.H2("DRIFT Intelligence", className="text-white mb-3"),
                                html.P(
                                    "Urban mobility pattern mining, spatial clustering, and anomaly detection "
                                    "modules are scheduled for Milestone 2 development.",
                                    className="drift-muted mb-4"
                                ),
                                dcc.Link(
                                    dbc.Button(
                                        "← Back to Portal",
                                        color="outline-light",
                                        size="sm",
                                        className="drift-btn-outline"
                                    ),
                                    href="/"
                                )
                            ]
                        ),
                        className="drift-card p-4"
                    ),
                    lg=8,
                    className="mx-auto"
                ),
                className="py-5"
            )
        ],
        fluid=True,
        className="drift-main"
    )


# ---------- Application Layout ----------

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
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

    if pathname == "/intelligence":
        return intelligence_page()

    return welcome_page()


if __name__ == "__main__":
    app.run(debug=True)
