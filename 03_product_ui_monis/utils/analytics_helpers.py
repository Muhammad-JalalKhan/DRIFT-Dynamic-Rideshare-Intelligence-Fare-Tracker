"""
DRIFT — Analytics & Visualization Helpers
UI Component & Plotly Chart Builders for DRIFT Intelligence
"""

from typing import Optional, List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import numpy as np

# ==============================================================================
# Theme Constants
# ==============================================================================
THEME_BG_CARD = "#172338"
THEME_BG_SURFACE = "#0E1B2C"
THEME_BG_DARK = "#07111F"
THEME_TEXT_PRIMARY = "#F8FAFC"
THEME_TEXT_SECONDARY = "#94A3B8"
THEME_TEXT_MUTED = "#64748B"
THEME_BORDER = "#334155"
THEME_BORDER_SUBTLE = "#1E293B"
THEME_CYAN = "#22D3EE"
THEME_AMBER = "#F59E0B"
THEME_PURPLE = "#818CF8"
THEME_GREEN = "#10B981"
THEME_RED = "#EF4444"


def apply_dark_layout(
    fig: go.Figure,
    title: str = "",
    height: int = 340,
    margin: Optional[dict] = None
) -> go.Figure:
    """Applies unified DRIFT dark aesthetic to any Plotly figure."""
    chart_margin = margin if margin is not None else {"l": 50, "r": 20, "t": 50, "b": 45}
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 14, "color": THEME_TEXT_PRIMARY, "family": "-apple-system, Inter, sans-serif"},
            "x": 0.02,
            "y": 0.95
        },
        paper_bgcolor=THEME_BG_CARD,
        plot_bgcolor=THEME_BG_SURFACE,
        margin=chart_margin,
        height=height,
        font={"color": THEME_TEXT_SECONDARY, "family": "-apple-system, Inter, sans-serif", "size": 12},
        xaxis={
            "gridcolor": "rgba(51, 65, 85, 0.35)",
            "linecolor": THEME_BORDER,
            "zerolinecolor": THEME_BORDER,
            "tickfont": {"color": THEME_TEXT_SECONDARY, "size": 11}
        },
        yaxis={
            "gridcolor": "rgba(51, 65, 85, 0.35)",
            "linecolor": THEME_BORDER,
            "zerolinecolor": THEME_BORDER,
            "tickfont": {"color": THEME_TEXT_SECONDARY, "size": 11}
        },
        hoverlabel={
            "bgcolor": "#0B1120",
            "bordercolor": THEME_CYAN,
            "font": {"color": THEME_TEXT_PRIMARY, "size": 12}
        },
        legend={
            "font": {"color": THEME_TEXT_SECONDARY, "size": 11},
            "bgcolor": "rgba(14, 27, 44, 0.6)",
            "bordercolor": THEME_BORDER
        },
        autosize=True
    )
    return fig


# ==============================================================================
# Empty State Chart Builder
# ==============================================================================

def build_empty_chart(
    title: str,
    message: str,
    required_contract: Optional[str] = None,
    height: int = 300,
    header: Optional[str] = None
) -> go.Figure:
    """Renders a clean, responsive empty-state chart without horizontal text clipping."""
    fig = go.Figure()

    header_text = header if header is not None else "📂 Awaiting Telemetry Export"
    lines = [
        f"<b>{header_text}</b>",
        f"<span style='font-size:12px; color:#94A3B8;'>{message}</span>"
    ]
    if required_contract:
        lines.append(f"<span style='font-size:10px; color:#22D3EE;'>Expected: {required_contract}</span>")

    formatted_text = "<br>".join(lines)

    fig.add_annotation(
        text=formatted_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"color": THEME_TEXT_MUTED, "size": 12, "family": "-apple-system, Inter, sans-serif"},
        align="center",
        bgcolor="rgba(14, 27, 44, 0.85)",
        bordercolor=THEME_BORDER,
        borderwidth=1,
        borderpad=12
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, visible=False)
    return apply_dark_layout(fig, title=title, height=height)


# ==============================================================================
# KPI Card Component
# ==============================================================================

def build_kpi_card(
    title: str,
    value: str,
    unit: str = "",
    icon: str = "📊",
    subtitle: str = "",
    status: str = "active"
) -> dbc.Card:
    """Builds an executive KPI metric card with responsive styling."""
    is_pending = status == "pending"
    val_color = THEME_TEXT_MUTED if is_pending else THEME_TEXT_PRIMARY
    badge = (
        dbc.Badge("Awaiting Data", color="secondary", className="drift-badge-status small ms-auto")
        if is_pending else None
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span(icon, className="drift-kpi-icon me-2"),
                        html.Span(title, className="drift-input-label mb-0"),
                        badge
                    ],
                    className="d-flex align-items-center mb-2"
                ),
                html.Div(
                    [
                        html.Span(value, className="drift-kpi-val", style={"color": val_color}),
                        html.Span(f" {unit}", className="drift-kpi-unit ms-1", style={"color": THEME_CYAN}) if unit else None
                    ],
                    className="d-flex align-items-baseline"
                ),
                html.P(subtitle, className="drift-muted small mt-2 mb-0") if subtitle else None
            ],
            className="p-3"
        ),
        className="drift-card drift-kpi-card h-100"
    )


# ==============================================================================
# Visualization Functions (Grounded in Real Data)
# ==============================================================================

def build_trips_over_time_chart(df: pd.DataFrame, date_col: str = "tpep_pickup_datetime") -> go.Figure:
    """Line/Area or discrete Bar chart showing daily trip frequency over time."""
    if df.empty or date_col not in df.columns:
        if df.empty:
            return build_empty_chart("Trips Over Time", "Zero records match active filter criteria. Try resetting filters.", header="🔍 No Matching Records")
        return build_empty_chart("Trips Over Time", "No timestamped trip data available", required_contract=f"'{date_col}' (datetime)")

    try:
        temp = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(temp[date_col]):
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col])
        if temp.empty:
            return build_empty_chart("Trips Over Time", "Timestamp conversion yielded zero valid dates", header="🔍 No Matching Records")

        # Group strictly by calendar date (removes hours, minutes, seconds, microseconds)
        temp["_calendar_date"] = temp[date_col].dt.date
        daily_counts = temp.groupby("_calendar_date").size().reset_index(name="trip_count")
        daily_counts.columns = ["date", "trip_count"]
        daily_counts = daily_counts.sort_values("date").reset_index(drop=True)

        dates_dt = pd.to_datetime(daily_counts["date"])
        # Format cleanly as "Nov 22" or "22 Nov 2026" (if multi-year or long span)
        if dates_dt.dt.year.nunique() > 1:
            daily_counts["date_label"] = dates_dt.dt.strftime("%d %b %Y")
        else:
            daily_counts["date_label"] = dates_dt.dt.strftime("%b %d")

        fig = go.Figure()

        # Small sample visualization: When 1 to 3 records/dates match, use Bar chart for clean readability
        if len(daily_counts) <= 3:
            bar_w = 0.3 if len(daily_counts) == 1 else (0.4 if len(daily_counts) == 2 else 0.45)
            fig.add_trace(
                go.Bar(
                    x=daily_counts["date_label"],
                    y=daily_counts["trip_count"],
                    marker={"color": THEME_CYAN, "line": {"color": THEME_BORDER, "width": 1}},
                    width=bar_w,
                    name="Trips",
                    hovertemplate="<b>Date:</b> %{x}<br><b>Trips:</b> %{y:,}<extra></extra>"
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=daily_counts["date_label"],
                    y=daily_counts["trip_count"],
                    mode="lines+markers",
                    line={"color": THEME_CYAN, "width": 2.5},
                    marker={"size": 6, "color": THEME_CYAN},
                    fill="tozeroy",
                    fillcolor="rgba(34, 211, 238, 0.08)",
                    name="Trips",
                    hovertemplate="<b>Date:</b> %{x}<br><b>Trips:</b> %{y:,}<extra></extra>"
                )
            )

        # For small trip counts, enforce integer ticks so ticks like 0.2, 0.4 do not appear
        yaxis_cfg = {"title": "Trips", "rangemode": "tozero", "tick0": 0}
        if daily_counts["trip_count"].max() <= 5:
            yaxis_cfg.update({"tickformat": "d", "dtick": 1, "tick0": 0})

        fig.update_layout(
            xaxis={
                "type": "category",
                "title": "Date",
                "tickangle": -30 if len(daily_counts) > 10 else 0
            },
            yaxis=yaxis_cfg
        )
        return apply_dark_layout(fig, title="Daily Trip Demand Volume", height=320)
    except Exception as e:
        return build_empty_chart("Trips Over Time", f"Unable to aggregate temporal data: {str(e)}")


def build_hourly_demand_chart(df: pd.DataFrame, hour_col: str = "hour_of_day") -> go.Figure:
    """Bar chart illustrating 24-hour mobility diurnal patterns strictly bounded 0–23."""
    if df.empty:
        return build_empty_chart("Hourly Demand Pattern", "Zero records match active filter criteria. Try resetting filters.", header="🔍 No Matching Records")

    try:
        temp = df.copy()
        if hour_col not in temp.columns:
            if "tpep_pickup_datetime" in temp.columns:
                temp["tpep_pickup_datetime"] = pd.to_datetime(temp["tpep_pickup_datetime"], errors="coerce")
                temp[hour_col] = temp["tpep_pickup_datetime"].dt.hour
            else:
                return build_empty_chart("Hourly Demand Pattern", "Missing hour field", required_contract=f"'{hour_col}' [0-23] or datetime")

        hourly = temp.groupby(hour_col).size().reindex(range(24), fill_value=0).reset_index()
        hourly.columns = ["hour", "trip_count"]

        # Peak hours highlight (e.g. 4pm-7pm amber, regular cyan)
        colors = [THEME_AMBER if 16 <= h <= 19 else THEME_CYAN for h in hourly["hour"]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=hourly["hour"],
                y=hourly["trip_count"],
                marker={"color": colors, "line": {"color": THEME_BORDER, "width": 1}},
                hovertemplate="<b>Hour %{x}:00:</b> %{y:,} pickups<extra></extra>"
            )
        )
        yaxis_cfg = {"title": "Trip Count", "rangemode": "tozero", "tick0": 0}
        if hourly["trip_count"].max() <= 5:
            yaxis_cfg.update({"tickformat": "d", "dtick": 1, "tick0": 0})

        fig.update_layout(
            xaxis={
                "title": "Hour of Day (0–23)",
                "tickmode": "array",
                "tickvals": list(range(0, 24, 2)),
                "ticktext": [f"{h:02d}:00" for h in range(0, 24, 2)],
                "range": [-0.5, 23.5]
            },
            yaxis=yaxis_cfg
        )
        return apply_dark_layout(fig, title="24-Hour Diurnal Demand Curve", height=320)
    except Exception as e:
        return build_empty_chart("Hourly Demand Pattern", f"Error calculating hourly demand: {str(e)}")


def build_day_of_week_chart(df: pd.DataFrame) -> go.Figure:
    """Compares weekday vs weekend trip activity."""
    if df.empty:
        return build_empty_chart("Day of Week Demand", "Zero records match active filter criteria. Try resetting filters.", header="🔍 No Matching Records")

    try:
        temp = df.copy()
        if "day_of_week" in temp.columns:
            dow_series = temp["day_of_week"]
        elif "tpep_pickup_datetime" in temp.columns:
            temp["tpep_pickup_datetime"] = pd.to_datetime(temp["tpep_pickup_datetime"], errors="coerce")
            dow_series = temp["tpep_pickup_datetime"].dt.dayofweek
        else:
            return build_empty_chart("Day of Week Demand", "Missing weekday field", required_contract="'day_of_week' or datetime")

        days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        counts = dow_series.map(days_map).value_counts().reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fill_value=0)

        colors = [THEME_AMBER if d in ["Fri", "Sat"] else THEME_CYAN for d in counts.index]

        fig = go.Figure(
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker={"color": colors, "line": {"color": THEME_BORDER, "width": 1}},
                hovertemplate="<b>%{x}:</b> %{y:,} trips<extra></extra>"
            )
        )
        yaxis_cfg = {"title": "Trips", "rangemode": "tozero", "tick0": 0}
        if counts.max() <= 5:
            yaxis_cfg.update({"tickformat": "d", "dtick": 1, "tick0": 0})

        fig.update_layout(xaxis={"title": "Day of Week"}, yaxis=yaxis_cfg)
        return apply_dark_layout(fig, title="Demand by Day of Week", height=320)
    except Exception as e:
        return build_empty_chart("Day of Week Demand", f"Error: {str(e)}")


def build_fare_distribution_chart(df: pd.DataFrame, fare_col: str = "fare_amount") -> go.Figure:
    """Histogram showing true fare distribution."""
    if df.empty or fare_col not in df.columns:
        if df.empty:
            return build_empty_chart("Fare Distribution", "Zero records match active filter criteria. Try resetting filters.", header="🔍 No Matching Records")
        return build_empty_chart("Fare Distribution", "No fare data available", required_contract=f"'{fare_col}' (numeric)")

    try:
        fares = pd.to_numeric(df[fare_col], errors="coerce").dropna()
        fares = fares[fares > 0]
        if fares.empty:
            return build_empty_chart("Fare Distribution", "Zero valid positive fare records found", header="🔍 No Matching Records")

        # Capped for chart visibility at 99th percentile if extreme
        cap = float(fares.quantile(0.99))
        filtered = fares[fares <= cap]

        nbins = 40 if len(filtered) > 10 else max(len(filtered), 5)

        fig = go.Figure(
            go.Histogram(
                x=filtered,
                nbinsx=nbins,
                marker={"color": THEME_CYAN, "line": {"color": THEME_BG_DARK, "width": 0.8}},
                hovertemplate="<b>Fare Bin:</b> $%{x:.2f}<br><b>Trips:</b> %{y:,}<extra></extra>"
            )
        )
        yaxis_cfg = {"title": "Frequency", "rangemode": "tozero", "tick0": 0}
        if len(filtered) <= 5:
            yaxis_cfg.update({"tickformat": "d", "dtick": 1, "tick0": 0})

        fig.update_layout(xaxis={"title": "Fare Amount (USD $)", "rangemode": "tozero", "tick0": 0}, yaxis=yaxis_cfg)
        return apply_dark_layout(fig, title=f"Fare Distribution (Up to 99th Pct: ${cap:.2f})", height=320)
    except Exception as e:
        return build_empty_chart("Fare Distribution", f"Error: {str(e)}")


def build_fare_vs_distance_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot illustrating empirical distance vs fare correlation with complete default range."""
    if df.empty or "fare_amount" not in df.columns or "trip_distance" not in df.columns:
        if df.empty:
            return build_empty_chart("Fare vs Trip Distance", "Zero records match active filter criteria. Try resetting filters.", header="🔍 No Matching Records")
        return build_empty_chart("Fare vs Trip Distance", "Requires fare and distance columns", required_contract="'fare_amount', 'trip_distance'")

    try:
        sample_df = df[["fare_amount", "trip_distance"]].dropna()
        sample_df["fare_amount"] = pd.to_numeric(sample_df["fare_amount"], errors="coerce")
        sample_df["trip_distance"] = pd.to_numeric(sample_df["trip_distance"], errors="coerce")
        sample_df = sample_df[(sample_df["fare_amount"] > 0) & (sample_df["trip_distance"] > 0)]

        if sample_df.empty:
            return build_empty_chart("Fare vs Trip Distance", "Zero valid positive fare records found", header="🔍 No Matching Records")

        # Downsample to 1,500 points to keep UI fast and prevent browser lag
        if len(sample_df) > 1500:
            sample_df = sample_df.sample(n=1500, random_state=42)

        # Prominent discrete markers when only a few records exist
        marker_size = 11 if len(sample_df) <= 3 else (8 if len(sample_df) <= 10 else 5)
        marker_opacity = 0.9 if len(sample_df) <= 3 else (0.75 if len(sample_df) <= 10 else 0.5)

        fig = go.Figure(
            go.Scatter(
                x=sample_df["trip_distance"],
                y=sample_df["fare_amount"],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": THEME_CYAN,
                    "opacity": marker_opacity,
                    "line": {"color": THEME_AMBER, "width": 1 if len(sample_df) <= 3 else 0.5}
                },
                hovertemplate="<b>Distance:</b> %{x:.2f} mi<br><b>Fare:</b> $%{y:.2f}<extra></extra>"
            )
        )
        fig.update_layout(
            xaxis={"title": "Trip Distance (Miles)", "rangemode": "tozero", "tick0": 0, "autorange": True},
            yaxis={"title": "Fare Amount ($)", "rangemode": "tozero", "tick0": 0, "autorange": True}
        )
        return apply_dark_layout(fig, title="Fare vs Distance Scatter (Telemetry Sample)", height=320)
    except Exception as e:
        return build_empty_chart("Fare vs Trip Distance", f"Error: {str(e)}")


def build_fare_by_hour_chart(df: pd.DataFrame) -> go.Figure:
    """Average fare grouped by hour of day strictly bounded 0–23."""
    if df.empty or "fare_amount" not in df.columns:
        if df.empty:
            return build_empty_chart("Average Fare by Hour", "Zero records match active filter criteria. Try resetting filters.", header="🔍 No Matching Records")
        return build_empty_chart("Average Fare by Hour", "No fare data available", required_contract="'fare_amount'")

    try:
        temp = df.copy()
        if "hour_of_day" not in temp.columns:
            if "tpep_pickup_datetime" in temp.columns:
                temp["tpep_pickup_datetime"] = pd.to_datetime(temp["tpep_pickup_datetime"], errors="coerce")
                temp["hour_of_day"] = temp["tpep_pickup_datetime"].dt.hour
            else:
                return build_empty_chart("Average Fare by Hour", "Missing hour column")

        temp["fare_amount"] = pd.to_numeric(temp["fare_amount"], errors="coerce")
        temp = temp.dropna(subset=["hour_of_day", "fare_amount"])
        if temp.empty:
            return build_empty_chart("Average Fare by Hour", "No valid fare records found", header="🔍 No Matching Records")

        # Handle small samples gracefully (bar chart if <= 2 distinct hours)
        unique_hours = temp["hour_of_day"].nunique()
        if unique_hours <= 2:
            hourly_fare = temp.groupby("hour_of_day")["fare_amount"].mean().reset_index()
            fig = go.Figure(
                go.Bar(
                    x=hourly_fare["hour_of_day"],
                    y=hourly_fare["fare_amount"],
                    marker={"color": THEME_AMBER, "line": {"color": THEME_BORDER, "width": 1}},
                    width=0.35 if len(hourly_fare) == 1 else 0.45,
                    hovertemplate="<b>Hour %{x}:00:</b> Avg $%{y:.2f}<extra></extra>"
                )
            )
            fig.update_layout(
                xaxis={
                    "title": "Hour of Day (0–23)",
                    "tickmode": "array",
                    "tickvals": [int(h) for h in hourly_fare["hour_of_day"]],
                    "ticktext": [f"{int(h):02d}:00" for h in hourly_fare["hour_of_day"]],
                    "range": [-0.5, 23.5]
                },
                yaxis={"title": "Mean Fare (USD $)", "rangemode": "tozero", "tick0": 0}
            )
        else:
            hourly_fare = temp.groupby("hour_of_day")["fare_amount"].mean().reindex(range(24), fill_value=0).reset_index()
            fig = go.Figure(
                go.Scatter(
                    x=hourly_fare["hour_of_day"],
                    y=hourly_fare["fare_amount"],
                    mode="lines+markers",
                    line={"color": THEME_AMBER, "width": 2.5},
                    marker={"size": 6, "color": THEME_AMBER},
                    hovertemplate="<b>Hour %{x}:00:</b> Avg $%{y:.2f}<extra></extra>"
                )
            )
            fig.update_layout(
                xaxis={
                    "title": "Hour of Day (0–23)",
                    "tickmode": "array",
                    "tickvals": list(range(0, 24, 2)),
                    "ticktext": [f"{h:02d}:00" for h in range(0, 24, 2)],
                    "range": [-0.5, 23.5]
                },
                yaxis={"title": "Mean Fare (USD $)", "rangemode": "tozero", "tick0": 0}
            )
        return apply_dark_layout(fig, title="Average Fare by Hour of Day", height=320)
    except Exception as e:
        return build_empty_chart("Average Fare by Hour", f"Error: {str(e)}")


def build_borough_volume_chart(df: pd.DataFrame, col: str = "PU_borough", title: str = "Pickups by Borough") -> go.Figure:
    """Bar chart of pickup or dropoff counts by borough, strictly beginning at zero."""
    if df.empty or col not in df.columns:
        if df.empty:
            return build_empty_chart(title, "Zero records match active filter criteria. Try resetting filters.", header="🔍 No Matching Records")
        return build_empty_chart(title, f"Missing '{col}' geographic column", required_contract=f"'{col}'")

    try:
        counts = df[col].dropna().value_counts().reset_index()
        counts.columns = ["Borough", "Count"]

        bar_width = 0.35 if len(counts) == 1 else (0.45 if len(counts) == 2 else None)

        fig = go.Figure(
            go.Bar(
                x=counts["Borough"],
                y=counts["Count"],
                width=bar_width,
                marker={"color": THEME_CYAN, "line": {"color": THEME_BORDER, "width": 1}},
                hovertemplate="<b>%{x}:</b> %{y:,} trips<extra></extra>"
            )
        )
        yaxis_cfg = {"title": "Trips", "rangemode": "tozero", "tick0": 0}
        if counts["Count"].max() <= 5:
            yaxis_cfg.update({"tickformat": "d", "dtick": 1, "tick0": 0})

        fig.update_layout(xaxis={"title": "Borough"}, yaxis=yaxis_cfg)
        return apply_dark_layout(fig, title=title, height=320)
    except Exception as e:
        return build_empty_chart(title, f"Error: {str(e)}")


def build_top_corridors_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing the highest-frequency mobility corridors visually starting at zero."""
    pu_col = "PU_borough" if "PU_borough" in df.columns else ("PULocationID" if "PULocationID" in df.columns else None)
    do_col = "DO_borough" if "DO_borough" in df.columns else ("DOLocationID" if "DOLocationID" in df.columns else None)

    if df.empty or not pu_col or not do_col:
        if df.empty:
            return build_empty_chart("Top Mobility Corridors", "Zero records match active filter criteria. Try resetting filters.", header="🔍 No Matching Records")
        return build_empty_chart("Top Mobility Corridors", "Requires pickup and dropoff location columns", required_contract="'PU_borough' / 'PULocationID', 'DO_borough' / 'DOLocationID'")

    try:
        valid_df = df.dropna(subset=[pu_col, do_col])
        valid_df = valid_df[
            ~valid_df[pu_col].astype(str).str.lower().isin(["nan", "none", "unknown", ""]) &
            ~valid_df[do_col].astype(str).str.lower().isin(["nan", "none", "unknown", ""])
        ]
        if valid_df.empty:
            return build_empty_chart("Top Mobility Corridors", "Zero valid origin-destination pairs found in records", header="🔍 No Matching Records")

        corridors = (valid_df[pu_col].astype(str) + " → " + valid_df[do_col].astype(str)).value_counts().head(8).reset_index()
        corridors.columns = ["Corridor", "Trips"]
        corridors = corridors.iloc[::-1]  # Ascending for clean horizontal bar

        bar_width = 0.35 if len(corridors) == 1 else (0.45 if len(corridors) == 2 else None)

        fig = go.Figure(
            go.Bar(
                x=corridors["Trips"],
                y=corridors["Corridor"],
                orientation="h",
                width=bar_width,
                marker={"color": THEME_PURPLE, "line": {"color": THEME_BORDER, "width": 1}},
                hovertemplate="<b>%{y}:</b> %{x:,} trips<extra></extra>"
            )
        )
        xaxis_cfg = {"title": "Trip Volume", "rangemode": "tozero", "tick0": 0}
        if corridors["Trips"].max() <= 5:
            xaxis_cfg.update({"tickformat": "d", "dtick": 1, "tick0": 0})

        fig.update_layout(xaxis=xaxis_cfg, yaxis={"title": ""})
        # Generous left margin (150px) to prevent long corridor names from clipping
        return apply_dark_layout(fig, title="Top Mobility Corridors (Origin → Destination)", height=320, margin={"l": 150, "r": 25, "t": 50, "b": 45})
    except Exception as e:
        return build_empty_chart("Top Mobility Corridors", f"Error: {str(e)}")


# ==============================================
# Data Mining Insights Section Component Builders
# ==============================================

def build_data_mining_status_card(
    title: str,
    stage_name: str,
    status_label: str,
    status_color: str,
    description: str,
    required_artifact: str,
    expected_schema: str
) -> dbc.Card:
    """Builds a standardized, high-integrity card describing mining track status."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span(stage_name, className="drift-eyebrow"),
                        dbc.Badge(status_label, color=status_color, className="drift-badge-status ms-auto")
                    ],
                    className="d-flex align-items-center mb-2"
                ),
                html.H5(title, className="text-white mb-2"),
                html.P(description, className="drift-muted small mb-3"),
                html.Div(
                    [
                        html.Strong("Expected Teammate Artifact:", className="text-info small d-block mb-1"),
                        html.Code(required_artifact, className="drift-code-snippet d-block mb-2"),
                        html.Strong("Input Schema Contract:", className="text-secondary small d-block mb-1"),
                        html.Code(expected_schema, className="drift-code-snippet d-block")
                    ],
                    className="drift-contract-box p-2 rounded"
                )
            ]
        ),
        className="drift-card h-100 mb-3"
    )
