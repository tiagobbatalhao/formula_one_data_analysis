import math
from datetime import timedelta

import plotly.graph_objects as go
import streamlit as st
from data_loader import load_session_laps, load_session_results
from tab_utils import choose_driver_colors


def tab_session_laps(session_info):
    selected_year = session_info["year"]
    selected_session_id = session_info["session_id"]

    df_results = load_session_results(selected_year)
    df_results = df_results[df_results["session_id"] == selected_session_id]
    df_results = df_results.sort_values(by=["position", "driver_number"])
    df_results = choose_driver_colors(df_results)

    df = load_session_laps(selected_year)
    df = df.merge(
        df_results[
            ["driver_number", "team_color", "driver_color", "driver_index_in_team"]
        ],
        on=["driver_number"],
    )
    df = df[df["session_id"] == selected_session_id]
    keep_laps = (
        df["is_accurate"] & df["timestamp_lap_start"].notna() & df["time_lap"].notna()
    )
    df.loc[keep_laps, "timestamp_lap_end"] = df.loc[keep_laps].apply(
        lambda row: row["timestamp_lap_start"] + timedelta(seconds=row["time_lap"]),
        axis=1,
    )
    df.loc[~keep_laps, "timestamp_lap_end"] = None
    df.loc[~keep_laps, "time_lap"] = None
    df = df.sort_values(by=["driver_number", "lap_number"])

    min_lap_time = df["time_lap"].dropna().min()

    plot_data = []
    for driver in sorted(df["driver_number"].unique()):
        this = df[df["driver_number"] == driver]
        if len(this) == 0:
            continue
        color = "#" + this["team_color"].iloc[0]
        dash_type = None if this["driver_index_in_team"].iloc[0] == 0 else "dash"
        symbol_type = (
            "circle" if this["driver_index_in_team"].iloc[0] == 0 else "circle-open"
        )
        plot_data += [
            go.Scatter(
                x=this["timestamp_lap_end"],
                y=this["time_lap"],
                mode="lines+markers",
                name=this.iloc[0]["driver_name"],
                marker=dict(color=color, symbol=symbol_type),
                line=dict(color=color, dash=dash_type),
            )
        ]

    plot_layout = dict(
        height=600,
        title="Laptime history",
        yaxis=dict(
            range=(
                math.floor(min_lap_time * 0.99 / 0.1) * 0.1,
                math.ceil(min_lap_time * 1.11 / 0.1) * 0.1,
            )
        ),
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)
