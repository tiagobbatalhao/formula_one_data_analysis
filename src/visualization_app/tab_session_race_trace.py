import plotly.graph_objects as go
import sklearn.linear_model
import streamlit as st
from data_loader import load_session_laps, load_session_results
from tab_utils import choose_driver_colors


def tab_session_race_trace(session_info):
    if session_info["session_name"] not in ["Race", "Sprint"]:
        return
    selected_year = session_info["year"]
    selected_session_id = session_info["session_id"]

    df_results = load_session_results(selected_year)
    df_results = df_results[df_results["session_id"] == selected_session_id]
    df_results = df_results.sort_values(by=["position", "driver_number"])
    df_results = choose_driver_colors(df_results)

    df = load_session_laps(selected_year)
    df = df[df["session_id"] == selected_session_id]
    df = df.merge(
        df_results[
            ["driver_number", "team_color", "driver_color", "driver_index_in_team"]
        ],
        on=["driver_number"],
    )
    to_fit = df["timing_end_sector3"].notna() & df["lap_number"].notna()
    df = df[to_fit]
    regressor = sklearn.linear_model.QuantileRegressor(quantile=0.2).fit(
        df.loc[:, ["lap_number"]], df.loc[:, "timing_end_sector3"]
    )
    df["expected"] = regressor.predict(df[["lap_number"]])
    st.dataframe(df[["driver_number", "lap_number", "timing_end_sector3", "expected"]])

    plot_data = []
    for driver in sorted(df["driver_number"].unique()):
        this = df[df["driver_number"] == driver]
        if len(this) == 0:
            continue
        color = "#" + this["team_color"].iloc[0]
        symbol_type = (
            "circle" if this["driver_index_in_team"].iloc[0] == 0 else "circle-open"
        )
        dash_type = None if this["driver_index_in_team"].iloc[0] == 0 else "dash"
        plot_data += [
            go.Scatter(
                x=this["lap_number"],
                y=(this["timing_end_sector3"] - this["expected"]) * (-1),
                mode="lines+markers",
                name=this.iloc[0]["driver_name"],
                marker=dict(color=color, symbol=symbol_type),
                line=dict(color=color, dash=dash_type),
            )
        ]

    plot_layout = dict(
        height=600,
        title="Race trace (delta to average)",
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)
