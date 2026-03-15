import pandas as pd
import streamlit as st
from data_loader import load_session_results


def tab_session_result(session_info):
    selected_year = session_info["year"]
    selected_session_id = session_info["session_id"]
    df = load_session_results(selected_year)
    df = df[df["session_id"] == selected_session_id]
    df = df.sort_values(by=["position", "driver_number"]).reset_index(drop=True)
    columns = []
    columns.append(
        ("position", st.column_config.NumberColumn(label="Position", width=None))
    )
    columns.append(
        ("driver_number", st.column_config.NumberColumn(label="Number", width=None))
    )
    columns.append(
        (
            "driver_broadcast_name",
            st.column_config.TextColumn(label="Driver", width=None),
        )
    )
    columns.append(("team_name", st.column_config.TextColumn(label="Team", width=None)))
    if session_info["session_name"] in ["Qualifying", "Sprint Qualifying"]:
        lap_times = [
            x
            for x in df[["time_q1", "time_q2", "time_q3"]].values.flatten()
            if pd.notna(x)
        ]
        bounds = dict(min_value=min(lap_times), max_value=max(lap_times))

        columns.append(
            (
                "time_q1",
                st.column_config.ProgressColumn(
                    label="Time (Q1)", width=None, format="%.3f", **bounds
                ),
            )
        )
        columns.append(
            (
                "time_q2",
                st.column_config.ProgressColumn(
                    label="Time (Q2)", width=None, format="%.3f", **bounds
                ),
            )
        )
        columns.append(
            (
                "time_q3",
                st.column_config.ProgressColumn(
                    label="Time (Q3)", width=None, format="%.3f", **bounds
                ),
            )
        )
    if session_info["session_name"] in ["Race", "Sprint"]:
        columns.append(
            ("status", st.column_config.TextColumn(label="Status", width=None))
        )
        columns.append(
            (
                "points",
                st.column_config.ProgressColumn(
                    label="Points", width=None, min_value=0, max_value=25, format="%d"
                ),
            )
        )

    st.dataframe(
        df,
        column_config={x[0]: x[1] for x in columns},
        column_order=[x[0] for x in columns],
        hide_index=True,
    )
    st.dataframe(
        df,
    )
