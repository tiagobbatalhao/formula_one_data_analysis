import plotly.graph_objects as go
import streamlit as st
from data_loader import load_session_metadata
from tab_lap_telemetry import get_circuit_map, tab_lap_telemetry
from tab_session_laps import tab_session_laps
from tab_session_race_trace import tab_session_race_trace
from tab_session_result import tab_session_result

st.set_page_config(layout="wide")


def main():

    # Load historical sessions for selection
    st.sidebar.header("Select Session")
    years = list(range(1950, 2027))[::-1]
    selected_year = st.sidebar.selectbox("Year", years)
    sessions_for_year = load_session_metadata(year=selected_year)
    round_names = sessions_for_year.drop_duplicates(subset=["event_name"])[
        "event_name"
    ].tolist()
    selected_round = st.sidebar.selectbox("Round", round_names)
    event_info = sessions_for_year[sessions_for_year["event_name"] == selected_round]
    selected_session_name = st.sidebar.selectbox(
        "Session", event_info["session_name"].tolist()
    )
    session_info = (
        event_info[event_info["session_name"] == selected_session_name]
        .iloc[0]
        .to_dict()
    )

    st.title(
        "{} - {}".format(session_info["meeting_name"], session_info["session_name"])
    )

    tabs = st.tabs(
        ["Result", "Lap history", "Race trace", "Lap telemetry", "Circuit map"]
    )

    with tabs[0]:
        tab_session_result(session_info)
    with tabs[1]:
        tab_session_laps(session_info)
    with tabs[2]:
        tab_session_race_trace(session_info)
    with tabs[3]:
        tab_lap_telemetry(session_info)
    with tabs[4]:
        df = get_circuit_map(session_info["session_id"])
        spanX = (df["coordinate_x"].min(), df["coordinate_x"].max())
        spanY = (df["coordinate_y"].min(), df["coordinate_y"].max())
        centerX = (spanX[1] + spanX[0]) / 2
        rangeX = spanX[1] - spanX[0]
        centerY = (spanY[1] + spanY[0]) / 2
        plot_data = [
            go.Scatter(
                x=(df["coordinate_x"] - centerX) / 10.0,
                y=(df["coordinate_y"] - centerY) / 10.0,
                text=(df["absolute_distance"] / 10).apply(
                    lambda x: f"Distance {x:.1f} m"
                ),
                hoverinfo="text",
                mode="markers",
                marker=dict(size=2),
            )
        ]

        rangeY = spanY[1] - spanY[0]
        range_ = max(rangeX, rangeY)
        spanXY = [-0.55 * range_, 0.55 * range_]
        spanXY = [x / 10.0 for x in spanXY]
        plot_layout = dict(
            title="Circuit Map",
            xaxis=dict(range=spanXY, tickvals=[], showgrid=False, zeroline=False),
            yaxis=dict(range=spanXY, tickvals=[], showgrid=False, zeroline=False),
            width=1200,
            height=1200,
            showlegend=False,
        )
        fig = go.Figure(data=plot_data, layout=plot_layout)
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
