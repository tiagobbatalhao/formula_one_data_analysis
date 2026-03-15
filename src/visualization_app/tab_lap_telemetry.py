import data_loader
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.interpolate
import streamlit as st
from circuit_map_utils import create_circuit_map
from sklearn.neighbors import KDTree


@st.cache_data
def load_session_laps(year: int):
    return data_loader.load_session_laps(year=year)


@st.cache_data
def load_telemetry_car(year: int, round_id: str):
    return data_loader.load_telemetry_car(year=year, round_id=round_id)


@st.cache_data
def load_telemetry_pos(year: int, round_id: str):
    return data_loader.load_telemetry_pos(year=year, round_id=round_id)


@st.cache_data
def load_circuit_markers(year: int):
    return data_loader.load_circuit_markers(year=year)


@st.cache_data
def get_circuit_map(session_id: str):
    df_laps = load_session_laps(int(session_id[1:5]))
    df_laps = df_laps[df_laps["session_id"] == session_id]
    keep_laps = (
        df_laps[
            df_laps["is_accurate"]
            & df_laps["timing_pit_out"].isna()
            & df_laps["timing_pit_in"].isna()
            & (df_laps["time_lap"] <= 1.07 * df_laps["time_lap"].min())
        ]
        .sort_values(by=["time_lap"])
        .head(200)[["session_id", "driver_number", "lap_number"]]
    )
    df_position = load_telemetry_pos(int(session_id[1:5]), session_id[5:8])
    df_position = df_position[
        (df_position["session_id"] == session_id)
        & (df_position["coordinate_x"] != 0)
        & (df_position["coordinate_y"] != 0)
    ]
    circuit_map = create_circuit_map(df_position.merge(keep_laps), 50)
    circuit_map["idx"] = range(len(circuit_map))

    df_markers = load_circuit_markers(int(session_id[1:5]))
    df_markers = df_markers[df_markers["session_id"] == session_id]
    cols = ["coordinate_x", "coordinate_y"]
    _, encoding = KDTree(circuit_map[cols]).query(df_markers[cols], k=1)
    df_markers["idx"] = encoding

    def get_annotation_name(row):
        output = ""
        if pd.notna(row["number"]):
            output += str(row["number"])
        if pd.notna(row["letter"]):
            output += str(row["letter"])
        return output

    circuit_map = (
        circuit_map.assign(idx=lambda df: range(len(df)))
        .merge(
            df_markers[df_markers["annotation_type"] == "corner"][
                ["idx", "number", "letter"]
            ].assign(corner_id=lambda df: df.apply(get_annotation_name, axis=1))[
                ["idx", "corner_id"]
            ],
            on=["idx"],
            how="left",
        )
        .merge(
            df_markers[df_markers["annotation_type"] == "marshal_lights"][
                ["idx", "number", "letter"]
            ].assign(marshal_light_id=lambda df: df.apply(get_annotation_name, axis=1))[
                ["idx", "marshal_light_id"]
            ],
            on=["idx"],
            how="left",
        )
        .merge(
            df_markers[df_markers["annotation_type"] == "marshal_sectors"][
                ["idx", "number", "letter"]
            ].assign(
                marshal_sector_id=lambda df: df.apply(get_annotation_name, axis=1)
            )[
                ["idx", "marshal_sector_id"]
            ],
            on=["idx"],
            how="left",
        )
    )

    return circuit_map


def adjust_position(position_, total_distance):
    pos = [0] + list(position_)
    for i in range(1, len(pos)):
        while pos[i] - pos[i - 1] > 0.5 * total_distance:
            pos[i] -= total_distance
        while pos[i] - pos[i - 1] < -0.5 * total_distance:
            pos[i] += total_distance
    return np.array(pos)[1:]


def format_lap_time(lap_time):
    if np.isnan(lap_time):
        return "No time"
    minutes, rest = divmod(lap_time, 60)
    seconds, rest = divmod(rest, 1)
    milli = int(round(rest * 1000))
    return f"{int(minutes)}:{int(seconds):02d}:{int(milli):03d}"


def interpolate(target_x, reference_x, reference_y):
    diff = np.insert(np.diff(reference_x), 0, 0)
    keep = diff >= 0
    # return scipy.interpolate.CubicSpline(
    #     np.array(reference_x)[keep],
    #     np.array(reference_y)[keep],
    # )(target_x)
    return np.interp(target_x, np.array(reference_x)[keep], np.array(reference_y)[keep])


def process_data_lap(df_car, df_pos, circuit_map):
    columns = ["coordinate_x", "coordinate_y"]
    circuit_kdtree = KDTree(circuit_map[columns].astype(float))
    encoding = circuit_kdtree.query(df_pos[columns], k=1)

    df_pos["idx"] = encoding[1]
    df_pos = df_pos.merge(
        circuit_map[["idx", "absolute_distance"] + columns],
        on=["idx"],
        suffixes=["", "_fit"],
    )

    total_distance = circuit_map["total_distance"].iloc[0]
    reference_time = df_pos["timing_from_lap"].values
    reference_position = df_pos["absolute_distance"].values
    reference_position = adjust_position(reference_position, total_distance)
    target_time = df_car["timing_from_lap"].values
    target_position = interpolate(target_time, reference_time, reference_position)
    df_car["absolute_distance"] = target_position

    return df_car, df_pos


def calc_time_difference(df_lap1, df_lap2, time_lap1, time_lap2, total_distance):
    x1 = [0] + list(df_lap1["absolute_distance"]) + [total_distance]
    y1 = [0] + list(df_lap1["timing_from_lap"]) + [time_lap1]
    x2 = [0] + list(df_lap2["absolute_distance"]) + [total_distance]
    y2 = [0] + list(df_lap2["timing_from_lap"]) + [time_lap2]
    distance = np.linspace(0, total_distance, 1001)
    timing_lap1 = interpolate(distance, x1, y1)
    timing_lap2 = interpolate(distance, x2, y2)
    df = pd.DataFrame([distance, timing_lap1, timing_lap2]).T
    df.columns = ["absolute_distance", "timing_lap1", "timing_lap2"]
    return df


def tab_lap_telemetry(session_info):
    selected_year = session_info["year"]
    selected_session_id = session_info["session_id"]

    df_laps = load_session_laps(selected_year)
    df_laps = df_laps[df_laps["session_id"] == selected_session_id]
    df_laps["drop_down_driver"] = df_laps.apply(
        lambda r: "{} - {}".format(
            str(r["driver_number"]).zfill(2),
            r["driver_name"],
        ),
        axis=1,
    )
    df_laps["drop_down_lap"] = df_laps.apply(
        lambda r: "Lap {} - {}".format(r["lap_number"], format_lap_time(r["time_lap"])),
        axis=1,
    )
    columns = st.columns(4)
    with columns[0]:
        selected_driver1 = st.selectbox(
            "Driver 1", sorted(df_laps["drop_down_driver"].unique())
        )
    with columns[1]:
        selected_lap1 = st.selectbox(
            "Lap 1",
            df_laps[df_laps["drop_down_driver"] == selected_driver1]["drop_down_lap"],
        )
    with columns[2]:
        selected_driver2 = st.selectbox(
            "Driver 2", sorted(df_laps["drop_down_driver"].unique())
        )
    with columns[3]:
        selected_lap2 = st.selectbox(
            "Lap 2",
            df_laps[df_laps["drop_down_driver"] == selected_driver2]["drop_down_lap"],
        )

    lap_info1 = (
        df_laps[
            (df_laps["drop_down_driver"] == selected_driver1)
            & (df_laps["drop_down_lap"] == selected_lap1)
        ]
        .iloc[0]
        .to_dict()
    )
    lap_info2 = (
        df_laps[
            (df_laps["drop_down_driver"] == selected_driver2)
            & (df_laps["drop_down_lap"] == selected_lap2)
        ]
        .iloc[0]
        .to_dict()
    )

    df_lap_pos = load_telemetry_pos(
        year=int(selected_year), round_id=selected_session_id[5:8]
    )
    df_lap_pos1 = df_lap_pos[
        (df_lap_pos["session_id"] == lap_info1["session_id"])
        & (df_lap_pos["driver_number"] == lap_info1["driver_number"])
        & (df_lap_pos["lap_number"] == lap_info1["lap_number"])
    ]
    df_lap_pos2 = df_lap_pos[
        (df_lap_pos["session_id"] == lap_info2["session_id"])
        & (df_lap_pos["driver_number"] == lap_info2["driver_number"])
        & (df_lap_pos["lap_number"] == lap_info2["lap_number"])
    ]

    df_lap_car = load_telemetry_car(
        year=int(selected_year), round_id=selected_session_id[5:8]
    )
    df_lap_car1 = df_lap_car[
        (df_lap_car["session_id"] == lap_info1["session_id"])
        & (df_lap_car["driver_number"] == lap_info1["driver_number"])
        & (df_lap_car["lap_number"] == lap_info1["lap_number"])
    ]
    df_lap_car2 = df_lap_car[
        (df_lap_car["session_id"] == lap_info2["session_id"])
        & (df_lap_car["driver_number"] == lap_info2["driver_number"])
        & (df_lap_car["lap_number"] == lap_info2["lap_number"])
    ]

    circuit_map = get_circuit_map(session_id=selected_session_id)

    df_car1, df_pos1 = process_data_lap(df_lap_car1, df_lap_pos1, circuit_map)
    df_car2, df_pos2 = process_data_lap(df_lap_car2, df_lap_pos2, circuit_map)

    xaxis_ticks = [
        (r["absolute_distance"] / 10.0, "T" + str(r["corner_id"]))
        for _, r in circuit_map.dropna(subset=["corner_id"]).iterrows()
    ]

    df_timing_difference = calc_time_difference(
        df_lap1=df_pos1,
        df_lap2=df_pos2,
        time_lap1=lap_info1["time_lap"],
        time_lap2=lap_info2["time_lap"],
        total_distance=circuit_map["total_distance"].iloc[0],
    )

    plot_data = [
        go.Scatter(
            x=df_timing_difference["absolute_distance"] / 10.0,
            y=df_timing_difference["timing_lap1"] - df_timing_difference["timing_lap2"],
            mode="lines+markers",
            name="Timing difference",
        ),
        # go.Scatter(
        #     x=df_timing_difference["absolute_distance"] / 10.0,
        #     y=(
        #         df_timing_difference["timing_lap1"]
        #         -df_timing_difference["timing_lap2"]
        #     ).rolling(10).mean(),
        #     mode="lines+markers",
        #     name="Timing difference (averaged)"
        # ),
    ]
    plot_layout = dict(
        title="Timing",
        xaxis=dict(
            tickvals=[x[0] for x in xaxis_ticks], ticktext=[x[1] for x in xaxis_ticks]
        ),
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    plot_data = [
        go.Scatter(
            x=df_timing_difference["absolute_distance"] / 10.0,
            y=df_timing_difference[col],
            mode="lines+markers",
            name=name,
        )
        for col, name in zip(["timing_lap1", "timing_lap2"], ["Lap 1", "Lap2"])
    ]
    plot_layout = dict(
        title="Timing",
        xaxis=dict(
            tickvals=[x[0] for x in xaxis_ticks], ticktext=[x[1] for x in xaxis_ticks]
        ),
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    plot_data = [
        go.Scatter(
            x=df["absolute_distance"] / 10.0,
            y=df["throttle"],
            mode="lines+markers",
            name=name,
        )
        for df, name in zip([df_car1, df_car2], ["Lap 1", "Lap2"])
    ]
    plot_layout = dict(
        title="Throttle",
        xaxis=dict(
            tickvals=[x[0] for x in xaxis_ticks], ticktext=[x[1] for x in xaxis_ticks]
        ),
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    plot_data = [
        go.Scatter(
            x=df["absolute_distance"] / 10.0,
            y=df["brake"] * 100,
            mode="lines+markers",
            name=name,
        )
        for df, name in zip([df_car1, df_car2], ["Lap 1", "Lap2"])
    ]
    plot_layout = dict(
        title="Brakes",
        xaxis=dict(
            tickvals=[x[0] for x in xaxis_ticks], ticktext=[x[1] for x in xaxis_ticks]
        ),
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    plot_data = [
        go.Scatter(
            x=df["absolute_distance"] / 10.0,
            y=df["speed"],
            mode="lines+markers",
            name=name,
        )
        for df, name in zip([df_car1, df_car2], ["Lap 1", "Lap2"])
    ]
    plot_layout = dict(
        title="Speed",
        xaxis=dict(
            tickvals=[x[0] for x in xaxis_ticks], ticktext=[x[1] for x in xaxis_ticks]
        ),
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    plot_data = [
        go.Scatter(
            x=df["absolute_distance"] / 10.0,
            y=df["ngear"],
            mode="lines+markers",
            name=name,
        )
        for df, name in zip([df_car1, df_car2], ["Lap 1", "Lap2"])
    ]
    plot_layout = dict(
        title="Gear",
        xaxis=dict(
            tickvals=[x[0] for x in xaxis_ticks], ticktext=[x[1] for x in xaxis_ticks]
        ),
        yaxis=dict(
            range=(0.5, 8.5),
            tickvals=list(range(1, 9)),
        ),
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    plot_data = [
        go.Scatter(
            x=df["absolute_distance"] / 10.0,
            y=df["rpm"],
            mode="lines+markers",
            name=name,
        )
        for df, name in zip([df_car1, df_car2], ["Lap 1", "Lap2"])
    ]
    plot_layout = dict(
        title="RPM",
        xaxis=dict(
            tickvals=[x[0] for x in xaxis_ticks], ticktext=[x[1] for x in xaxis_ticks]
        ),
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)
