import argparse
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import data_engineering.gold_layer as gold_layer


def rotate(array, angle_deg):
    angle_rad = angle_deg * np.pi / 180
    cos_ = np.cos(angle_rad)
    sin_ = np.sin(angle_rad)
    matrix = np.array([[cos_, sin_], [-sin_, cos_]])
    return np.dot(array, matrix)


def main(session_id, rotation_angle=None):
    session_id = session_id.replace(",", "_")
    session_id_markers = session_id.split("_")[0]
    save_folder = pathlib.Path(__file__).resolve().parent.parent / "data" / "artifacts"
    df_circuit_map = pd.read_parquet(save_folder / f"circuit_map_{session_id}.parquet")
    df_markers = gold_layer.CircuitMarkers(int(session_id_markers[1:5])).read()
    df_markers = df_markers[df_markers["session_id"] == session_id_markers]

    if isinstance(rotation_angle, float):
        rotation_angle = rotation_angle
    elif rotation_angle is None:
        rotation_angle = 0.0
    elif isinstance(rotation_angle, bool) and rotation_angle:
        rotation_angle = df_markers["rotation"].iloc[0]

    plot_data = []
    cols_plot = ["coordinate_x", "coordinate_y"]

    this = df_circuit_map.iloc[::1]
    plot_ = rotate(this[cols_plot].values, rotation_angle) / 10.0
    plot_data += [
        go.Scatter(
            x=plot_[:, 0],
            y=plot_[:, 1],
            mode="markers",
            text=this["distance_m"],
            hoverinfo="text",
            marker=dict(
                size=2,
                colorscale="Inferno",
                cmin=0,
                cmax=1,
                color=this["encoding"],
            ),
            name="Racing Line",
        )
    ]

    this = df_circuit_map.iloc[:1]
    plot_ = rotate(this[cols_plot].values, rotation_angle) / 10.0
    plot_data += [
        go.Scatter(
            x=plot_[:, 0],
            y=plot_[:, 1],
            mode="markers",
            text=this["distance_m"],
            hoverinfo="text",
            marker=dict(size=8, color="blue"),
            name="Starting point",
        )
    ]

    this = df_markers[df_markers["annotation_type"] == "corner"]
    plot_ = rotate(this[cols_plot].values, rotation_angle) / 10.0
    plot_data += [
        go.Scatter(
            x=plot_[:, 0],
            y=plot_[:, 1],
            mode="markers",
            text=this.apply(
                lambda row: "T{}{}".format(row["number"], row["letter"] or ""), axis=1
            ),
            hoverinfo="text",
            marker=dict(size=8, color="black"),
            name="Corners",
        )
    ]

    this = df_markers[df_markers["annotation_type"] == "marshal_sectors"]
    plot_ = rotate(this[cols_plot].values, rotation_angle) / 10.0
    plot_data += [
        go.Scatter(
            x=plot_[:, 0],
            y=plot_[:, 1],
            mode="markers",
            text=this.apply(
                lambda row: "MS{}{}".format(row["number"], row["letter"] or ""), axis=1
            ),
            hoverinfo="text",
            marker=dict(size=8, color="green"),
            name="Marshal Sectors",
        )
    ]

    this = df_markers[df_markers["annotation_type"] == "marshal_lights"]
    plot_ = rotate(this[cols_plot].values, rotation_angle) / 10.0
    plot_data += [
        go.Scatter(
            x=plot_[:, 0],
            y=plot_[:, 1],
            mode="markers",
            text=this.apply(
                lambda row: "ML{}{}".format(row["number"], row["letter"] or ""), axis=1
            ),
            hoverinfo="text",
            marker=dict(size=8, color="red"),
            name="Marshal Lights",
        )
    ]

    plot_layout = dict(
        # width=1200, height=1200,
        plot_bgcolor="RGBA(0,0,0,0)",
        paper_bgcolor="RGBA(0,0,0,0)",
        xaxis=dict(tickvals=[]),
        yaxis=dict(tickvals=[]),
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    path_save = (save_folder / f"circuit_figure_{session_id}.html").as_posix()
    fig.write_html(path_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id", type=str)
    parser.add_argument("--rotation", type=float, default=None)
    parser.add_argument("--rotate", action="store_true")
    args = parser.parse_args()
    if args.rotate:
        main(session_id=args.session_id, rotation_angle=True)
    else:
        main(session_id=args.session_id, rotation_angle=args.rotation)
