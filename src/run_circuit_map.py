import argparse

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.linear_model import LinearRegression

import data_engineering.gold_layer as gold_layer


class FourierFit:
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def _get_basis_function(self, val):
        twopi = 2 * np.pi
        max_degree = self.max_degree
        arr = np.zeros(1 + 2 * max_degree)
        arr[0] = 1.0
        arr[1::2] = [np.cos(val * k * twopi) for k in range(1, 1 + max_degree)]
        arr[2::2] = [np.sin(val * k * twopi) for k in range(1, 1 + max_degree)]
        return arr

    def _get_basis_d1(self, val):
        twopi = 2 * np.pi
        max_degree = self.max_degree
        arr = np.zeros(1 + 2 * max_degree)
        arr[1::2] = [
            -1 * k * twopi * np.sin(val * k * twopi) for k in range(1, 1 + max_degree)
        ]
        arr[2::2] = [
            +1 * k * twopi * np.cos(val * k * twopi) for k in range(1, 1 + max_degree)
        ]
        return arr

    def _get_basis_d2(self, val):
        twopi = 2 * np.pi
        max_degree = self.max_degree
        arr = np.zeros(1 + 2 * max_degree)
        arr[1::2] = [
            -1 * (k * twopi) ** 2 * np.cos(val * k * twopi)
            for k in range(1, 1 + max_degree)
        ]
        arr[2::2] = [
            -1 * (k * twopi) ** 2 * np.sin(val * k * twopi)
            for k in range(1, 1 + max_degree)
        ]
        return arr

    def fit(self, X, y):
        features = np.concat(
            [
                self._get_basis_function(x).reshape(1, -1)
                for x in np.array(X).reshape(-1)
            ],
            axis=0,
        )
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(features, np.array(y).reshape(-1))
        return self

    def predict(self, X):
        features = np.concat(
            [
                self._get_basis_function(x).reshape(1, -1)
                for x in np.array(X).reshape(-1)
            ],
            axis=0,
        )
        pred = self.model.predict(features)
        return pred

    def predict_d1(self, X):
        features = np.concat(
            [self._get_basis_d1(x).reshape(1, -1) for x in np.array(X).reshape(-1)],
            axis=0,
        )
        pred = self.model.predict(features)
        return pred

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    @property
    def _coefficients(self):
        return self.model.coef_.round(12)


def fit_map_by_time(df_position, max_degree):
    encoding_by_time = df_position["timing_from_lap"] / df_position["time_lap"]
    fitting_by_time = {}
    fitting_by_time["x"] = FourierFit(max_degree).fit(
        encoding_by_time, df_position["coordinate_x"]
    )
    fitting_by_time["y"] = FourierFit(max_degree).fit(
        encoding_by_time, df_position["coordinate_y"]
    )
    fitting_by_time["z"] = FourierFit(max_degree).fit(
        encoding_by_time, df_position["coordinate_z"]
    )
    return fitting_by_time


def correct_fit_by_distance(fitting_by_time, max_degree, predict_size):
    predict_parameter = np.linspace(0, 1, int(predict_size) + 1)
    data_by_time = np.concatenate(
        [
            fitting_by_time[k].predict(predict_parameter).reshape(-1, 1)
            for k in ["x", "y", "z"]
        ],
        axis=1,
    )
    distance = np.sqrt((np.diff(data_by_time[:, :2], axis=0) ** 2).sum(axis=1))
    encoding_by_distance = np.insert(distance.cumsum() / distance.sum(), 0, 0)
    fitting_by_distance = {}
    fitting_by_distance["x"] = FourierFit(max_degree).fit(
        encoding_by_distance, data_by_time[:, 0]
    )
    fitting_by_distance["y"] = FourierFit(max_degree).fit(
        encoding_by_distance, data_by_time[:, 1]
    )
    fitting_by_distance["z"] = FourierFit(max_degree).fit(
        encoding_by_distance, data_by_time[:, 2]
    )
    return fitting_by_distance


def adjust_starting_point(fitting, df_pos):
    index_lap = ["session_id", "driver_number", "lap_number"]
    df = df_pos.sort_values(by=index_lap + ["timing_from_lap"])
    first_point_in_lap = df.drop_duplicates(subset=index_lap, keep="first")[
        ["coordinate_x", "coordinate_y"]
    ].median()
    last_point_in_lap = df.drop_duplicates(subset=index_lap, keep="last")[
        ["coordinate_x", "coordinate_y"]
    ].median()
    start_point = (first_point_in_lap + last_point_in_lap) / 2
    start_point = start_point.values.reshape(-1)

    def error_function(angle):
        pred_d0 = np.array([fitting[k].predict(angle) for k in ["x", "y"]])
        pred_d1 = np.array([fitting[k].predict_d1(angle) for k in ["x", "y"]])
        diff = pred_d0.reshape(-1) - start_point
        fun = diff * diff
        deriv = 2 * diff * pred_d1.reshape(-1)
        return float(fun.sum()), float(deriv.sum())

    start_encoding = scipy.optimize.minimize(error_function, 0.0, jac=True)
    start_encoding = start_encoding.x[0]
    while start_encoding >= 0.5:
        start_encoding -= 1
    while start_encoding < -0.5:
        start_encoding += 1
    return start_encoding


def format_output(fitting, predict_size, adjustment):
    predict_parameter = np.linspace(0, 1, int(predict_size) + 1)
    call_parameter = predict_parameter + adjustment
    output_f = np.concatenate(
        [fitting[k].predict(call_parameter).reshape(-1, 1) for k in ["x", "y", "z"]],
        axis=1,
    )
    output_der = np.concatenate(
        [fitting[k].predict_d1(call_parameter).reshape(-1, 1) for k in ["x", "y", "z"]],
        axis=1,
    )
    save = pd.DataFrame(
        np.concatenate(
            [predict_parameter.reshape(-1, 1), output_f, output_der], axis=1
        )[:-1]
    )
    save.columns = [
        "encoding",
        "coordinate_x",
        "coordinate_y",
        "coordinate_z",
        "derivative_x",
        "derivative_y",
        "derivative_z",
    ]
    save = save.sort_values(by=["encoding"])

    diff_coordinate_x = np.diff(
        np.append(save["coordinate_x"], save["coordinate_x"][0])
    )
    diff_coordinate_y = np.diff(
        np.append(save["coordinate_y"], save["coordinate_y"][0])
    )
    diffXY = np.sqrt(diff_coordinate_x**2 + diff_coordinate_y**2)
    save["total_distance_m"] = float(diffXY.sum()) / 10.0
    save["distance_m"] = np.insert(diffXY.cumsum()[:-1], 0, 0) / 10.0
    return save


def find_circuit_map(df_position, max_degree, predict_size):
    fitting_by_time = fit_map_by_time(df_position, max_degree)
    fitting_by_distance = correct_fit_by_distance(
        fitting_by_time, max_degree, predict_size
    )
    start_encoding = adjust_starting_point(fitting_by_distance, df_position)
    save = format_output(fitting_by_distance, predict_size, start_encoding)

    return dict(
        fitting=fitting_by_distance,
        data=save,
        adjustment=start_encoding,
    )


def main(session_ids, best_laps, max_degree, predict_size):
    df_laps_ls = []
    df_pos_ls = []
    for session_id in session_ids:
        year = int(session_id.strip()[1:5])
        round_id = session_id.strip()[5:8]
        df1 = gold_layer.SessionLaps(year).read()
        df1 = df1[df1["session_id"] == session_id]
        df2 = gold_layer.TelemetryPosData(year, round_id).read()
        df2 = df2[df2["session_id"] == session_id]
        df_laps_ls.append(df1)
        df_pos_ls.append(df2)
    df_laps = pd.concat(df_laps_ls)
    df_pos = pd.concat(df_pos_ls)
    df_laps_keep = (
        df_laps[(df_laps["is_accurate"])].sort_values(by=["time_lap"]).iloc[:best_laps]
    )
    idx = ["year", "session_id", "driver_number", "lap_number"]
    df_pos_keep = df_pos.merge(df_laps_keep[idx + ["time_lap"]], on=idx, how="inner")

    output = find_circuit_map(
        df_position=df_pos_keep, max_degree=max_degree, predict_size=predict_size
    )
    output["data"].to_parquet(f"circuit_map_{session_id}.parquet", index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id", type=str)
    parser.add_argument("--max_degree", type=int, default=100)
    parser.add_argument("--best_laps", type=int, default=200)
    args = parser.parse_args()
    main(
        session_ids=[x.strip() for x in str(args.session_id).split(",")],
        best_laps=int(args.best_laps),
        max_degree=int(args.max_degree),
        predict_size=int(1e5),
    )
