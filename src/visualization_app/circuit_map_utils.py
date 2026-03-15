import numpy as np
import pandas as pd
from scipy.spatial import KDTree


class FourierFitting:
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def _create_matrix_independent_variables(self, X):
        arrx = np.array(X).reshape(-1, 1)
        matrix = []
        matrix.append(np.ones_like(arrx))
        for deg in range(1, 1 + self.max_degree):
            matrix.append(np.sin(arrx * deg * 2 * np.pi))
            matrix.append(np.cos(arrx * deg * 2 * np.pi))
        return np.concatenate(matrix, axis=1)

    def fit(self, X, y):
        independent_vars = self._create_matrix_independent_variables(X)
        independent_inv = np.linalg.pinv(independent_vars)
        self.coef_ = np.dot(independent_inv, y)
        return self

    def predict(self, X):
        independent_vars = self._create_matrix_independent_variables(X)
        return np.dot(independent_vars, self.coef_)


def create_circuit_map(df_, max_degree):
    bounds = df_.groupby(["driver_number", "lap_number"]).agg(
        {"timing_from_lap": "max"}
    )
    bounds.columns = ["max_time"]
    df_fit1 = df_.merge(
        bounds.reset_index(), on=bounds.index.names, how="inner", suffixes=["", "_DROP"]
    )

    fit_time = df_fit1["timing_from_lap"] / df_fit1["max_time"]
    fitting_part1 = {
        c: FourierFitting(max_degree).fit(fit_time, df_fit1[f"coordinate_{c}"])
        for c in ["x", "y", "z"]
    }

    df_fit2 = pd.DataFrame()
    df_fit2["time"] = np.linspace(0, 1, 10001)
    for coord, fit_object in fitting_part1.items():
        df_fit2[f"coordinate_{coord}"] = fit_object.predict(df_fit2["time"])
        df_fit2[f"previous_{coord}"] = df_fit2[f"coordinate_{coord}"].shift(1)
    df_fit2["distance"] = np.sqrt(
        (df_fit2["coordinate_x"] - df_fit2["previous_x"]) ** 2
        + (df_fit2["coordinate_y"] - df_fit2["previous_y"]) ** 2
    )
    df_fit2["distance_cumulative"] = df_fit2["distance"].cumsum()

    total_distance = df_fit2["distance"].sum()
    fit_distance = df_fit2["distance_cumulative"].fillna(0) / total_distance
    fitting_part2 = {
        c: FourierFitting(max_degree).fit(fit_distance, df_fit2[f"coordinate_{c}"])
        for c in ["x", "y", "z"]
    }

    df_output = pd.DataFrame()
    df_output["relative_distance"] = np.linspace(0, 1, 10001)
    for coord, fit_object in fitting_part2.items():
        df_output[f"coordinate_{coord}"] = fit_object.predict(
            df_output["relative_distance"]
        )
    df_output["absolute_distance"] = df_output["relative_distance"] * total_distance
    df_output["total_distance"] = total_distance

    return df_output


def apply_circuit_encoding(df_, df_circuit_map):
    tree_columns = ["coordinate_x", "coordinate_y"]
    info_columns = [
        "relative_distance",
        "absolute_distance",
        "coordinate_x",
        "coordinate_y",
        "coordinate_z",
    ]
    tree = KDTree(df_circuit_map[tree_columns])
    aux = tree.query(df_[tree_columns])
    for col in info_columns:
        df_[col + "_circuit"] = df_circuit_map.iloc[aux[1]][col].values
    return df_
