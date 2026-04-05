import argparse
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import data_engineering.gold_layer as gold_layer

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("formula_one_circuit_map")

DATA_FOLDER = pathlib.Path(__file__).parent.resolve() / "data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)


class FourierFit:
    """A class for fitting and predicting using Fourier series.

    This class implements a Fourier series fit using a linear regression model.
    It supports up to a specified maximum degree of harmonics.

    Attributes:
        max_degree (int): Maximum degree of harmonics to use in the Fourier series
        model (LinearRegression): The fitted linear regression model
    """

    def __init__(self, max_degree: int, suffix: str = ""):
        """Initialize FourierFit with maximum degree.

        Args:
            max_degree (int): Maximum degree of harmonics to use

        Raises:
            ValueError: If max_degree is not positive
        """
        if max_degree <= 0:
            raise ValueError("max_degree must be positive")
        self.max_degree = max_degree
        self.model: Optional[LinearRegression] = None
        self._twopi = 2 * np.pi  # Cache frequently used value
        self.suffix = suffix

    def _get_basis_function(self, val: float) -> np.ndarray:
        """Calculate the basis functions for a given value.

        Args:
            val (float): Input value to calculate basis functions for

        Returns:
            np.ndarray: Array of basis function values
        """
        arr = np.zeros(1 + 2 * self.max_degree)
        arr[0] = 1.0
        arr[1::2] = [
            np.cos(val * k * self._twopi) for k in range(1, 1 + self.max_degree)
        ]
        arr[2::2] = [
            np.sin(val * k * self._twopi) for k in range(1, 1 + self.max_degree)
        ]
        return arr

    def _get_basis_d1(self, val: float) -> np.ndarray:
        """Calculate the first derivative of basis functions.

        Args:
            val (float): Input value to calculate derivatives for

        Returns:
            np.ndarray: Array of first derivative values
        """
        arr = np.zeros(1 + 2 * self.max_degree)
        arr[1::2] = [
            -1 * k * self._twopi * np.sin(val * k * self._twopi)
            for k in range(1, 1 + self.max_degree)
        ]
        arr[2::2] = [
            +1 * k * self._twopi * np.cos(val * k * self._twopi)
            for k in range(1, 1 + self.max_degree)
        ]
        return arr

    def _get_basis_d2(self, val: float) -> np.ndarray:
        """Calculate the second derivative of basis functions.

        Args:
            val (float): Input value to calculate derivatives for

        Returns:
            np.ndarray: Array of second derivative values
        """
        arr = np.zeros(1 + 2 * self.max_degree)
        arr[1::2] = [
            -1 * (k * self._twopi) ** 2 * np.cos(val * k * self._twopi)
            for k in range(1, 1 + self.max_degree)
        ]
        arr[2::2] = [
            -1 * (k * self._twopi) ** 2 * np.sin(val * k * self._twopi)
            for k in range(1, 1 + self.max_degree)
        ]
        return arr

    def fit(
        self, X: Union[np.ndarray, List[float]], y: Union[np.ndarray, List[float]]
    ) -> "FourierFit":
        """Fit the Fourier series to the data.

        Args:
            X: Input values
            y: Target values

        Returns:
            self: The fitted model
        """
        X = np.array(X).reshape(-1)
        y = np.array(y).reshape(-1)
        features = np.vstack([self._get_basis_function(x) for x in X])

        suffix = self.suffix
        tmp_file = DATA_FOLDER / f"input{suffix}.npy"
        np.save(tmp_file, X)
        mlflow.log_artifact(tmp_file.as_posix())
        tmp_file.unlink()
        tmp_file = DATA_FOLDER / f"target{suffix}.npy"
        np.save(tmp_file, y)
        mlflow.log_artifact(tmp_file.as_posix())
        tmp_file.unlink()
        tmp_file = DATA_FOLDER / f"features{suffix}.npy"
        np.save(tmp_file, features)
        mlflow.log_artifact(tmp_file.as_posix())
        tmp_file.unlink()

        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(features, y)

        model_info = mlflow.sklearn.log_model(
            sk_model=self.model, name=("fourier_coefficients" + suffix)
        )

        tmp_file = DATA_FOLDER / f"coefficients{suffix}.npy"
        np.save(tmp_file, self.model.coef_)
        mlflow.log_artifact(tmp_file.as_posix())
        tmp_file.unlink()

        pred = self.model.predict(features)
        mlflow.log_metric("mae" + suffix, mean_absolute_error(y, pred))
        mlflow.log_metric("rmse" + suffix, root_mean_squared_error(y, pred))

        return self

    def predict(self, X: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Predict values using the fitted model.

        Args:
            X: Input values to predict for

        Returns:
            np.ndarray: Predicted values

        Raises:
            ValueError: If model hasn't been fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X).reshape(-1)
        features = np.vstack([self._get_basis_function(x) for x in X])
        return self.model.predict(features)

    def predict_d1(self, X: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Predict first derivatives using the fitted model.

        Args:
            X: Input values to predict derivatives for

        Returns:
            np.ndarray: Predicted first derivatives

        Raises:
            ValueError: If model hasn't been fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X).reshape(-1)
        features = np.vstack([self._get_basis_d1(x) for x in X])
        return self.model.predict(features)

    def fit_predict(
        self, X: Union[np.ndarray, List[float]], y: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        """Fit the model and predict values in one step.

        Args:
            X: Input values
            y: Target values

        Returns:
            np.ndarray: Predicted values
        """
        return self.fit(X, y).predict(X)

    @property
    def _coefficients(self) -> np.ndarray:
        """Get the model coefficients.

        Returns:
            np.ndarray: Model coefficients rounded to 12 decimal places

        Raises:
            ValueError: If model hasn't been fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before accessing coefficients")
        return self.model.coef_.round(12)


def fit_map_by_time(
    df_position: pd.DataFrame, max_degree: int, mlflow_tags: Dict[str, Any] = {}
) -> Dict[str, FourierFit]:
    """Fit Fourier series to position data using time-based encoding.

    Args:
        df_position (pd.DataFrame): DataFrame containing position data
        max_degree (int): Maximum degree of harmonics to use

    Returns:
        Dict[str, FourierFit]: Dictionary of fitted models for x, y, z coordinates

    Raises:
        ValueError: If required columns are missing from df_position
    """
    required_columns = [
        "timing_from_lap",
        "time_lap",
        "coordinate_x",
        "coordinate_y",
        "coordinate_z",
    ]
    if not all(col in df_position.columns for col in required_columns):
        raise ValueError(f"df_position must contain columns: {required_columns}")

    encoding_by_time = df_position["timing_from_lap"] / df_position["time_lap"]
    fitting_by_time = {}
    for coord in ["x", "y", "z"]:
        fitting_by_time[coord] = FourierFit(
            max_degree,
            suffix="-{}-{}".format(mlflow_tags.get("method"), coord),
        ).fit(encoding_by_time, df_position[f"coordinate_{coord}"])
    return fitting_by_time


def correct_fit_by_distance(
    fitting_by_time: Dict[str, FourierFit],
    max_degree: int,
    predict_size: int,
    mlflow_tags: Dict[str, Any] = {},
) -> Dict[str, FourierFit]:
    """Correct the time-based fit using distance-based encoding.

    Args:
        fitting_by_time (Dict[str, FourierFit]): Time-based fitted models
        max_degree (int): Maximum degree of harmonics to use
        predict_size (int): Number of points to use for prediction

    Returns:
        Dict[str, FourierFit]: Dictionary of corrected fitted models
    """
    predict_parameter = np.linspace(0, 1, int(predict_size) + 1)
    data_by_time = np.column_stack(
        [fitting_by_time[k].predict(predict_parameter) for k in ["x", "y", "z"]]
    )

    # Calculate distance-based encoding
    distance = np.sqrt((np.diff(data_by_time[:, :2], axis=0) ** 2).sum(axis=1))
    encoding_by_distance = np.insert(distance.cumsum() / distance.sum(), 0, 0)

    # Fit new models using distance-based encoding
    fitting_by_distance = {}
    for i, coord in enumerate(["x", "y", "z"]):
        fitting_by_distance[coord] = FourierFit(
            max_degree,
            suffix="-{}-{}".format(mlflow_tags.get("method"), coord),
        ).fit(encoding_by_distance, data_by_time[:, i])
    return fitting_by_distance


def adjust_starting_point(
    fitting: Dict[str, FourierFit], df_pos: pd.DataFrame
) -> float:
    """Adjust the starting point of the circuit map.

    Args:
        fitting (Dict[str, FourierFit]): Fitted models
        df_pos (pd.DataFrame): Position data

    Returns:
        float: Adjusted starting encoding value

    Raises:
        ValueError: If required columns are missing from df_pos
    """
    index_lap = ["session_id", "driver_number", "lap_number"]
    required_columns = index_lap + ["timing_from_lap", "coordinate_x", "coordinate_y"]
    if not all(col in df_pos.columns for col in required_columns):
        raise ValueError(f"df_pos must contain columns: {required_columns}")

    df = df_pos.sort_values(by=index_lap + ["timing_from_lap"])
    first_point_in_lap = df.drop_duplicates(subset=index_lap, keep="first")[
        ["coordinate_x", "coordinate_y"]
    ].median()
    last_point_in_lap = df.drop_duplicates(subset=index_lap, keep="last")[
        ["coordinate_x", "coordinate_y"]
    ].median()
    start_point = (first_point_in_lap + last_point_in_lap) / 2
    start_point = start_point.values.reshape(-1)

    def error_function(angle: float) -> Tuple[float, float]:
        """Calculate error and its derivative for optimization.

        Args:
            angle (float): Current angle value

        Returns:
            Tuple[float, float]: Error value and its derivative
        """
        pred_d0 = np.array([fitting[k].predict(angle) for k in ["x", "y"]])
        pred_d1 = np.array([fitting[k].predict_d1(angle) for k in ["x", "y"]])
        diff = pred_d0.reshape(-1) - start_point
        fun = diff * diff
        deriv = 2 * diff * pred_d1.reshape(-1)
        return float(fun.sum()), float(deriv.sum())

    result = scipy.optimize.minimize(error_function, 0.0, jac=True)
    start_encoding = result.x[0]

    # Normalize to [-0.5, 0.5)
    while start_encoding >= 0.5:
        start_encoding -= 1
    while start_encoding < -0.5:
        start_encoding += 1
    return start_encoding


def format_output(
    fitting: Dict[str, FourierFit], predict_size: int, adjustment: float
) -> pd.DataFrame:
    """Format the circuit map output.

    Args:
        fitting (Dict[str, FourierFit]): Fitted models
        predict_size (int): Number of points to predict
        adjustment (float): Starting point adjustment

    Returns:
        pd.DataFrame: Formatted circuit map data
    """
    predict_parameter = np.linspace(0, 1, int(predict_size) + 1)
    call_parameter = predict_parameter + adjustment

    # Calculate coordinates and derivatives
    output_f = np.column_stack(
        [fitting[k].predict(call_parameter) for k in ["x", "y", "z"]]
    )
    output_der = np.column_stack(
        [fitting[k].predict_d1(call_parameter) for k in ["x", "y", "z"]]
    )

    # Create DataFrame
    save = pd.DataFrame(
        np.column_stack([predict_parameter, output_f, output_der])[:-1],
        columns=[
            "encoding",
            "coordinate_x",
            "coordinate_y",
            "coordinate_z",
            "derivative_x",
            "derivative_y",
            "derivative_z",
        ],
    )
    save = save.sort_values(by=["encoding"])

    # Calculate distances
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


def find_circuit_map(
    df_position: pd.DataFrame,
    max_degree: int,
    predict_size: int,
    mlflow_tags: Dict[str, Any] = {},
) -> Dict[str, Union[Dict[str, FourierFit], pd.DataFrame, float]]:
    """Find the circuit map using position data.

    Args:
        df_position (pd.DataFrame): Position data
        max_degree (int): Maximum degree of harmonics to use
        predict_size (int): Number of points to predict

    Returns:
        Dict containing:
            - fitting: Dictionary of fitted models
            - data: Circuit map data
            - adjustment: Starting point adjustment
    """
    with mlflow.start_run(run_name=mlflow_tags.get("run_id")):
        mlflow.set_tags(mlflow_tags)
        mlflow.log_param("max_degree", max_degree)
        mlflow.log_param("predict_size", predict_size)

        tmp_file = DATA_FOLDER / "features.parquet"
        df_position.to_parquet(tmp_file)
        mlflow.log_artifact(tmp_file.as_posix())
        tmp_file.unlink()

        fitting_by_time = fit_map_by_time(
            df_position, max_degree, mlflow_tags={**mlflow_tags, "method": "time"}
        )
        fitting_by_distance = correct_fit_by_distance(
            fitting_by_time,
            max_degree,
            predict_size,
            mlflow_tags={**mlflow_tags, "method": "distance"},
        )

        start_encoding = adjust_starting_point(fitting_by_distance, df_position)
        mlflow.log_metric("adjustment", start_encoding)

        save = format_output(fitting_by_distance, predict_size, start_encoding)
        tmp_file = DATA_FOLDER / "result.parquet"
        save.to_parquet(tmp_file)
        mlflow.log_artifact(tmp_file.as_posix())
        tmp_file.unlink()

    return dict(
        fitting=fitting_by_distance,
        data=save,
        adjustment=start_encoding,
    )


def main(
    session_ids: List[str], best_laps: int, max_degree: int, predict_size: int
) -> Dict:
    """Main function to generate circuit map.

    Args:
        session_ids (List[str]): List of session IDs to process
        best_laps (int): Number of best laps to use
        max_degree (int): Maximum degree of harmonics to use
        predict_size (int): Number of points to predict

    Returns:
        Dict: Circuit map data

    Raises:
        ValueError: If session_ids is empty or invalid
    """
    if not session_ids:
        raise ValueError("session_ids cannot be empty")

    df_laps_ls = []
    df_pos_ls = []
    for session_id in session_ids:
        try:
            year = int(session_id.strip()[1:5])
            round_id = session_id.strip()[5:8]
        except (ValueError, IndexError):
            raise ValueError(f"Invalid session_id format: {session_id}")

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
        df_position=df_pos_keep,
        max_degree=max_degree,
        predict_size=predict_size,
        mlflow_tags={
            "session_ids": "_".join(session_ids),
            "run_id": str(uuid4()),
        },
    )
    session_ids_str = "_".join(session_ids)
    save_folder = pathlib.Path(__file__).resolve().parent.parent / "data" / "artifacts"
    save_folder.mkdir(parents=True, exist_ok=True)
    filename = f"circuit_map_{session_ids_str}.parquet"
    output["data"].to_parquet(save_folder / filename, index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate circuit map from F1 telemetry data"
    )
    parser.add_argument(
        "session_id",
        type=str,
        help="Comma-separated list of session IDs (e.g., 'Y2023R01S01,Y2023R01S02')",
    )
    parser.add_argument(
        "--max_degree",
        type=int,
        default=100,
        help="Maximum degree of harmonics to use (default: 100)",
    )
    parser.add_argument(
        "--best_laps",
        type=int,
        default=200,
        help="Number of best laps to use (default: 200)",
    )
    args = parser.parse_args()

    main(
        session_ids=[x.strip() for x in str(args.session_id).split(",")],
        best_laps=int(args.best_laps),
        max_degree=int(args.max_degree),
        predict_size=int(1e5),
    )
