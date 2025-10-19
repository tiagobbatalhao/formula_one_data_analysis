import pandas as pd
from typing import Optional, Union

from .datasets import DatasetLocal


class YearSchedule(DatasetLocal):
    """
    Dataset for retrieving the F1 event schedule aggregated in the silver layer.
    """

    def __init__(self):
        """
        Initialize YearSchedule dataset.
        """
        self.name: str = "silver/event_schedule"

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate event schedules from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated event schedule or None if no data.
        """
        dataset_bronze = DatasetLocal(name="bronze/schedule_Y*")
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "RoundNumber"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class SessionMetadata(DatasetLocal):
    """
    Dataset for retrieving session metadata aggregated in the silver layer.
    """

    def __init__(self, year: int):
        """
        Initialize SessionMetadata dataset.

        Args:
            year (int): The year of the session.
        """
        self.year: int = year
        self.name: str = "silver/session_metadata_Y{:04d}".format(year)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate session metadata from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated session metadata or None if no data.
        """
        dataset_bronze = DatasetLocal(
            name="bronze/session_metadata_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class SessionResults(DatasetLocal):
    """
    Dataset for retrieving session results aggregated in the silver layer.
    """

    def __init__(self, year: int):
        """
        Initialize SessionResults dataset.

        Args:
            year (int): The year of the session.
        """
        self.year: int = year
        self.name: str = "silver/session_results_Y{:04d}".format(year)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate session results from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated session results or None if no data.
        """
        dataset_bronze = DatasetLocal(
            name="bronze/session_results_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId", "DriverNumber"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class SessionLaps(DatasetLocal):
    """
    Dataset for retrieving session laps aggregated in the silver layer.
    """

    def __init__(self, year: int):
        """
        Initialize SessionLaps dataset.

        Args:
            year (int): The year of the session.
        """
        self.year: int = year
        self.name: str = "silver/session_laps_Y{:04d}".format(year)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate session laps from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated session laps or None if no data.
        """
        dataset_bronze = DatasetLocal(
            name="bronze/session_laps_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId", "DriverNumber", "LapNumber"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class SessionWeather(DatasetLocal):
    """
    Dataset for retrieving session weather aggregated in the silver layer.
    """

    def __init__(self, year: int):
        """
        Initialize SessionWeather dataset.

        Args:
            year (int): The year of the session.
        """
        self.year: int = year
        self.name: str = "silver/session_weather_Y{:04d}".format(year)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate session weather data from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated session weather data or None if no data.
        """
        dataset_bronze = DatasetLocal(
            name="bronze/session_weather_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId", "Time"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class SessionTrackStatus(DatasetLocal):
    """
    Dataset for retrieving session track status aggregated in the silver layer.
    """

    def __init__(self, year: int):
        """
        Initialize SessionTrackStatus dataset.

        Args:
            year (int): The year of the session.
        """
        self.year: int = year
        self.name: str = "silver/session_track_status_Y{:04d}".format(year)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate session track status data from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated session track status data or None if no data.
        """
        dataset_bronze = DatasetLocal(
            name="bronze/session_track_status_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId", "Time"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class SessionRaceControlMessages(DatasetLocal):
    """
    Dataset for retrieving session race control messages aggregated in the silver layer.
    """

    def __init__(self, year: int):
        """
        Initialize SessionRaceControlMessages dataset.

        Args:
            year (int): The year of the session.
        """
        self.year: int = year
        self.name: str = "silver/session_race_control_messages_Y{:04d}".format(year)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate session race control messages from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated session race control messages or None if no data.
        """
        dataset_bronze = DatasetLocal(
            name="bronze/session_race_control_messages_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId", "Time"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class TelemetryCarData(DatasetLocal):
    """
    Dataset for retrieving telemetry car data aggregated in the silver layer.
    """

    def __init__(self, year: int, round_id: str):
        """
        Initialize TelemetryCarData dataset.

        Args:
            year (int): The year of the session.
            round_id (str): The round identifier.
        """
        self.year: int = year
        self.round_id: str = round_id
        self.name: str = "silver/telemetry_car_data_Y{:04d}{:s}".format(year, round_id)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate telemetry car data from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated telemetry car data or None if no data.
        """
        dataset_bronze = DatasetLocal(
            name="bronze/telemetry_car_Y{:04d}{:s}*".format(self.year, self.round_id)
        )
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId", "Date"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class TelemetryPosData(DatasetLocal):
    """
    Dataset for retrieving telemetry position data aggregated in the silver layer.
    """

    def __init__(self, year: int, round_id: str):
        """
        Initialize TelemetryPosData dataset.

        Args:
            year (int): The year of the session.
            round_id (str): The round identifier.
        """
        self.year: int = year
        self.round_id: str = round_id
        self.name: str = "silver/telemetry_pos_data_Y{:04d}{:s}".format(year, round_id)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate telemetry position data from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated telemetry position data or None if no data.
        """
        dataset_bronze = DatasetLocal(
            name="bronze/telemetry_pos_Y{:04d}{:s}*".format(self.year, self.round_id)
        )
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId", "Date"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class CircuitMarkers(DatasetLocal):
    """
    Dataset for retrieving circuit marker data aggregated in the silver layer.
    """

    def __init__(self, year: int):
        """
        Initialize CircuitMarkers dataset.

        Args:
            year (int): The year of the session.
        """
        self.year: int = year
        self.name: str = "silver/circuit_{:04d}".format(year)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Aggregate circuit marker data from the bronze layer.

        Returns:
            Optional[pd.DataFrame]: Aggregated circuit marker data or None if no data.
        """
        dataset_bronze = DatasetLocal(name="bronze/circuit_Y{:04d}*".format(self.year))
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df
