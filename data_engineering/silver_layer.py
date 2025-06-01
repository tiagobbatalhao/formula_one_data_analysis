import pandas as pd

from .datasets import DatasetLocal


class YearSchedule(DatasetLocal):
    def __init__(self):
        self.name = "silver/event_schedule"

    def run(self):
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
    def __init__(self, year):
        self.year = year
        self.name = "silver/session_metadata_Y{:04d}".format(year)

    def run(self):
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
    def __init__(self, year):
        self.year = year
        self.name = "silver/session_results_Y{:04d}".format(year)

    def run(self):
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
    def __init__(self, year):
        self.year = year
        self.name = "silver/session_laps_Y{:04d}".format(year)

    def run(self):
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
    def __init__(self, year):
        self.year = year
        self.name = "silver/session_weather_Y{:04d}".format(year)

    def run(self):
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
    def __init__(self, year):
        self.year = year
        self.name = "silver/session_track_status_Y{:04d}".format(year)

    def run(self):
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
    def __init__(self, year):
        self.year = year
        self.name = "silver/session_race_control_messages_Y{:04d}".format(year)

    def run(self):
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
    def __init__(self, year, round_id):
        self.year = year
        self.round_id = round_id
        self.name = "silver/telemetry_car_data_Y{:04d}{:s}".format(year, round_id)

    def run(self):
        dataset_bronze = DatasetLocal(
            name="bronze/telemetry_car_Y{:04d}{:s}*".format(self.year, self.round_id)
        )
        ls = list(dataset_bronze.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        idx = ["Year", "SessionId", "Time"]
        other = [c for c in df.columns if c not in idx]
        df = df.sort_values(by=idx)[idx + other].reset_index(drop=True)
        return df


class TelemetryPosData(DatasetLocal):
    def __init__(self, year, round_id):
        self.year = year
        self.round_id = round_id
        self.name = "silver/telemetry_pos_data_Y{:04d}{:s}".format(year, round_id)

    def run(self):
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


class TelemetryCarData(DatasetLocal):
    def __init__(self, year, round_id):
        self.year = year
        self.round_id = round_id
        self.name = "silver/telemetry_car_data_Y{:04d}{:s}".format(year, round_id)

    def run(self):
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
