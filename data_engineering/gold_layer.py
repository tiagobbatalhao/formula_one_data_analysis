from datetime import datetime

import pandas as pd

from .datasets import DatasetLocal


def fix_string(value):
    if not isinstance(value, str):
        return None
    if len(value) == 0:
        return None
    if value.lower() in ["none", "nan"]:
        return None
    return str(value)


def fix_integer(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


class HistoricalSessions(DatasetLocal):
    def __init__(self):
        self.name = "gold/historical_sessions"

    def run(self):
        dataset_source = DatasetLocal(name="silver/event_schedule*")
        df = pd.concat(list(dataset_source.read_with_pattern()))
        aux = []
        for _, row in df.iterrows():
            this_event = {}
            this_event["year"] = int(row["Year"])
            this_event["round_number"] = int(row["RoundNumber"])
            this_event["country"] = fix_string(row["Country"])
            this_event["location"] = fix_string(row["Location"])
            this_event["event_name"] = fix_string(row["EventName"])
            this_event["official_event_name"] = fix_string(row["OfficialEventName"])
            this_event["event_format"] = fix_string(row["EventFormat"])
            this_event["event_date"] = row["EventDate"].date()
            this_event["has_api_support"] = row["F1ApiSupport"]
            for session_number in range(1, 6):
                this_session = {}
                this_session["session_number"] = session_number
                this_session["session_name"] = fix_string(
                    row[f"Session{session_number}"]
                )
                this_session["session_time_utc"] = row[
                    f"Session{session_number}DateUtc"
                ]
                aux.append({**this_event, **this_session})
        df_output = pd.DataFrame(aux)
        for col in ["session_time_utc"]:
            df_output[col] = pd.to_datetime(df_output[col])
        for col in ["year", "round_number"]:
            df_output[col] = df_output[col].astype(pd.Int64Dtype())
        columns = [
            "year",
            "round_number",
            "event_name",
            "event_format",
            "event_date",
            "country",
            "location",
            "official_event_name",
            "has_api_support",
            "session_number",
            "session_name",
            "session_time_utc",
        ]
        df_output = (
            df_output.dropna(subset=["session_name"])
            .sort_values(by="session_time_utc")[columns]
            .reset_index(drop=True)
        )
        return df_output


class SessionResults(DatasetLocal):
    def __init__(self, year):
        self.year = year
        self.name = "gold/session_results_Y{:04d}".format(year)

    def run(self):
        dataset_source = DatasetLocal(
            name="silver/session_results_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_source.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        df["year"] = df["Year"].astype(pd.Int64Dtype())
        df["session_id"] = df["SessionId"].apply(fix_string)
        df["driver_number"] = df["DriverNumber"].astype(pd.Int64Dtype())
        df["driver_broadcast_name"] = df["BroadcastName"].apply(fix_string)
        df["driver_abbreviation"] = df["Abbreviation"].apply(fix_string)
        df["driver_id"] = df["DriverId"].apply(fix_string)
        df["driver_first_name"] = df["FirstName"].apply(fix_string)
        df["driver_last_name"] = df["LastName"].apply(fix_string)
        df["driver_full_name"] = df["FullName"].apply(fix_string)
        df["driver_headshot_url"] = df["HeadshotUrl"].apply(fix_string)
        df["driver_country_code"] = df["CountryCode"].apply(fix_string)
        df["team_name"] = df["TeamName"].apply(fix_string)
        df["team_color"] = df["TeamColor"].apply(fix_string)
        df["team_id"] = df["TeamId"].apply(fix_string)
        df["position"] = df["Position"].apply(fix_integer).astype(pd.Int64Dtype())
        df["classified_position"] = (
            df["ClassifiedPosition"].apply(fix_integer).astype(pd.Int64Dtype())
        )
        df["grid_position"] = (
            df["GridPosition"].apply(fix_integer).astype(pd.Int64Dtype())
        )
        df["time_q1"] = df["Q1"].apply(lambda x: x.total_seconds()).astype(float)
        df["time_q2"] = df["Q2"].apply(lambda x: x.total_seconds()).astype(float)
        df["time_q3"] = df["Q3"].apply(lambda x: x.total_seconds()).astype(float)
        df["time"] = df["Time"].apply(lambda x: x.total_seconds()).astype(float)
        df["status"] = df["Status"].apply(fix_string)
        df["points"] = df["Points"].astype(float)
        columns = [
            "year",
            "session_id",
            "driver_id",
            "driver_number",
            "driver_broadcast_name",
            "driver_abbreviation",
            "driver_first_name",
            "driver_last_name",
            "driver_full_name",
            "driver_headshot_url",
            "driver_country_code",
            "team_id",
            "team_name",
            "team_color",
            "position",
            "classified_position",
            "grid_position",
            "time_q1",
            "time_q2",
            "time_q3",
            "time",
            "status",
            "points",
        ]
        df_output = df.sort_values(by=["year", "session_id"])[columns].reset_index(
            drop=True
        )
        return df_output


class SessionWeather(DatasetLocal):
    def __init__(self, year):
        self.year = year
        self.name = "gold/session_weather_Y{:04d}".format(year)

    def run(self):
        dataset_source = DatasetLocal(
            name="silver/session_weather_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_source.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        df["year"] = df["Year"].astype(pd.Int64Dtype())
        df["session_id"] = df["SessionId"].apply(fix_string)
        df["time"] = df["Time"].apply(lambda x: x.total_seconds()).astype(float)
        df["rainfall"] = df["Rainfall"].astype(bool)
        data_columns = [
            ("AirTemp", "air_temperature"),
            ("Humidity", "humidity"),
            ("Pressure", "pressure"),
            ("TrackTemp", "track_temperature"),
            ("WindDirection", "wind_direction"),
            ("WindSpeed", "wind_speed"),
        ]
        for old, new in data_columns:
            df[new] = df[old].astype(float)
        columns = [
            "year",
            "session_id",
            "time",
            "rainfall",
            "air_temperature",
            "humidity",
            "pressure",
            "track_temperature",
            "wind_direction",
            "wind_speed",
        ]
        df_output = df.sort_values(by=["year", "session_id"])[columns].reset_index(
            drop=True
        )
        return df_output
