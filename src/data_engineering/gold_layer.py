import json
from datetime import UTC, datetime

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
            this_event["year"] = fix_integer(row["Year"])
            this_event["round_number"] = fix_integer(row["RoundNumber"])
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


class SessionMetadata(DatasetLocal):
    def __init__(self, year):
        self.year = year
        self.name = "gold/session_metadata_Y{:04d}".format(year)

    @staticmethod
    def parse_session(row):
        output = {}
        output["year"] = fix_integer(row["Year"])
        output["session_id"] = fix_string(row["SessionId"])
        output["session_name"] = fix_string(row["name"])
        output["f1_api_support"] = bool(row["f1_api_support"])
        output["timestamp_reference"] = datetime.fromisoformat(row["t0_date"]).replace(
            tzinfo=UTC
        )
        output["timing_start"] = float(row["session_start_time"])
        output["session_scheduled_time"] = datetime.fromisoformat(row["date"]).replace(
            tzinfo=UTC
        )
        output["total_laps"] = row["total_laps"]
        output["driver_list"] = sorted([fix_integer(x) for x in row["drivers"]])

        session_info = json.loads(row["session_info"])
        output["meeting_key"] = fix_integer(session_info.get("Meeting", {}).get("Key"))
        output["meeting_name"] = fix_string(session_info.get("Meeting", {}).get("Name"))
        output["meeting_official_name"] = fix_string(
            session_info.get("Meeting", {}).get("OfficialName")
        )
        output["meeting_location"] = fix_string(
            session_info.get("Meeting", {}).get("Location")
        )
        output["meeting_number"] = fix_integer(
            session_info.get("Meeting", {}).get("Number")
        )
        output["country_key"] = fix_integer(
            session_info.get("Meeting", {}).get("Country", {}).get("Key")
        )
        output["country_code"] = fix_string(
            session_info.get("Meeting", {}).get("Country", {}).get("Code")
        )
        output["country_name"] = fix_string(
            session_info.get("Meeting", {}).get("Country", {}).get("Name")
        )
        output["circuit_key"] = fix_integer(
            session_info.get("Meeting", {}).get("Circuit", {}).get("Key")
        )
        output["circuit_short_name"] = fix_string(
            session_info.get("Meeting", {}).get("Circuit", {}).get("ShortName")
        )

        event_info = json.loads(row["event"])
        output["round_number"] = fix_integer(event_info.get("RoundNumber"))
        output["event_country"] = fix_string(event_info.get("Country"))
        output["event_location"] = fix_string(event_info.get("Location"))
        output["event_name"] = fix_string(event_info.get("EventName"))
        output["event_official_name"] = fix_string(event_info.get("OfficialEventName"))
        output["event_date"] = datetime.fromisoformat(
            event_info.get("EventDate")
        ).date()
        output["event_format"] = fix_string(event_info.get("EventFormat"))

        sn = int(row["SessionId"][-1])
        output["type"] = fix_string(event_info.get(f"Session{sn}"))
        output["scheduled_time_utc"] = datetime.fromisoformat(
            event_info.get(f"Session{sn}DateUtc")
        ).replace(tzinfo=UTC)
        output["scheduled_time_local"] = fix_string(event_info.get(f"Session{sn}Date"))
        return output

    def run(self):
        dataset_source = DatasetLocal(
            name="silver/session_metadata_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_source.read_with_pattern())
        if len(ls) == 0:
            return None
        df_in = pd.concat(ls)
        df_out = pd.DataFrame([self.parse_session(r) for _, r in df_in.iterrows()])
        df_out = df_out.sort_values(by=["year", "session_id"]).reset_index(drop=True)
        return df_out


class SessionLaps(DatasetLocal):
    def __init__(self, year):
        self.year = year
        self.name = "gold/session_laps_Y{:04d}".format(year)

    def run(self):
        dataset_source = DatasetLocal(
            name="silver/session_laps_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_source.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)

        df["year"] = df["Year"].apply(fix_integer)
        df["session_id"] = df["SessionId"].apply(fix_string)
        df["driver_number"] = df["DriverNumber"].apply(fix_integer)
        df["driver_name"] = df["Driver"].apply(fix_string)
        df["driver_team"] = df["Team"].apply(fix_string)
        df["lap_number"] = df["LapNumber"].apply(fix_integer)
        df["timing_start_lap"] = df["LapStartTime"].apply(lambda x: x.total_seconds())
        df["timing_end_lap"] = df["Time"].apply(lambda x: x.total_seconds())
        df["timing_end_sector1"] = df["Sector1SessionTime"].apply(
            lambda x: x.total_seconds()
        )
        df["timing_end_sector2"] = df["Sector2SessionTime"].apply(
            lambda x: x.total_seconds()
        )
        df["timing_end_sector3"] = df["Sector3SessionTime"].apply(
            lambda x: x.total_seconds()
        )
        df["timing_pit_out"] = df["PitOutTime"].apply(lambda x: x.total_seconds())
        df["timing_pit_in"] = df["PitInTime"].apply(lambda x: x.total_seconds())
        df["timestamp_lap_start"] = pd.to_datetime(df["LapStartDate"], utc=True)
        df["time_lap"] = df["LapTime"].apply(lambda x: x.total_seconds())
        df["time_sector1"] = df["Sector1Time"].apply(lambda x: x.total_seconds())
        df["time_sector2"] = df["Sector2Time"].apply(lambda x: x.total_seconds())
        df["time_sector3"] = df["Sector3Time"].apply(lambda x: x.total_seconds())
        df["stint"] = df["Stint"].apply(fix_integer)
        df["tyre_life"] = df["TyreLife"].apply(fix_integer)
        df["tyre_compound"] = df["Compound"].apply(fix_string)
        df["tyre_fresh"] = df["FreshTyre"].astype(bool)
        df["speed_i1"] = df["SpeedI1"].astype(float)
        df["speed_i2"] = df["SpeedI2"].astype(float)
        df["speed_fl"] = df["SpeedFL"].astype(float)
        df["speed_st"] = df["SpeedST"].astype(float)
        df["is_personal_best"] = df["IsPersonalBest"].fillna(False).astype(bool)
        df["track_status"] = df["TrackStatus"].apply(fix_integer)
        df["position"] = df["Position"].apply(fix_integer)
        df["deleted_status"] = df["TrackStatus"].fillna(False).astype(bool)
        df["deleted_reason"] = df["DeletedReason"].apply(fix_string)
        df["is_accurate"] = df["IsAccurate"].fillna(False).astype(bool)
        df["is_fastf1_generated"] = df["FastF1Generated"].fillna(False).astype(bool)

        columns = [
            "year",
            "session_id",
            "driver_number",
            "driver_name",
            "driver_team",
            "lap_number",
            "stint",
            "timestamp_lap_start",
            "timing_start_lap",
            "timing_end_lap",
            "timing_end_sector1",
            "timing_end_sector2",
            "timing_end_sector3",
            "timing_pit_out",
            "timing_pit_in",
            "time_lap",
            "time_sector1",
            "time_sector2",
            "time_sector3",
            "tyre_compound",
            "tyre_life",
            "tyre_fresh",
            "speed_i1",
            "speed_i2",
            "speed_fl",
            "speed_st",
            "is_personal_best",
            "track_status",
            "position",
            "deleted_status",
            "deleted_reason",
            "is_accurate",
            "is_fastf1_generated",
        ]
        output = df.sort_values(by=["year", "session_id"])[columns].reset_index(
            drop=True
        )
        return output


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
