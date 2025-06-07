import json
from datetime import UTC, datetime, timedelta

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
        output = df.sort_values(
            by=["year", "session_id", "driver_number", "lap_number"]
        )[columns].reset_index(drop=True)
        return output


class LapTiming(DatasetLocal):
    def __init__(self, year):
        self.year = year
        self.name = "gold/session_laptimings_Y{:04d}".format(year)

    @staticmethod
    def get_timestamp_start(info):
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_start_lap"]
            )
        except (TypeError, ValueError):
            pass
        try:
            return info["timestamp_lap_start"]
        except (TypeError, ValueError):
            pass

    @staticmethod
    def get_timestamp_end(info):
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_end_lap"]
            )
        except (TypeError, ValueError):
            pass

    @staticmethod
    def get_timestamp_pitout(info):
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_pit_out"]
            )
        except (TypeError, ValueError):
            pass

    @staticmethod
    def get_timestamp_pitin(info):
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_pit_in"]
            )
        except (TypeError, ValueError):
            pass

    @staticmethod
    def get_timestamp_sector1(info):
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_end_sector1"]
            )
        except (TypeError, ValueError):
            pass
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_end_sector2"] - info["time_sector2"]
            )
        except (TypeError, ValueError):
            pass

    @staticmethod
    def get_timestamp_sector2(info):
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_end_sector2"]
            )
        except (TypeError, ValueError):
            pass
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_end_sector1"] + info["time_sector2"]
            )
        except (TypeError, ValueError):
            pass

    @staticmethod
    def get_timestamp_sector3(info):
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_end_sector3"]
            )
        except (TypeError, ValueError):
            pass
        try:
            return info["timestamp_reference"] + timedelta(
                seconds=info["timing_end_sector2"] + info["time_sector3"]
            )
        except (TypeError, ValueError):
            pass

    def run(self):
        df_session = SessionMetadata(self.year).read()
        df_laps = SessionLaps(self.year).read()
        idx = ["year", "session_id", "driver_number", "lap_number"]
        df_lap_timing = df_laps.merge(
            df_session[["year", "session_id", "timestamp_reference"]],
            on=["year", "session_id"],
        )
        df_lap_timing["timestamp_start"] = pd.to_datetime(
            df_lap_timing.apply(self.get_timestamp_start, axis=1)
        )
        df_lap_timing["timestamp_end"] = pd.to_datetime(
            df_lap_timing.apply(self.get_timestamp_end, axis=1)
        )
        df_lap_timing["timestamp_pitout"] = pd.to_datetime(
            df_lap_timing.apply(self.get_timestamp_pitout, axis=1)
        )
        df_lap_timing["timestamp_pitin"] = pd.to_datetime(
            df_lap_timing.apply(self.get_timestamp_pitin, axis=1)
        )
        df_lap_timing["timestamp_sector1"] = pd.to_datetime(
            df_lap_timing.apply(self.get_timestamp_sector1, axis=1)
        )
        df_lap_timing["timestamp_sector2"] = pd.to_datetime(
            df_lap_timing.apply(self.get_timestamp_sector2, axis=1)
        )
        df_lap_timing["timestamp_sector3"] = pd.to_datetime(
            df_lap_timing.apply(self.get_timestamp_sector3, axis=1)
        )

        df_lap_timing = df_lap_timing[
            idx
            + [
                "timestamp_start",
                "timestamp_pitout",
                "timestamp_sector1",
                "timestamp_sector2",
                "timestamp_pitin",
                "timestamp_sector3",
                "timestamp_end",
            ]
        ]
        return df_lap_timing


class TelemetryPosData(DatasetLocal):
    def __init__(self, year, round_id):
        self.year = year
        self.round_id = round_id
        self.name = "gold/telemetry_pos_Y{:04d}{:s}".format(year, round_id)

    def run(self):
        dataset_source = DatasetLocal(
            name="silver/telemetry_pos_data_Y{:04d}{:s}*".format(
                self.year, self.round_id
            )
        )
        ls = list(dataset_source.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)

        df["year"] = df["Year"].apply(fix_integer)
        df["session_id"] = df["SessionId"].apply(fix_string)
        df["driver_number"] = df["DriverNumber"].apply(fix_integer)
        df["lap_number"] = df["LapNumber"].apply(fix_integer)
        df["timestamp"] = pd.to_datetime(df["Date"], utc=True)
        df["timing_from_session"] = (
            df["SessionTime"].apply(lambda x: x.total_seconds()).astype(float)
        )
        df["timing_from_lap"] = (
            df["Time"].apply(lambda x: x.total_seconds()).astype(float)
        )

        df["position_status"] = df["Status"].apply(fix_string)
        df["track_status"] = (
            df["TrackStatus"].apply(fix_integer).astype(pd.Int64Dtype())
        )
        data_columns = [
            ("X", "coordinate_x"),
            ("Y", "coordinate_y"),
            ("Z", "coordinate_z"),
        ]
        for old, new in data_columns:
            df[new] = df[old].astype(float)
        columns = [
            "year",
            "session_id",
            "driver_number",
            "lap_number",
            "timestamp",
            "timing_from_session",
            "timing_from_lap",
            "track_status",
            "coordinate_x",
            "coordinate_y",
            "coordinate_z",
            "position_status",
            # Distance # TODO
            # DifferentialDistance # TODO
            # RelativeDistance # TODO
            # DriverAhead # TODO
            # DistanceToDriverAhead # TODO
            # Source # TODO
        ]
        output = df.sort_values(
            by=["year", "session_id", "driver_number", "lap_number"]
        )[columns].reset_index(drop=True)
        return output


class TelemetryCarData(DatasetLocal):
    def __init__(self, year, round_id):
        self.year = year
        self.round_id = round_id
        self.name = "gold/telemetry_car_Y{:04d}{:s}".format(year, round_id)

    def run(self):
        dataset_source = DatasetLocal(
            name="silver/telemetry_car_data_Y{:04d}{:s}*".format(
                self.year, self.round_id
            )
        )
        ls = list(dataset_source.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)

        df["year"] = df["Year"].apply(fix_integer)
        df["session_id"] = df["SessionId"].apply(fix_string)
        df["driver_number"] = df["DriverNumber"].apply(fix_integer)
        df["lap_number"] = df["LapNumber"].apply(fix_integer)
        df["timestamp"] = pd.to_datetime(df["Date"], utc=True)
        df["timing_from_session"] = (
            df["SessionTime"].apply(lambda x: x.total_seconds()).astype(float)
        )
        df["timing_from_lap"] = (
            df["Time"].apply(lambda x: x.total_seconds()).astype(float)
        )

        df["track_status"] = (
            df["TrackStatus"].apply(fix_integer).astype(pd.Int64Dtype())
        )
        data_columns = [
            ("RPM", "rpm"),
            ("Speed", "speed"),
            ("nGear", "ngear"),
            ("Throttle", "throttle"),
            ("Brake", "brake"),
            ("DRS", "drs"),
        ]
        for old, new in data_columns:
            df[new] = df[old].astype(float)
        columns = [
            "year",
            "session_id",
            "driver_number",
            "lap_number",
            "timestamp",
            "timing_from_session",
            "timing_from_lap",
            "track_status",
            "rpm",
            "speed",
            "ngear",
            "throttle",
            "brake",
            "drs",
            # Distance # TODO
            # DifferentialDistance # TODO
            # RelativeDistance # TODO
            # DriverAhead # TODO
            # DistanceToDriverAhead # TODO
            # Source # TODO
        ]
        output = df.sort_values(
            by=["year", "session_id", "driver_number", "lap_number"]
        )[columns].reset_index(drop=True)
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
        df["timing_from_session"] = (
            df["Time"].apply(lambda x: x.total_seconds()).astype(float)
        )
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
        df_session = SessionMetadata(self.year).read()
        df = df.merge(
            df_session[["year", "session_id", "timestamp_reference"]],
            on=["year", "session_id"],
            how="left",
        )
        df["timestamp"] = df.apply(
            lambda r: r["timestamp_reference"]
            + timedelta(seconds=r["timing_from_session"]),
            axis=1,
        )
        columns = [
            "year",
            "session_id",
            "timestamp",
            "timing_from_session",
            "rainfall",
            "air_temperature",
            "humidity",
            "pressure",
            "track_temperature",
            "wind_direction",
            "wind_speed",
        ]
        df_output = df.sort_values(by=["year", "session_id", "timestamp"])[
            columns
        ].reset_index(drop=True)
        return df_output


class SessionTrackStatus(DatasetLocal):
    def __init__(self, year):
        self.year = year
        self.name = "gold/session_track_status_Y{:04d}".format(year)

    def run(self):
        dataset_source = DatasetLocal(
            name="silver/session_track_status_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_source.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        df["year"] = df["Year"].astype(pd.Int64Dtype())
        df["session_id"] = df["SessionId"].apply(fix_string)
        df["timing_from_session"] = (
            df["Time"].apply(lambda x: x.total_seconds()).astype(float)
        )
        df["status"] = df["Status"].apply(fix_integer).astype(pd.Int64Dtype())
        df["message"] = df["Message"].apply(fix_string)
        df_session = SessionMetadata(self.year).read()
        df = df.merge(
            df_session[["year", "session_id", "timestamp_reference"]],
            on=["year", "session_id"],
            how="left",
        )
        df["timestamp"] = df.apply(
            lambda r: r["timestamp_reference"]
            + timedelta(seconds=r["timing_from_session"]),
            axis=1,
        )
        columns = [
            "year",
            "session_id",
            "timestamp",
            "timing_from_session",
            "status",
            "message",
        ]
        df_output = df.sort_values(by=["year", "session_id", "timestamp"])[
            columns
        ].reset_index(drop=True)
        return df_output


class SessionRaceControlMessages(DatasetLocal):
    def __init__(self, year):
        self.year = year
        self.name = "gold/session_race_control_messages_Y{:04d}".format(year)

    def run(self):
        dataset_source = DatasetLocal(
            name="silver/session_race_control_messages_Y{:04d}*".format(self.year)
        )
        ls = list(dataset_source.read_with_pattern())
        if len(ls) == 0:
            return None
        df = pd.concat(ls)
        df["year"] = df["Year"].astype(pd.Int64Dtype())
        df["session_id"] = df["SessionId"].apply(fix_string)
        df["timestamp"] = pd.to_datetime(df['Time'], utc=True)
        df["category"] = df["Category"].apply(fix_string)
        df["message"] = df["Message"].apply(fix_string)
        df["status"] = df["Status"].apply(fix_string)
        df["flag"] = df["Flag"].apply(fix_string)
        df["scope"] = df["Scope"].apply(fix_string)
        df["sector"] = df["Sector"].apply(fix_integer).astype(pd.Int64Dtype())
        df["driver_number"] = (
            df["RacingNumber"].apply(fix_integer).astype(pd.Int64Dtype())
        )
        df["lap_number"] = df["Lap"].apply(fix_integer).astype(pd.Int64Dtype())
        columns = [
            "year",
            "session_id",
            "timestamp",
            "category",
            "message",
            "status",
            "flag",
            "scope",
            "sector",
            "driver_number",
            "lap_number",
        ]
        df_output = df.sort_values(by=["year", "session_id", "timestamp"])[
            columns
        ].reset_index(drop=True)
        return df_output
