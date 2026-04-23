from datetime import date, datetime
from typing import List, Literal, Optional

import numpy as np
from pydantic import BaseModel, field_validator


class HistoricalSessions(BaseModel):
    year: int
    round_number: int
    event_name: str
    event_format: Literal[
        "conventional", "testing", "sprint", "sprint_shootout", "sprint_qualifying"
    ]
    event_date: date
    country: str
    location: str
    official_event_name: Optional[str]
    has_api_support: bool
    session_number: Literal[1, 2, 3, 4, 5]
    session_name: Literal[
        "Practice 1",
        "Practice 2",
        "Practice 3",
        "Qualifying",
        "Race",
        "Sprint",
        "Sprint Qualifying",
        "Sprint Shootout",
    ]
    session_time_utc: datetime


class SessionMetadata(BaseModel):
    year: int
    session_id: str
    session_name: Literal[
        "Practice 1",
        "Practice 2",
        "Practice 3",
        "Qualifying",
        "Race",
        "Sprint",
        "Sprint Qualifying",
        "Sprint Shootout",
    ]
    f1_api_support: bool
    timestamp_reference: datetime
    session_scheduled_time: datetime
    timing_start: float
    total_laps: Optional[int]
    driver_list: List[int]
    meeting_key: int
    meeting_name: str
    meeting_official_name: str
    meeting_location: str
    meeting_number: int
    country_key: int
    country_code: str
    country_name: str
    circuit_key: int
    circuit_short_name: str
    round_number: int
    event_country: str
    event_location: str
    event_name: str
    event_official_name: str
    event_date: date
    event_format: Literal[
        "conventional", "testing", "sprint", "sprint_shootout", "sprint_qualifying"
    ]
    type: Literal[
        "Practice 1",
        "Practice 2",
        "Practice 3",
        "Qualifying",
        "Race",
        "Sprint",
        "Sprint Qualifying",
        "Sprint Shootout",
    ]
    scheduled_time_utc: datetime
    scheduled_time_local: datetime

    @field_validator("total_laps", mode="before")
    @staticmethod
    def total_laps(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


class SessionResults(BaseModel):
    year: int
    session_id: str
    driver_id: Optional[str]
    driver_number: int
    driver_broadcast_name: str
    driver_abbreviation: str
    driver_first_name: str
    driver_last_name: str
    driver_full_name: str
    driver_headshot_url: Optional[str]
    team_color: str
    position: Optional[int]
    classified_position: Optional[int]
    grid_position: Optional[int]
    time_q1: Optional[float]
    time_q2: Optional[float]
    time_q3: Optional[float]
    time: Optional[float]
    status: Literal[
        None, "Finished", "Retired", "Disqualified", "Lapped", "Did not start"
    ]
    points: Optional[int]

    @staticmethod
    def parse_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def parse_float(value):
        try:
            if np.isnan(value):
                return None
            if value:
                return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("position", mode="before")
    @staticmethod
    def position(value):
        return SessionResults.parse_int(value)

    @field_validator("classified_position", mode="before")
    @staticmethod
    def classified_position(value):
        return SessionResults.parse_int(value)

    @field_validator("grid_position", mode="before")
    @staticmethod
    def grid_position(value):
        return SessionResults.parse_int(value)

    @field_validator("points", mode="before")
    @staticmethod
    def points(value):
        return SessionResults.parse_int(value)

    @field_validator("time_q1", mode="before")
    @staticmethod
    def time_q1(value):
        return SessionResults.parse_float(value)

    @field_validator("time_q2", mode="before")
    @staticmethod
    def time_q2(value):
        return SessionResults.parse_float(value)

    @field_validator("time_q3", mode="before")
    @staticmethod
    def time_q3(value):
        return SessionResults.parse_float(value)

    @field_validator("time", mode="before")
    @staticmethod
    def time(value):
        return SessionResults.parse_float(value)


class SessionLaps(BaseModel):
    year: int
    session_id: str
    driver_number: int
    driver_name: str
    driver_team: str
    lap_number: int
    stint: Optional[int]
    timestamp_lap_start: datetime
    timing_start_lap: Optional[float]
    timing_end_lap: Optional[float]
    timing_end_sector1: Optional[float]
    timing_end_sector2: Optional[float]
    timing_end_sector3: Optional[float]
    timing_pit_out: Optional[float]
    timing_pit_in: Optional[float]
    time_lap: Optional[float]
    time_sector1: Optional[float]
    time_sector2: Optional[float]
    time_sector3: Optional[float]
    tyre_compound: Literal[
        None, "SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"
    ]
    tyre_life: Optional[int]
    speed_i1: Optional[float]
    speed_i2: Optional[float]
    speed_fl: Optional[float]
    speed_st: Optional[float]
    is_personal_best: bool
    track_status: Optional[int]
    position: Optional[int]
    deleted_reason: Optional[str]
    is_accurate: bool
    is_fastf1_generated: bool

    @staticmethod
    def parse_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def parse_float(value):
        try:
            if np.isnan(value):
                return None
            if value:
                return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("stint", mode="before")
    @staticmethod
    def stint(value):
        return SessionLaps.parse_int(value)

    @field_validator("tyre_life", mode="before")
    @staticmethod
    def tyre_life(value):
        return SessionLaps.parse_int(value)

    @field_validator("track_status", mode="before")
    @staticmethod
    def track_status(value):
        return SessionLaps.parse_int(value)

    @field_validator("position", mode="before")
    @staticmethod
    def position(value):
        return SessionLaps.parse_int(value)

    @field_validator("timing_start_lap", mode="before")
    @staticmethod
    def timing_start_lap(value):
        return SessionLaps.parse_float(value)

    @field_validator("timing_end_lap", mode="before")
    @staticmethod
    def timing_end_lap(value):
        return SessionLaps.parse_float(value)

    @field_validator("timing_end_sector1", mode="before")
    @staticmethod
    def timing_end_sector1(value):
        return SessionLaps.parse_float(value)

    @field_validator("timing_end_sector2", mode="before")
    @staticmethod
    def timing_end_sector2(value):
        return SessionLaps.parse_float(value)

    @field_validator("timing_end_sector3", mode="before")
    @staticmethod
    def timing_end_sector3(value):
        return SessionLaps.parse_float(value)

    @field_validator("timing_pit_in", mode="before")
    @staticmethod
    def timing_pit_in(value):
        return SessionLaps.parse_float(value)

    @field_validator("timing_pit_out", mode="before")
    @staticmethod
    def timing_pit_out(value):
        return SessionLaps.parse_float(value)

    @field_validator("time_lap", mode="before")
    @staticmethod
    def time_lap(value):
        return SessionLaps.parse_float(value)

    @field_validator("time_sector1", mode="before")
    @staticmethod
    def time_sector1(value):
        return SessionLaps.parse_float(value)

    @field_validator("time_sector2", mode="before")
    @staticmethod
    def time_sector2(value):
        return SessionLaps.parse_float(value)

    @field_validator("time_sector3", mode="before")
    @staticmethod
    def time_sector3(value):
        return SessionLaps.parse_float(value)

    @field_validator("speed_i1", mode="before")
    @staticmethod
    def speed_i1(value):
        return SessionLaps.parse_float(value)

    @field_validator("speed_i2", mode="before")
    @staticmethod
    def speed_i2(value):
        return SessionLaps.parse_float(value)

    @field_validator("speed_fl", mode="before")
    @staticmethod
    def speed_fl(value):
        return SessionLaps.parse_float(value)

    @field_validator("speed_st", mode="before")
    @staticmethod
    def speed_st(value):
        return SessionLaps.parse_float(value)


class SessionLapTimings(BaseModel):
    year: int
    session_id: str
    driver_number: int
    lap_number: int
    timestamp_start: Optional[datetime]
    timestamp_pitout: Optional[datetime]
    timestamp_sector1: Optional[datetime]
    timestamp_sector2: Optional[datetime]
    timestamp_pitin: Optional[datetime]
    timestamp_sector3: Optional[datetime]
    timestamp_end: Optional[datetime]


class SessionWeather(BaseModel):
    year: int
    session_id: str
    timestamp: datetime
    timing_from_session: float
    rainfall: bool
    air_temperature: Optional[float]
    humidity: Optional[float]
    pressure: Optional[float]
    track_temperature: Optional[float]
    wind_direction: Optional[float]
    wind_speed: Optional[float]

    @staticmethod
    def parse_float(value):
        try:
            if np.isnan(value):
                return None
            if value:
                return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("air_temperature", mode="before")
    @staticmethod
    def air_temperature(value):
        return SessionWeather.parse_float(value)

    @field_validator("humidity", mode="before")
    @staticmethod
    def humidity(value):
        return SessionWeather.parse_float(value)

    @field_validator("pressure", mode="before")
    @staticmethod
    def pressure(value):
        return SessionWeather.parse_float(value)

    @field_validator("track_temperature", mode="before")
    @staticmethod
    def track_temperature(value):
        return SessionWeather.parse_float(value)

    @field_validator("wind_direction", mode="before")
    @staticmethod
    def wind_direction(value):
        return SessionWeather.parse_float(value)

    @field_validator("wind_speed", mode="before")
    @staticmethod
    def wind_speed(value):
        return SessionWeather.parse_float(value)


class SessionRaceControlMessages(BaseModel):
    year: int
    session_id: str
    timestamp: datetime
    timing_from_session: float
    category: Literal["Flag", "Drs", "SafetyCar", "Other"]
    message: Optional[str]
    status: Literal[
        None,
        "ENABLED",
        "DISABLED",
        "DEPLOYED",
        "ENDING",
        "IN THIS LAP",
        "THROUGH THE PIT LANE",
    ]
    flag: Literal[
        None,
        "CLEAR",
        "YELLOW",
        "DOUBLE YELLOW",
        "BLUE",
        "GREEN",
        "CHEQUERED",
        "RED",
        "BLACK AND WHITE",
    ]
    scope: Literal[None, "Sector", "Track", "Driver"]
    sector: Optional[int]
    driver_number: Optional[int]
    lap_number: Optional[int]

    @staticmethod
    def parse_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @field_validator("sector", mode="before")
    @staticmethod
    def sector(value):
        return SessionResults.parse_int(value)

    @field_validator("driver_number", mode="before")
    @staticmethod
    def driver_number(value):
        return SessionResults.parse_int(value)

    @field_validator("lap_number", mode="before")
    @staticmethod
    def lap_number(value):
        return SessionResults.parse_int(value)


class SessionTrackStatus(BaseModel):
    year: int
    session_id: str
    timestamp: datetime
    timing_from_session: float
    status: Literal[1, 2, 3, 4, 5, 6, 7]
    message: Literal[
        "Yellow", "AllClear", "Red", "SCDeployed", "VSCDeployed", "VSCEnding"
    ]


class CircuitMarkers(BaseModel):
    year: int
    session_id: str
    annotation_type: Literal["corner", "marshal_lights", "marshal_sectors"]
    number: int
    letter: Optional[str]
    coordinate_x: float
    coordinate_y: float
    angle: float
    distance: float
    rotation: float


class TelemetryCar(BaseModel):
    year: int
    session_id: str
    driver_number: int
    lap_number: int
    timestamp: datetime
    timing_from_session: float
    timing_from_lap: float
    track_status: Optional[int]
    rpm: float
    speed: float
    ngear: int
    speed: float
    brake: float
    drs: int

    @staticmethod
    def parse_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @field_validator("track_status", mode="before")
    @staticmethod
    def track_status(value):
        return SessionLaps.parse_int(value)


class TelemetryPos(BaseModel):
    year: int
    session_id: str
    driver_number: int
    lap_number: int
    timestamp: datetime
    timing_from_session: float
    timing_from_lap: float
    track_status: Optional[int]
    coordinate_x: float
    coordinate_y: float
    coordinate_z: float
    position_status: Literal["OnTrack", "OffTrack"]

    @staticmethod
    def parse_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @field_validator("track_status", mode="before")
    @staticmethod
    def track_status(value):
        return SessionLaps.parse_int(value)
