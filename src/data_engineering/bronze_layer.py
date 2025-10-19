import json

import fastf1
import pandas as pd
from fastf1.core import DataNotLoadedError
from loguru import logger
from typing import Optional, Union

from .datasets import DatasetLocal


class YearSchedule(DatasetLocal):
    """
    Dataset for retrieving the F1 event schedule for a given year, including testing sessions.
    """

    def __init__(self, year: int):
        """
        Initialize YearSchedule with the specified year.

        Args:
            year (int): The year for which to retrieve the event schedule.
        """
        self.year: int = int(year)
        self.name: str = "bronze/schedule_Y{:04d}".format(year)

    def run(self) -> pd.DataFrame:
        """
        Fetch the event schedule using fastf1 and return as a DataFrame.

        Returns:
            pd.DataFrame: Event schedule data with a 'Year' column.
        """
        logger.info("Calling fastf1.get_event_schedule...", end="")
        data = fastf1.get_event_schedule(
            year=self.year,
            include_testing=True,
        )
        data = pd.DataFrame(data)
        if "Year" not in data.columns:
            data["Year"] = self.year
        logger.info("Done")
        return data


class OfficialSession:
    """
    Mixin class for loading official F1 sessions and adding metadata.
    """

    def load_session(self) -> Optional[fastf1.core.Session]:
        """
        Load an official F1 session with weather, laps, and messages data.

        Returns:
            Optional[fastf1.core.Session]: The loaded session or None if not found or data not loaded.
        """
        try:
            session = fastf1.get_session(
                self.year, self.round_number, self.session_number
            )
        except ValueError:
            return None
        try:
            session.load(weather=True, laps=True, messages=True)
        except DataNotLoadedError:
            return None
        return session

    def add_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'Year' and 'SessionId' metadata columns to the DataFrame if missing.

        Args:
            data (pd.DataFrame): The DataFrame to add metadata to.

        Returns:
            pd.DataFrame: The DataFrame with added metadata columns.
        """
        if "Year" not in data.columns:
            data["Year"] = self.year
        if "SessionId" not in data.columns:
            data["SessionId"] = self.get_session_id()
        return data

    def get_session_id(self) -> str:
        """
        Generate a session ID string for official sessions.

        Returns:
            str: The session ID in the format 'Y{year}R{round}S{session}'.
        """
        return "Y{:04d}R{:02d}S{:01d}".format(
            self.year, self.round_number, self.session_number
        )


class TestingSession:
    """
    Mixin class for loading testing F1 sessions and adding metadata.
    """

    def load_session(self) -> Optional[fastf1.core.Session]:
        """
        Load a testing F1 session with weather, laps, and messages data.

        Returns:
            Optional[fastf1.core.Session]: The loaded session or None if not found or data not loaded.
        """
        try:
            logger.info("Calling fastf1.get_session...")
            session = fastf1.get_testing_session(
                self.year, self.round_number, self.session_number
            )
            logger.info("Done")
        except ValueError:
            return None
        try:
            session.load(weather=True, laps=True, messages=True)
        except DataNotLoadedError:
            return None
        return session

    def add_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'Year' and 'SessionId' metadata columns to the DataFrame if missing.

        Args:
            data (pd.DataFrame): The DataFrame to add metadata to.

        Returns:
            pd.DataFrame: The DataFrame with added metadata columns.
        """
        if "Year" not in data.columns:
            data["Year"] = self.year
        if "SessionId" not in data.columns:
            data["SessionId"] = self.get_session_id()
        return data

    def get_session_id(self) -> str:
        """
        Generate a session ID string for testing sessions.

        Returns:
            str: The session ID in the format 'Y{year}T{round}S{session}'.
        """
        return "Y{:04d}T{:02d}S{:01d}".format(
            self.year, self.round_number, self.session_number
        )


class SessionMetadata(DatasetLocal):
    """
    Dataset for retrieving metadata about a specific F1 session.
    """

    def __init__(self, year: int, round_number: int, session_number: int):
        """
        Initialize SessionMetadata with year, round number, and session number.

        Args:
            year (int): The year of the session.
            round_number (int): The round number of the session.
            session_number (int): The session number.
        """
        self.year: int = int(year)
        self.round_number: int = int(round_number)
        self.session_number: int = int(session_number)
        self.name: str = "bronze/session_metadata_{}".format(self.get_session_id())

    def run(self) -> Union[pd.DataFrame, None]:
        """
        Load the session and extract metadata as a DataFrame.

        Returns:
            pd.DataFrame | None: Metadata DataFrame or None if session not loaded.
        """
        session = self.load_session()
        if session is None:
            return None
        try:
            session_info = session.session_info.copy()
            session_info["StartDate"] = session_info["StartDate"].isoformat()
            session_info["EndDate"] = session_info["EndDate"].isoformat()
            session_info["GmtOffset"] = session_info["GmtOffset"].total_seconds()
            event = session.event.to_dict()
            for k in event.keys():
                if "Date" in k:
                    event[k] = event[k].isoformat()
            sr = pd.Series(
                {
                    "session_info": json.dumps(session_info),
                    "session_start_time": session.session_start_time.total_seconds(),
                    "t0_date": session.t0_date.isoformat(),
                    "event": json.dumps(event),
                    "total_laps": session.total_laps,
                    "drivers": session.drivers,
                    "date": session.date.isoformat(),
                    "name": session.name,
                    "f1_api_support": session.f1_api_support,
                }
            )
        except DataNotLoadedError:
            return None
        data = pd.DataFrame(sr).T
        data = self.add_metadata(data)
        return data


class SessionResults(DatasetLocal):
    """
    Dataset for retrieving results of a specific F1 session.
    """

    def __init__(self, year: int, round_number: int, session_number: int):
        """
        Initialize SessionResults with year, round number, and session number.

        Args:
            year (int): The year of the session.
            round_number (int): The round number of the session.
            session_number (int): The session number.
        """
        self.year: int = int(year)
        self.round_number: int = int(round_number)
        self.session_number: int = int(session_number)
        self.name: str = "bronze/session_results_{}".format(self.get_session_id())

    def run(self) -> Union[pd.DataFrame, None]:
        """
        Load the session and extract results as a DataFrame.

        Returns:
            pd.DataFrame | None: Results DataFrame or None if session not loaded.
        """
        session = self.load_session()
        if session is None:
            return None
        try:
            data = session.results
        except DataNotLoadedError:
            return None
        data = self.add_metadata(pd.DataFrame(data))
        return data


class SessionLaps(DatasetLocal):
    """
    Dataset for retrieving lap data of a specific F1 session.
    """

    def __init__(self, year: int, round_number: int, session_number: int):
        """
        Initialize SessionLaps with year, round number, and session number.

        Args:
            year (int): The year of the session.
            round_number (int): The round number of the session.
            session_number (int): The session number.
        """
        self.year: int = int(year)
        self.round_number: int = int(round_number)
        self.session_number: int = int(session_number)
        self.name: str = "bronze/session_laps_{}".format(self.get_session_id())

    def run(self) -> Union[pd.DataFrame, None]:
        """
        Load the session and extract lap data as a DataFrame.

        Returns:
            pd.DataFrame | None: Lap data DataFrame or None if session not loaded.
        """
        session = self.load_session()
        if session is None:
            return None
        try:
            data = session.laps
        except DataNotLoadedError:
            return None
        data = self.add_metadata(pd.DataFrame(data))
        return data


class SessionWeather(DatasetLocal):
    """
    Dataset for retrieving weather data of a specific F1 session.
    """

    def __init__(self, year: int, round_number: int, session_number: int):
        """
        Initialize SessionWeather with year, round number, and session number.

        Args:
            year (int): The year of the session.
            round_number (int): The round number of the session.
            session_number (int): The session number.
        """
        self.year: int = int(year)
        self.round_number: int = int(round_number)
        self.session_number: int = int(session_number)
        self.name: str = "bronze/session_weather_{}".format(self.get_session_id())

    def run(self) -> Union[pd.DataFrame, None]:
        """
        Load the session and extract weather data as a DataFrame.

        Returns:
            pd.DataFrame | None: Weather data DataFrame or None if session not loaded.
        """
        session = self.load_session()
        if session is None:
            return None
        try:
            data = session.weather_data
        except DataNotLoadedError:
            return None
        data = self.add_metadata(pd.DataFrame(data))
        return data


class SessionTrackStatus(DatasetLocal):
    """
    Dataset for retrieving track status data of a specific F1 session.
    """

    def __init__(self, year: int, round_number: int, session_number: int):
        """
        Initialize SessionTrackStatus with year, round number, and session number.

        Args:
            year (int): The year of the session.
            round_number (int): The round number of the session.
            session_number (int): The session number.
        """
        self.year: int = int(year)
        self.round_number: int = int(round_number)
        self.session_number: int = int(session_number)
        self.name: str = "bronze/session_track_status_{}".format(self.get_session_id())

    def run(self) -> Union[pd.DataFrame, None]:
        """
        Load the session and extract track status data as a DataFrame.

        Returns:
            pd.DataFrame | None: Track status data DataFrame or None if session not loaded.
        """
        session = self.load_session()
        if session is None:
            return None
        try:
            data = session.track_status
        except DataNotLoadedError:
            return None
        data = self.add_metadata(pd.DataFrame(data))
        return data


class SessionRaceControlMessages(DatasetLocal):
    """
    Dataset for retrieving race control messages of a specific F1 session.
    """

    def __init__(self, year: int, round_number: int, session_number: int):
        """
        Initialize SessionRaceControlMessages with year, round number, and session number.

        Args:
            year (int): The year of the session.
            round_number (int): The round number of the session.
            session_number (int): The session number.
        """
        self.year: int = int(year)
        self.round_number: int = int(round_number)
        self.session_number: int = int(session_number)
        self.name: str = "bronze/session_race_control_messages_{}".format(
            self.get_session_id()
        )

    def run(self) -> Union[pd.DataFrame, None]:
        """
        Load the session and extract race control messages as a DataFrame.

        Returns:
            pd.DataFrame | None: Race control messages DataFrame or None if session not loaded.
        """
        session = self.load_session()
        if session is None:
            return None
        try:
            data = session.race_control_messages
        except DataNotLoadedError:
            return None
        data = self.add_metadata(pd.DataFrame(data))
        return data


class TelemetryCarData(DatasetLocal):
    """
    Dataset for retrieving telemetry car data of a specific F1 session.
    """

    def __init__(self, year: int, round_number: int, session_number: int):
        """
        Initialize TelemetryCarData with year, round number, and session number.

        Args:
            year (int): The year of the session.
            round_number (int): The round number of the session.
            session_number (int): The session number.
        """
        self.year: int = int(year)
        self.round_number: int = int(round_number)
        self.session_number: int = int(session_number)
        self.name: str = "bronze/telemetry_car_{}".format(self.get_session_id())

    def run(self) -> Union[pd.DataFrame, None]:
        """
        Load the session and extract telemetry car data as a DataFrame.

        Returns:
            pd.DataFrame | None: Telemetry car data DataFrame or None if session not loaded.
        """
        session = self.load_session()
        if session is None:
            return None
        try:
            list_laps = session.laps
        except DataNotLoadedError:
            return None
        aux = []
        for _, lap in list_laps.iterlaps():
            this = lap.get_car_data()
            try:
                this = this.add_distance()
            except Exception as e:
                this = this.assign(Distance=None)
            try:
                this = this.add_differential_distance()
            except Exception as e:
                this = this.assign(DifferentialDistance=None)
            try:
                this = this.add_relative_distance()
            except Exception as e:
                this = this.assign(RelativeDistance=None)
            try:
                this = this.add_track_status()
            except Exception as e:
                this = this.assign(TrackStatus=None)
            try:
                this = this.add_driver_ahead()
            except Exception as e:
                this = this.assign(DriverAhead=None, DistanceToDriverAhead=None)
            this = this.assign(**{c: lap[c] for c in ["DriverNumber", "LapNumber"]})
            aux.append(pd.DataFrame(this))
        data = self.add_metadata(pd.concat(aux))
        return data


class TelemetryPosData(DatasetLocal):
    """
    Dataset for retrieving telemetry position data of a specific F1 session.
    """

    def __init__(self, year: int, round_number: int, session_number: int):
        """
        Initialize TelemetryPosData with year, round number, and session number.

        Args:
            year (int): The year of the session.
            round_number (int): The round number of the session.
            session_number (int): The session number.
        """
        self.year: int = int(year)
        self.round_number: int = int(round_number)
        self.session_number: int = int(session_number)
        self.name: str = "bronze/telemetry_pos_{}".format(self.get_session_id())

    def run(self) -> Union[pd.DataFrame, None]:
        """
        Load the session and extract telemetry position data as a DataFrame.

        Returns:
            pd.DataFrame | None: Telemetry position data DataFrame or None if session not loaded.
        """
        session = self.load_session()
        if session is None:
            return None
        try:
            list_laps = session.laps
        except DataNotLoadedError:
            return None
        aux = []
        for _, lap in list_laps.iterlaps():
            this = lap.get_pos_data()
            try:
                this = this.add_distance()
            except Exception as e:
                this = this.assign(Distance=None)
            try:
                this = this.add_differential_distance()
            except Exception as e:
                this = this.assign(DifferentialDistance=None)
            try:
                this = this.add_relative_distance()
            except Exception as e:
                this = this.assign(RelativeDistance=None)
            try:
                this = this.add_track_status()
            except Exception as e:
                this = this.assign(TrackStatus=None)
            try:
                this = this.add_driver_ahead()
            except Exception as e:
                this = this.assign(DriverAhead=None, DistanceToDriverAhead=None)
            this = this.assign(**{c: lap[c] for c in ["DriverNumber", "LapNumber"]})
            aux.append(pd.DataFrame(this))
        data = self.add_metadata(pd.concat(aux))
        return data


class CircuitMarkers(DatasetLocal):
    """
    Dataset for retrieving circuit marker data of a specific F1 session.
    """

    def __init__(self, year: int, round_number: int, session_number: int):
        """
        Initialize CircuitMarkers with year, round number, and session number.

        Args:
            year (int): The year of the session.
            round_number (int): The round number of the session.
            session_number (int): The session number.
        """
        self.year: int = int(year)
        self.round_number: int = int(round_number)
        self.session_number: int = int(session_number)
        self.name: str = "bronze/circuit_{}".format(self.get_session_id())

    def run(self) -> Union[pd.DataFrame, None]:
        """
        Load the session and extract circuit marker data as a DataFrame.

        Returns:
            pd.DataFrame | None: Circuit marker data DataFrame or None if session not loaded.
        """
        session = self.load_session()
        if session is None:
            return None
        try:
            circuit_info = session.get_circuit_info()
        except DataNotLoadedError:
            return None
        aux = [
            pd.DataFrame(circuit_info.corners).assign(annotation_type="corner"),
            pd.DataFrame(circuit_info.marshal_lights).assign(
                annotation_type="marshal_lights"
            ),
            pd.DataFrame(circuit_info.marshal_sectors).assign(
                annotation_type="marshal_sectors"
            ),
        ]
        data = pd.concat(aux).assign(rotation=circuit_info.rotation)
        data = self.add_metadata(data)
        return data


class OfficialSessionMetadata(OfficialSession, SessionMetadata):
    """Official session metadata dataset."""
    pass


class TestingSessionMetadata(TestingSession, SessionMetadata):
    """Testing session metadata dataset."""
    pass


class OfficialSessionResults(OfficialSession, SessionResults):
    """Official session results dataset."""
    pass


class TestingSessionResults(TestingSession, SessionResults):
    """Testing session results dataset."""
    pass


class OfficialSessionLaps(OfficialSession, SessionLaps):
    """Official session laps dataset."""
    pass


class TestingSessionLaps(TestingSession, SessionLaps):
    """Testing session laps dataset."""
    pass


class OfficialSessionWeather(OfficialSession, SessionWeather):
    """Official session weather dataset."""
    pass


class TestingSessionWeather(TestingSession, SessionWeather):
    """Testing session weather dataset."""
    pass


class OfficialSessionTrackStatus(OfficialSession, SessionTrackStatus):
    """Official session track status dataset."""
    pass


class TestingSessionTrackStatus(TestingSession, SessionTrackStatus):
    """Testing session track status dataset."""
    pass


class OfficialSessionRaceControlMessages(OfficialSession, SessionRaceControlMessages):
    """Official session race control messages dataset."""
    pass


class TestingSessionRaceControlMessages(TestingSession, SessionRaceControlMessages):
    """Testing session race control messages dataset."""
    pass


class OfficialTelemetryCarData(OfficialSession, TelemetryCarData):
    """Official telemetry car data dataset."""
    pass


class TestingTelemetryCarData(TestingSession, TelemetryCarData):
    """Testing telemetry car data dataset."""
    pass


class OfficialTelemetryPosData(OfficialSession, TelemetryPosData):
    """Official telemetry position data dataset."""
    pass


class TestingTelemetryPosData(TestingSession, TelemetryPosData):
    """Testing telemetry position data dataset."""
    pass


class OfficialCircuitMarkers(OfficialSession, CircuitMarkers):
    """Official circuit markers dataset."""
    pass


class TestingCircuitMarkers(TestingSession, CircuitMarkers):
    """Testing circuit markers dataset."""
    pass
