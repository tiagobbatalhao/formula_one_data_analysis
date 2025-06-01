import json

import fastf1
import pandas as pd
from fastf1.core import DataNotLoadedError
from loguru import logger

from .datasets import DatasetLocal


class YearSchedule(DatasetLocal):
    def __init__(self, year: int):
        self.year = int(year)
        self.name = "bronze/schedule_Y{:04d}".format(year)

    def run(self):
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
    def load_session(self):
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

    def add_metadata(self, data):
        if "Year" not in data.columns:
            data["Year"] = self.year
        if "SessionId" not in data.columns:
            data["SessionId"] = self.get_session_id()
        return data

    def get_session_id(self):
        return "Y{:04d}R{:02d}S{:01d}".format(
            self.year, self.round_number, self.session_number
        )


class TestingSession:
    def load_session(self):
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

    def add_metadata(self, data):
        if "Year" not in data.columns:
            data["Year"] = self.year
        if "SessionId" not in data.columns:
            data["SessionId"] = self.get_session_id()
        return data

    def get_session_id(self):
        return "Y{:04d}T{:02d}S{:01d}".format(
            self.year, self.round_number, self.session_number
        )


class SessionMetadata(DatasetLocal):
    def __init__(self, year: int, round_number: int, session_number: int):
        self.year = int(year)
        self.round_number = int(round_number)
        self.session_number = int(session_number)
        self.name = "bronze/session_metadata_{}".format(self.get_session_id())

    def run(self):
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
    def __init__(self, year: int, round_number: int, session_number: int):
        self.year = int(year)
        self.round_number = int(round_number)
        self.session_number = int(session_number)
        self.name = "bronze/session_results_{}".format(self.get_session_id())

    def run(self):
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
    def __init__(self, year: int, round_number: int, session_number: int):
        self.year = int(year)
        self.round_number = int(round_number)
        self.session_number = int(session_number)
        self.name = "bronze/session_laps_{}".format(self.get_session_id())

    def run(self):
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
    def __init__(self, year: int, round_number: int, session_number: int):
        self.year = int(year)
        self.round_number = int(round_number)
        self.session_number = int(session_number)
        self.name = "bronze/session_weather_{}".format(self.get_session_id())

    def run(self):
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
    def __init__(self, year: int, round_number: int, session_number: int):
        self.year = int(year)
        self.round_number = int(round_number)
        self.session_number = int(session_number)
        self.name = "bronze/session_track_status_{}".format(self.get_session_id())

    def run(self):
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
    def __init__(self, year: int, round_number: int, session_number: int):
        self.year = int(year)
        self.round_number = int(round_number)
        self.session_number = int(session_number)
        self.name = "bronze/session_race_control_messages_{}".format(
            self.get_session_id()
        )

    def run(self):
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
    def __init__(self, year: int, round_number: int, session_number: int):
        self.year = int(year)
        self.round_number = int(round_number)
        self.session_number = int(session_number)
        self.name = "bronze/telemetry_car_{}".format(self.get_session_id())

    def run(self):
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
                # logger.error(f"Exception in add_distance: {e}")
            try:
                this = this.add_differential_distance()
            except Exception as e:
                this = this.assign(DifferentialDistance=None)
                # logger.error(f"Exception in add_differential_distance: {e}")
            try:
                this = this.add_relative_distance()
            except Exception as e:
                this = this.assign(RelativeDistance=None)
                # logger.error(f"Exception in add_relative_distance: {e}")
            try:
                this = this.add_track_status()
            except Exception as e:
                this = this.assign(TrackStatus=None)
                # logger.error(f"Exception in add_track_status: {e}")
            try:
                this = this.add_driver_ahead()
            except Exception as e:
                this = this.assign(DriverAhead=None, DistanceToDriverAhead=None)
                # logger.error(f"Exception in add_driver_ahead: {e}")
            this = this.assign(**{c: lap[c] for c in ["DriverNumber", "LapNumber"]})
            aux.append(pd.DataFrame(this))
        data = self.add_metadata(pd.concat(aux))
        return data


class TelemetryPosData(DatasetLocal):
    def __init__(self, year: int, round_number: int, session_number: int):
        self.year = int(year)
        self.round_number = int(round_number)
        self.session_number = int(session_number)
        self.name = "bronze/telemetry_pos_{}".format(self.get_session_id())

    def run(self):
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
                # logger.error(f"Exception in add_distance: {e}")
            try:
                this = this.add_differential_distance()
            except Exception as e:
                this = this.assign(DifferentialDistance=None)
                # logger.error(f"Exception in add_differential_distance: {e}")
            try:
                this = this.add_relative_distance()
            except Exception as e:
                this = this.assign(RelativeDistance=None)
                # logger.error(f"Exception in add_relative_distance: {e}")
            try:
                this = this.add_track_status()
            except Exception as e:
                this = this.assign(TrackStatus=None)
                # logger.error(f"Exception in add_track_status: {e}")
            try:
                this = this.add_driver_ahead()
            except Exception as e:
                this = this.assign(DriverAhead=None, DistanceToDriverAhead=None)
                # logger.error(f"Exception in add_driver_ahead: {e}")
            this = this.assign(**{c: lap[c] for c in ["DriverNumber", "LapNumber"]})
            aux.append(pd.DataFrame(this))
        data = self.add_metadata(pd.concat(aux))
        return data


class OfficialSessionMetadata(OfficialSession, SessionMetadata):
    pass


class TestingSessionMetadata(TestingSession, SessionMetadata):
    pass


class OfficialSessionResults(OfficialSession, SessionResults):
    pass


class TestingSessionResults(TestingSession, SessionResults):
    pass


class OfficialSessionLaps(OfficialSession, SessionLaps):
    pass


class TestingSessionLaps(TestingSession, SessionLaps):
    pass


class OfficialSessionWeather(OfficialSession, SessionWeather):
    pass


class TestingSessionWeather(TestingSession, SessionWeather):
    pass


class OfficialSessionTrackStatus(OfficialSession, SessionTrackStatus):
    pass


class TestingSessionTrackStatus(TestingSession, SessionTrackStatus):
    pass


class OfficialSessionRaceControlMessages(OfficialSession, SessionRaceControlMessages):
    pass


class TestingSessionRaceControlMessages(TestingSession, SessionRaceControlMessages):
    pass


class OfficialTelemetryCarData(OfficialSession, TelemetryCarData):
    pass


class TestingTelemetryCarData(TestingSession, TelemetryCarData):
    pass


class OfficialTelemetryPosData(OfficialSession, TelemetryPosData):
    pass


class TestingTelemetryPosData(TestingSession, TelemetryPosData):
    pass
