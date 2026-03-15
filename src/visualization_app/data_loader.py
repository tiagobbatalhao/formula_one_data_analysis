import pathlib
import sys

sys.path.append(pathlib.Path(__file__).resolve().parent.parent.as_posix())
from data_engineering.gold_layer import (CircuitMarkers, HistoricalSessions,
                                         SessionLaps, SessionMetadata,
                                         SessionResults, TelemetryCarData,
                                         TelemetryPosData)


def load_historical_sessions():
    dataset = HistoricalSessions()
    return dataset.read(force=False)


def load_session_results(year: int):
    dataset = SessionResults(year)
    return dataset.read(force=False)


def load_session_metadata(year: int):
    dataset = SessionMetadata(year)
    return dataset.read(force=False)


def load_session_laps(year: int):
    dataset = SessionLaps(year)
    return dataset.read(force=False)


def load_telemetry_car(year: int, round_id: str):
    dataset = TelemetryCarData(year, round_id)
    return dataset.read(force=False)


def load_telemetry_pos(year: int, round_id: str):
    dataset = TelemetryPosData(year, round_id)
    return dataset.read(force=False)


def load_circuit_markers(year: int):
    dataset = CircuitMarkers(year)
    return dataset.read(force=False)
