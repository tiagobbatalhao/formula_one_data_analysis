import pathlib
import sys

sys.path.append(pathlib.Path(__file__).resolve().parent.parent.as_posix())
from data_engineering.gold_layer import (HistoricalSessions, SessionLaps,
                                         SessionMetadata, SessionResults)


def load_historical_sessions():
    dataset = HistoricalSessions()
    return dataset.read(force=True)


def load_session_results(year: int):
    dataset = SessionResults(year)
    return dataset.read(force=True)


def load_session_metadata(year: int):
    dataset = SessionMetadata(year)
    return dataset.read(force=True)


def load_session_laps(year: int):
    dataset = SessionLaps(year)
    return dataset.read(force=True)
