import argparse
from pathlib import Path

import fastf1

import data_engineering.bronze_layer as bronze_layer

cache_dir = Path(__file__).parent.parent / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir=cache_dir)


def download_official_session(year, round_number, session_number, force):
    args = (year, round_number, session_number)
    bronze_layer.OfficialSessionMetadata(*args).read(force=force)
    bronze_layer.OfficialSessionResults(*args).read(force=force)
    bronze_layer.OfficialSessionLaps(*args).read(force=force)
    bronze_layer.OfficialSessionWeather(*args).read(force=force)
    bronze_layer.OfficialSessionTrackStatus(*args).read(force=force)
    bronze_layer.OfficialSessionRaceControlMessages(*args).read(force=force)
    bronze_layer.OfficialTelemetryCarData(*args).read(force=force)
    bronze_layer.OfficialTelemetryPosData(*args).read(force=force)
    bronze_layer.OfficialCircuitMarkers(*args).read(force=force)


def download_testing_session(year, round_number, session_number, force):
    args = (year, round_number, session_number)
    bronze_layer.TestingSessionMetadata(*args).read(force=force)
    bronze_layer.TestingSessionResults(*args).read(force=force)
    bronze_layer.TestingSessionLaps(*args).read(force=force)
    bronze_layer.TestingSessionWeather(*args).read(force=force)
    bronze_layer.TestingSessionTrackStatus(*args).read(force=force)
    bronze_layer.TestingSessionRaceControlMessages(*args).read(force=force)
    bronze_layer.TestingTelemetryCarData(*args).read(force=force)
    bronze_layer.TestingTelemetryPosData(*args).read(force=force)
    bronze_layer.TestingCircuitMarkers(*args).read(force=force)


def main(year, round_id, force):
    round_type = round_id[0].upper()
    assert round_type in ["R", "T"], ""
    round_number = int(round_id[1:])
    if round_type == "R":
        for session in range(1, 6):
            download_official_session(
                year=year,
                round_number=round_number,
                session_number=session,
                force=force,
            )
    else:
        for session in range(1, 6):
            download_testing_session(
                year=year,
                round_number=round_number,
                session_number=session,
                force=force,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    parser.add_argument("round_id", type=str)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(year=args.year, round_id=args.round_id, force=args.force)
