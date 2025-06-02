import argparse
from pathlib import Path

import fastf1

import data_engineering.gold_layer as gold_layer
import data_engineering.silver_layer as silver_layer

cache_dir = Path(__file__).parent.parent / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir=cache_dir)


def run_silver_layer(year):
    force = True
    silver_layer.YearSchedule().read(force=force)
    silver_layer.SessionMetadata(year).read(force=force)
    silver_layer.SessionResults(year).read(force=force)
    silver_layer.SessionLaps(year).read(force=force)
    silver_layer.SessionWeather(year).read(force=force)
    silver_layer.SessionTrackStatus(year).read(force=force)
    silver_layer.SessionRaceControlMessages(year).read(force=force)
    for rn in range(1, 30):
        silver_layer.TelemetryCarData(year, f"R{rn:02d}").read(force=force)
        silver_layer.TelemetryPosData(year, f"R{rn:02d}").read(force=force)
    for rn in range(1, 3):
        silver_layer.TelemetryCarData(year, f"T{rn:02d}").read(force=force)
        silver_layer.TelemetryPosData(year, f"T{rn:02d}").read(force=force)


def run_gold_layer(year):
    force = True
    gold_layer.HistoricalSessions().read(force=force)
    gold_layer.SessionResults(year).read(force=force)
    gold_layer.SessionWeather(year).read(force=force)


def main(year):
    run_silver_layer(year=year)
    run_gold_layer(year=year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    args = parser.parse_args()
    main(year=args.year)
