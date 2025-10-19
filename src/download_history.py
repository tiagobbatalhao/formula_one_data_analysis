import argparse
from datetime import datetime

import data_engineering.bronze_layer as bronze_layer
import data_engineering.gold_layer as gold_layer
import data_engineering.silver_layer as silver_layer


def main(year_start, year_end, force):
    for year in range(year_end, year_start - 1, -1):
        bronze_layer.YearSchedule(year).read(force=force)
    silver_layer.YearSchedule().read(force=force)
    gold_layer.HistoricalSessions().read(force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    current_year = datetime.now().year

    parser.add_argument("--year_start", type=int, default=1950)
    parser.add_argument("--year_end", type=int, default=current_year)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(year_start=args.year_start, year_end=args.year_end, force=args.force)
