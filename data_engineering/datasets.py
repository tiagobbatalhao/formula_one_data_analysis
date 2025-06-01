from pathlib import Path

import pandas as pd


class DatasetLocal:
    BASE_FOLDER = Path(__file__).resolve().parent.parent / "data"

    def __init__(self, name):
        self.name = name

    def read(self, force=False):
        path = self.BASE_FOLDER / (self.name + ".parquet")
        if (force) or (not path.exists()):
            saved = self.save()
            if not saved:
                return None
        return pd.read_parquet(path)

    def save(self):
        path = self.BASE_FOLDER / (self.name + ".parquet")
        df = self.run()
        if df is None:
            return False
        if len(df) == 0:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return True

    def read_with_pattern(self):
        files = self.BASE_FOLDER.rglob(self.name.rstrip("*") + "*.parquet")
        for fl in files:
            yield pd.read_parquet(fl)
