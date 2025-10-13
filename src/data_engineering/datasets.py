from pathlib import Path

import pandas as pd


class DatasetLocal:
    """
    Base class for local dataset management in the medallion architecture.

    Handles reading, saving, and pattern-based loading of Parquet files
    for each dataset entity in the bronze, silver, and gold layers.
    """

    BASE_FOLDER: Path = Path(__file__).resolve().parent.parent.parent / "data"

    def __init__(self, name: str):
        """
        Initialize a DatasetLocal instance.

        Args:
            name (str): The dataset name, used as the file prefix.
        """
        self.name: str = name

    def read(self, force: bool = False) -> pd.DataFrame | None:
        """
        Read the dataset from a Parquet file. If the file does not exist or force is True,
        attempts to generate and save the dataset by calling self.save().

        Args:
            force (bool): If True, regenerate the dataset even if the file exists.

        Returns:
            pd.DataFrame | None: The loaded DataFrame, or None if loading failed.
        """
        path = self.BASE_FOLDER / (self.name + ".parquet")
        if (force) or (not path.exists()):
            saved = self.save()
            if not saved:
                return None
        return pd.read_parquet(path)

    def save(self) -> bool:
        """
        Generate and save the dataset to a Parquet file by calling self.run().

        Returns:
            bool: True if the dataset was saved successfully, False otherwise.
        """
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
        """
        Yield DataFrames for all Parquet files matching the dataset name pattern.

        Yields:
            pd.DataFrame: DataFrames loaded from matching Parquet files.
        """
        files = self.BASE_FOLDER.rglob(self.name.rstrip("*") + "*.parquet")
        for fl in files:
            yield pd.read_parquet(fl)
