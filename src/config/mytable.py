from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pandas as pd


class TableSaver:
    def __init__(self, table_name: str | Path) -> None:
        self.table_name = str(table_name)

        assert self.table_name.endswith(".csv")

    def put(self, row: dict) -> None:
        new_df = pd.DataFrame({col: [val] for col, val in row.items()})

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(self.table_name)

        if not file_exists:
            # Create new file with headers
            new_df.to_csv(self.table_name, index=False)
        else:
            # Check if we need to handle new columns
            existing_columns = set(pd.read_csv(self.table_name, nrows=0).columns)
            row_columns = set(row.keys())

            if row_columns.issubset(existing_columns):
                # No new columns, just append
                new_df.to_csv(self.table_name, mode="a", header=False, index=False)
            else:
                # New columns detected, need to rewrite the file
                existing_df = pd.read_csv(self.table_name)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv(self.table_name, index=False)

    def get(self) -> pd.DataFrame:
        return pd.read_csv(self.table_name)
