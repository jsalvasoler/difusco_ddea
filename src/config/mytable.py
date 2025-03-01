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

        if not os.path.exists(self.table_name):
            new_df.to_csv(self.table_name, index=False)
        else:
            try:
                df = pd.read_csv(self.table_name)
            except pd.errors.EmptyDataError:
                # File exists but is empty, treat it like a new file
                new_df.to_csv(self.table_name, index=False)
            else:
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(self.table_name, index=False)

    def get(self) -> pd.DataFrame:
        return pd.read_csv(self.table_name)
