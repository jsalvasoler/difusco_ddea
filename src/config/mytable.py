import os

import pandas as pd


class TableSaver:
    def __init__(self, table_name: str) -> None:
        self.table_name = table_name

        assert self.table_name.endswith(".csv")

    def put(self, row: dict) -> None:
        new_df = pd.DataFrame({col: [val] for col, val in row.items()})

        if not os.path.exists(self.table_name):
            new_df.to_csv(self.table_name, index=False)
        else:
            df = pd.read_csv(self.table_name)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(self.table_name, index=False)

    def get(self) -> pd.DataFrame:
        return pd.read_csv(self.table_name)
