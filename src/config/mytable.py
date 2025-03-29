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
        row_columns = list(row.keys())  # Get columns from the input row, and their order

        # Check if file exists
        file_exists = os.path.exists(self.table_name)

        if not file_exists:
            # Create new file with headers using the row's columns
            new_df.to_csv(self.table_name, index=False, columns=row_columns)
        else:
            # File exists, check if structure needs update
            existing_df_header = pd.read_csv(self.table_name, nrows=0)
            existing_columns = list(existing_df_header.columns)
            
            # Check for new columns or different order if the sets of columns are identical
            existing_columns_set = set(existing_columns)
            row_columns_set = set(row.keys())
            has_new_columns = not row_columns_set.issubset(existing_columns_set)
            
            rewrite_needed = False
            if has_new_columns:
                rewrite_needed = True
            elif row_columns_set == existing_columns_set and row_columns != existing_columns:
                # Only check order if the column sets are identical
                rewrite_needed = True

            # Determine the target column order for the append step
            target_column_order = existing_columns # Default to existing order
            if rewrite_needed:
                # Define the final column order based on the row that triggered the rewrite
                target_column_order = row_columns + [col for col in existing_columns if col not in row_columns_set]

            # Perform rewrite of existing data structure if needed (BEFORE append)
            if rewrite_needed:
                print(f"Rewrite needed for {self.table_name} based on columns: {row_columns}")
                existing_df = pd.read_csv(self.table_name)
                # Reindex the *existing* DataFrame to the final order
                existing_df_rewritten = existing_df.reindex(columns=target_column_order)
                # Write the rewritten structure back, overwriting the file
                existing_df_rewritten.to_csv(self.table_name, index=False)
            
            # Always append the new row efficiently, ensuring columns match target order
            new_df_reordered = new_df.reindex(columns=target_column_order)
            new_df_reordered.to_csv(self.table_name, mode="a", header=False, index=False)


    def get(self) -> pd.DataFrame:
        return pd.read_csv(self.table_name)
