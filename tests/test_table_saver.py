import os
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from src.config.mytable import TableSaver


@pytest.fixture
def temp_csv_file(tmp_path: Generator[Path, None, None]) -> str:
    """Create a temporary CSV file path for testing"""
    return str(tmp_path / "test_table.csv")


def test_init_validates_csv_extension() -> None:
    """Test that initialization requires .csv extension"""
    with pytest.raises(AssertionError):
        TableSaver("invalid_name.txt")


def test_put_creates_new_file(temp_csv_file: str) -> None:
    """Test putting a row into a new file"""
    saver = TableSaver(temp_csv_file)
    test_row = {"name": "John", "age": 30}

    saver.put(test_row)

    assert os.path.exists(temp_csv_file)
    df = pd.read_csv(temp_csv_file)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "John"
    assert df.iloc[0]["age"] == 30


def test_put_appends_to_existing_file(temp_csv_file: str) -> None:
    """Test putting multiple rows appends them correctly"""
    saver = TableSaver(temp_csv_file)

    # Put first row
    saver.put({"name": "John", "age": 30})

    # Put second row
    saver.put({"name": "Jane", "age": 25})

    df = pd.read_csv(temp_csv_file)
    assert len(df) == 2
    assert df.iloc[0]["name"] == "John"
    assert df.iloc[1]["name"] == "Jane"


def test_get_returns_dataframe(temp_csv_file: str) -> None:
    """Test getting data returns correct DataFrame"""
    saver = TableSaver(temp_csv_file)
    test_rows = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]

    for row in test_rows:
        saver.put(row)

    df = saver.get()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["name", "age"]
    assert df.iloc[0]["name"] == "John"
    assert df.iloc[1]["name"] == "Jane"


def test_put_new_field_creates_new_column(temp_csv_file: str) -> None:
    """Test putting a new field creates a new column"""
    saver = TableSaver(temp_csv_file)
    saver.put({"name": "John", "age": 30, "city": "New York"})

    saver.put({"surname": "Doe", "age": 25})

    df = saver.get()
    assert set(df.columns) == {"name", "age", "city", "surname"}
    assert df.iloc[-1]["surname"] == "Doe"
    assert pd.isna(df.iloc[-1]["name"])


def test_put_reordered_columns_preserves_data(temp_csv_file: str) -> None:
    """Test that putting columns in a different order preserves the data correctly"""
    saver = TableSaver(temp_csv_file)
    
    # First row with columns in order a, b, c
    saver.put({"a": 1, "b": 2, "c": 3})
    
    # Second row with columns in different order a, c, b
    saver.put({"a": 4, "c": 6, "b": 5})
    
    df = saver.get()
    assert len(df) == 2
    assert list(df.columns) == ["a", "c", "b"]
    
    # First row data preserved and reordered
    assert df.iloc[0]["a"] == 1
    assert df.iloc[0]["c"] == 3
    assert df.iloc[0]["b"] == 2
    
    # Second row data correctly mapped despite reordering
    assert df.iloc[1]["a"] == 4
    assert df.iloc[1]["c"] == 6
    assert df.iloc[1]["b"] == 5


def test_multiple_new_columns(temp_csv_file: str) -> None:
    """Test adding multiple new columns sequentially."""
    saver = TableSaver(temp_csv_file)
    saver.put({"a": 1, "b": 2})          # Initial: [a, b]
    saver.put({"a": 3, "c": 4})          # Add c: [a, c, b]
    saver.put({"b": 5, "d": 6})          # Add d: [b, d, a, c]

    df = saver.get()
    assert len(df) == 3
    assert list(df.columns) == ["b", "d", "a", "c"] # Final order based on last rewrite

    # Check data integrity
    assert df.iloc[0]["a"] == 1
    assert df.iloc[0]["b"] == 2
    assert pd.isna(df.iloc[0]["c"])
    assert pd.isna(df.iloc[0]["d"])

    assert df.iloc[1]["a"] == 3
    assert pd.isna(df.iloc[1]["b"])
    assert df.iloc[1]["c"] == 4
    assert pd.isna(df.iloc[1]["d"])

    assert pd.isna(df.iloc[2]["a"])
    assert df.iloc[2]["b"] == 5
    assert pd.isna(df.iloc[2]["c"])
    assert df.iloc[2]["d"] == 6


def test_add_then_reorder(temp_csv_file: str) -> None:
    """Test adding a column then reordering."""
    saver = TableSaver(temp_csv_file)
    saver.put({"a": 1, "b": 2})          # Initial: [a, b]
    saver.put({"a": 3, "c": 4})          # Add c: [a, c, b]
    saver.put({"c": 6, "a": 5, "b": 7}) # Reorder: [c, a, b]

    df = saver.get()
    assert len(df) == 3
    assert list(df.columns) == ["c", "a", "b"] # Final order based on last rewrite

    # Check data
    assert df.iloc[0]["a"] == 1
    assert df.iloc[0]["b"] == 2
    assert pd.isna(df.iloc[0]["c"])
    
    assert df.iloc[1]["a"] == 3
    assert pd.isna(df.iloc[1]["b"])
    assert df.iloc[1]["c"] == 4
    
    assert df.iloc[2]["a"] == 5
    assert df.iloc[2]["b"] == 7
    assert df.iloc[2]["c"] == 6
    

def test_reorder_then_add(temp_csv_file: str) -> None:
    """Test reordering then adding a column."""
    saver = TableSaver(temp_csv_file)
    saver.put({"a": 1, "b": 2})          # Initial: [a, b]
    saver.put({"b": 4, "a": 3})          # Reorder: [b, a]
    saver.put({"c": 5, "a": 6})          # Add c: [c, a, b]

    df = saver.get()
    assert len(df) == 3
    assert list(df.columns) == ["c", "a", "b"]

    # Check data
    assert df.iloc[0]["a"] == 1
    assert df.iloc[0]["b"] == 2
    assert pd.isna(df.iloc[0]["c"])
    
    assert df.iloc[1]["a"] == 3
    assert df.iloc[1]["b"] == 4
    assert pd.isna(df.iloc[1]["c"])

    assert df.iloc[2]["a"] == 6
    assert pd.isna(df.iloc[2]["b"])
    assert df.iloc[2]["c"] == 5


def test_subset_after_rewrite(temp_csv_file: str) -> None:
    """Test adding a subset of columns after a rewrite doesn't cause another rewrite."""
    saver = TableSaver(temp_csv_file)
    saver.put({"a": 1, "b": 2})          # Initial: [a, b]
    saver.put({"c": 4, "a": 3})          # Rewrite: [c, a, b]
    saver.put({"a": 5})                  # Append subset: should not rewrite
    saver.put({"b": 6, "c": 7})          # Append subset: should not rewrite

    df = saver.get()
    assert len(df) == 4
    assert list(df.columns) == ["c", "a", "b"] # Order from the rewrite

    # Check data
    assert df.iloc[0]["a"] == 1; assert df.iloc[0]["b"] == 2; assert pd.isna(df.iloc[0]["c"])
    assert df.iloc[1]["a"] == 3; assert pd.isna(df.iloc[1]["b"]); assert df.iloc[1]["c"] == 4
    assert df.iloc[2]["a"] == 5; assert pd.isna(df.iloc[2]["b"]); assert pd.isna(df.iloc[2]["c"])
    assert pd.isna(df.iloc[3]["a"]); assert df.iloc[3]["b"] == 6; assert df.iloc[3]["c"] == 7


def test_complex_sequence_of_rewrites(temp_csv_file: str) -> None:
    """Test a more complex sequence of puts involving adds and reorders."""
    saver = TableSaver(temp_csv_file)
    saver.put({"x": 1, "y": 2})          # Initial: [x, y]
    saver.put({"x": 3, "z": 4})          # Add z: [x, z, y]
    saver.put({"y": 6, "x": 5})          # Reorder: [y, x, z]
    saver.put({"a": 7, "x": 8})          # Add a: [a, x, y, z]
    saver.put({"z": 10, "y": 9})         # Append subset
    saver.put({"x": 11, "y": 12, "z": 13, "a": 14}) # Reorder: [x, y, z, a]
    saver.put({"b": 15, "x": 16})         # Add b: [b, x, y, z, a]

    df = saver.get()
    assert len(df) == 7
    assert list(df.columns) == ["b", "x", "y", "z", "a"]

    # Spot check some data
    # Row 0 (Initial)
    assert df.iloc[0]["x"] == 1; assert df.iloc[0]["y"] == 2; assert pd.isna(df.iloc[0]["z"]); assert pd.isna(df.iloc[0]["a"]); assert pd.isna(df.iloc[0]["b"])
    # Row 1 (Add z)
    assert df.iloc[1]["x"] == 3; assert pd.isna(df.iloc[1]["y"]); assert df.iloc[1]["z"] == 4; assert pd.isna(df.iloc[1]["a"]); assert pd.isna(df.iloc[1]["b"])
    # Row 2 (Reorder y,x)
    assert df.iloc[2]["x"] == 5; assert df.iloc[2]["y"] == 6; assert pd.isna(df.iloc[2]["z"]); assert pd.isna(df.iloc[2]["a"]); assert pd.isna(df.iloc[2]["b"])
    # Row 3 (Add a)
    assert df.iloc[3]["x"] == 8; assert pd.isna(df.iloc[3]["y"]); assert pd.isna(df.iloc[3]["z"]); assert df.iloc[3]["a"] == 7; assert pd.isna(df.iloc[3]["b"])
    # Row 4 (Subset z,y)
    assert pd.isna(df.iloc[4]["x"]); assert df.iloc[4]["y"] == 9; assert df.iloc[4]["z"] == 10; assert pd.isna(df.iloc[4]["a"]); assert pd.isna(df.iloc[4]["b"])
    # Row 5 (Reorder x,y,z,a)
    assert df.iloc[5]["x"] == 11; assert df.iloc[5]["y"] == 12; assert df.iloc[5]["z"] == 13; assert df.iloc[5]["a"] == 14; assert pd.isna(df.iloc[5]["b"])
    # Row 6 (Add b)
    assert df.iloc[6]["x"] == 16; assert pd.isna(df.iloc[6]["y"]); assert pd.isna(df.iloc[6]["z"]); assert pd.isna(df.iloc[6]["a"]); assert df.iloc[6]["b"] == 15
