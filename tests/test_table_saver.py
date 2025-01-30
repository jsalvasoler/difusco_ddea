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
