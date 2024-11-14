import os
from typing import Generator

import pandas as pd
import pytest
from ea.arg_parser import get_arg_parser
from ea.config import Config
from ea.ea_utils import filter_args_by_group, save_results


@pytest.fixture
def temp_dir(tmp_path: str) -> Generator[any, any, any]:
    """Fixture to create a temporary directory for the tests."""
    return tmp_path
    # Cleanup is handled by pytest


def test_filter_args_by_group() -> None:
    parser = get_arg_parser()
    ea_settings_args = filter_args_by_group(parser, "ea_settings")
    expected_args = ["device", "n_parallel_evals", "pop_size", "n_generations"]
    assert set(ea_settings_args) == set(expected_args)


def test_save_results_file_does_not_exist(temp_dir: Generator) -> None:
    config = Config(
        task="mis",
        wandb_logger_name="test_logger",
        results_path=str(temp_dir),
        device="cpu",
        n_parallel_evals=0,
        pop_size=100,
        n_generations=100,
    )
    results = {"a": 0.95, "b": 0.05}

    save_results(config, results)

    results_file = os.path.join(config.results_path, "ea_results.csv")
    assert os.path.exists(results_file)

    df = pd.read_csv(results_file)
    assert len(df) == 1
    assert df["task"].iloc[0] == "mis"
    assert df["wandb_logger_name"].iloc[0] == "test_logger"
    assert df["device"].iloc[0] == "cpu"
    assert df["n_parallel_evals"].iloc[0] == 0
    assert df["pop_size"].iloc[0] == 100
    assert df["n_generations"].iloc[0] == 100
    assert df["a"].iloc[0] == 0.95
    assert df["b"].iloc[0] == 0.05
    assert "timestamp" in df.columns


def test_save_results_file_exists(temp_dir: Generator) -> None:
    config = Config(
        task="mis",
        wandb_logger_name="test_logger",
        results_path=str(temp_dir),
        device="cpu",
        n_parallel_evals=0,
        pop_size=100,
        n_generations=100,
    )
    results = {"a": 0.95, "b": 0.05}

    # Create an initial results file
    initial_data = {"task": ["mis"], "wandb_logger_name": ["initial_logger"], "accuracy": [0.90], "loss": [0.10]}
    initial_df = pd.DataFrame(initial_data)
    results_file = os.path.join(config.results_path, "ea_results.csv")
    initial_df.to_csv(results_file, index=False)

    save_results(config, results)

    df = pd.read_csv(results_file)
    assert len(df) == 2
    assert df["task"].iloc[1] == "mis"
    assert df["wandb_logger_name"].iloc[1] == "test_logger"
    assert df["device"].iloc[1] == "cpu"
    assert df["n_parallel_evals"].iloc[1] == 0
    assert df["pop_size"].iloc[1] == 100
    assert df["n_generations"].iloc[1] == 100
    assert df["a"].iloc[1] == 0.95
    assert df["b"].iloc[1] == 0.05
    assert "timestamp" in df.columns
