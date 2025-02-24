from typing import Generator

import numpy as np
import pytest
from config.myconfig import Config
from ea.ea_arg_parser import get_arg_parser
from ea.ea_utils import dataset_factory, filter_args_by_group, get_results_dict, instance_factory
from ea.evolutionary_algorithm import (
    run_ea,
)
from torch_geometric.loader import DataLoader


@pytest.fixture
def temp_dir(tmp_path: str) -> Generator[any, any, any]:
    """Fixture to create a temporary directory for the tests."""
    return tmp_path
    # Cleanup is handled by pytest


def test_filter_args_by_group() -> None:
    parser = get_arg_parser()
    ea_settings_args = filter_args_by_group(parser, "ea_settings")
    expected_args = [
        "device",
        "n_parallel_evals",
        "pop_size",
        "n_generations",
        "max_two_opt_it",
        "initialization",
        "config_name",
    ]
    assert set(ea_settings_args) == set(expected_args)


def test_get_results_dict() -> None:
    config = Config(
        config_name="mis_inference",
        task="mis",
        algo="ga",
        wandb_logger_name="test_logger",
        results_path=str(temp_dir),
        device="cpu",
        n_parallel_evals=0,
        pop_size=100,
        n_generations=100,
        max_two_opt_it=200,
        sparse_factor=-1,
        np_eval=False,
        initialization="difusco_sampling",
    )
    results = {"a": 0.95, "b": 0.05}

    results_dict = get_results_dict(config, results)

    assert results_dict["task"] == "mis"
    assert results_dict["wandb_logger_name"] == "test_logger"
    assert results_dict["device"] == "cpu"
    assert results_dict["n_parallel_evals"] == 0
    assert results_dict["pop_size"] == 100
    assert results_dict["a"] == 0.95
    assert results_dict["b"] == 0.05
    assert "timestamp" in results_dict


def test_mis_gt_avg_cost_er_test_set() -> None:
    config = Config(
        config_name="mis_inference",
        task="mis",
        algo="ga",
        data_path="data",
        test_split="mis/er_700_800/test",
        test_split_label_dir=None,  # er_700_800/test already has labels!
        device="cpu",
        np_eval=True,
    )

    dataset = dataset_factory(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []

    for sample in dataloader:
        instance = instance_factory(config, sample)
        gt_cost = instance.get_gt_cost()
        results.append(gt_cost)

    assert np.mean(results) == 41.3828125


@pytest.mark.parametrize("task", ["mis"])  # tsp unsupported currently
@pytest.mark.parametrize("algo", ["ga"])
def test_ea_runs(task: str, algo: str) -> None:
    if task == "tsp":
        data_path = "data/tsp/tsp50_test_concorde.txt"
    elif task == "mis":
        data_path = "data/mis/er_700_800/test"
    else:
        error_msg = f"Invalid task: {task}"
        raise ValueError(error_msg)
    config = Config(
        config_name=f"{task}_inference",
        initialization="random_feasible",
        recombination="classic",
        test_split=data_path,
        test_split_label_dir=None,
        data_path=".",
        pop_size=5,
        device="cpu",
        n_parallel_evals=0,
        max_two_opt_it=1,
        task=task,
        algo=algo,
        sparse_factor=-1,
        n_generations=5,
        validate_samples=2,
        profiler=False,
    )
    run_ea(config)


def test_ea_for_sparse_tsp() -> None:
    config = Config(
        config_name="tsp_inference",
        task="tsp",
        algo="ga",
        sparse_factor=50,
        n_generations=2,
        pop_size=2,
        n_parallel_evals=0,
        max_two_opt_it=1,
        np_eval=True,
        test_split="data/tsp/tsp500_test_concorde.txt",
        test_split_label_dir=None,
        data_path=".",
        device="cuda",
        initialization="random_feasible",
        validate_samples=2,
        profiler=False,
    )
    run_ea(config)
