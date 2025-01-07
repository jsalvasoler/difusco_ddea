from typing import Generator

import numpy as np
import pytest
import torch
from config.myconfig import Config
from ea.ea_arg_parser import get_arg_parser
from ea.ea_utils import dataset_factory, filter_args_by_group, get_results_dict, instance_factory
from ea.evolutionary_algorithm import (
    handle_empty_queue,
    handle_process_error,
    handle_timeout,
    process_iteration,
    run_ea,
)
from problems.mis.mis_dataset import MISDataset
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from torch_geometric.loader import DataLoader


@pytest.fixture
def temp_dir(tmp_path: str) -> Generator[any, any, any]:
    """Fixture to create a temporary directory for the tests."""
    return tmp_path
    # Cleanup is handled by pytest


@pytest.fixture
def config(temp_dir: Generator) -> Config:
    """Fixture to create a Config object for the tests."""
    return Config(
        task="mis",
        algo="brkga",
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


def test_filter_args_by_group() -> None:
    parser = get_arg_parser()
    ea_settings_args = filter_args_by_group(parser, "ea_settings")
    expected_args = ["device", "n_parallel_evals", "pop_size", "n_generations", "max_two_opt_it", "initialization"]
    assert set(ea_settings_args) == set(expected_args)


def test_get_results_dict(config: Config) -> None:
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
        task="mis",
        algo="brkga",
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


@pytest.mark.parametrize("task", ["tsp", "mis"])
@pytest.mark.parametrize("algo", ["ga", "brkga"])
def test_gpu_memory_cleanup(task: str, algo: str) -> None:
    import multiprocessing as mp

    from tqdm import tqdm

    if task == "tsp":
        dataset = TSPGraphDataset(data_file="data/tsp/tsp100_test_concorde.txt", sparse_factor=-1)
    elif task == "mis":
        dataset = MISDataset(data_dir="data/mis/er_700_800/test")
    else:
        error_msg = f"Invalid task: {task}"
        raise ValueError(error_msg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    config = Config(
        pop_size=2,
        device="cuda",
        n_parallel_evals=0,
        max_two_opt_it=10,
        validate_samples=3,
        task=task,
        algo=algo,
        sparse_factor=-1,
        n_generations=2,
        np_eval=True,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    is_validation_run = config.validate_samples is not None
    results = []

    results = []
    ctx = mp.get_context("spawn")

    for i, sample in tqdm(enumerate(dataloader)):
        queue = ctx.Queue()
        process = ctx.Process(target=process_iteration, args=(config, sample, queue))

        process.start()
        try:
            process.join(timeout=30 * 60)  # 30 minutes timeout
            handle_timeout(process, i)
            handle_empty_queue(queue)

            run_results = queue.get()
            handle_process_error(run_results)

            results.append(run_results)
        except (TimeoutError, RuntimeError) as e:
            print(f"Error in iteration {i}: {e}")
        finally:
            if process.is_alive():
                process.terminate()
                process.join()

        if is_validation_run and i >= config.validate_samples - 1:
            break

        assert torch.cuda.memory_allocated() == 0, "CUDA memory not freed"

    _ = {
        "avg_cost": np.mean([r["cost"] for r in results]),
        "avg_gt_cost": np.mean([r["gt_cost"] for r in results]),
        "avg_gap": np.mean([r["gap"] for r in results]),
        "avg_runtime": np.mean([r["runtime"] for r in results]),
        "n_evals": len(results),
    }


@pytest.mark.parametrize("task", ["tsp", "mis"])
@pytest.mark.parametrize("algo", ["ga", "brkga"])
def test_ea_runs(task: str, algo: str) -> None:
    if task == "tsp":
        data_path = "data/tsp/tsp50_test_concorde.txt"
    elif task == "mis":
        data_path = "data/mis/er_700_800/test"
    else:
        error_msg = f"Invalid task: {task}"
        raise ValueError(error_msg)
    config = Config(
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
        np_eval=True,
        validate_samples=2,
    )
    run_ea(config)


def test_ea_for_sparse_tsp() -> None:
    config = Config(
        task="tsp",
        algo="ga",
        sparse_factor=50,
        n_generations=10,
        pop_size=10,
        n_parallel_evals=0,
        max_two_opt_it=1,
        np_eval=True,
        test_split="data/tsp/tsp500_test_concorde.txt",
        test_split_label_dir=None,
        data_path=".",
        device="cuda",
        initialization="random_feasible",
        validate_samples=2,
    )
    run_ea(config)
