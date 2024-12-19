from __future__ import annotations

from pathlib import Path

import pytest
import torch
from config.myconfig import Config
from config.configs.mis_inference import config as mis_inference_config
from config.configs.tsp_inference import config as tsp_inference_config
from problems.mis.mis_dataset import MISDataset
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from torch_geometric.loader import DataLoader

from difusco.sampler import DifuscoSampler


@pytest.fixture(params=[(1, 1), (3, 1), (1, 3), (3, 3)])
def config_tsp(request: tuple[int, int]) -> Config:
    """Fixture to create a Config object for the tests."""
    parallel_sampling, sequential_sampling = request.param
    config = Config(
        parallel_sampling=parallel_sampling,
        sequential_sampling=sequential_sampling,
    )
    return tsp_inference_config.update(config)


@pytest.fixture(params=[(1, 1), (3, 1), (1, 3), (3, 3)])
def config_mis(request: tuple[int, int]) -> Config:
    """Fixture to create a Config object for MIS tests."""
    parallel_sampling, sequential_sampling = request.param
    config = Config(
        parallel_sampling=parallel_sampling,
        sequential_sampling=sequential_sampling,
    )
    return mis_inference_config.update(config)


def get_dataloader(config: Config) -> tuple[Config, DataLoader]:
    """Fixture to create both config and dataloader for testing."""
    data_file = Path(config.data_path) / config.test_split

    if config.task == "tsp":
        dataset = TSPGraphDataset(data_file=data_file, sparse_factor=config.sparse_factor)
    elif config.task == "mis":
        dataset = MISDataset(data_dir=data_file, data_label_dir=None)

    return DataLoader(dataset, batch_size=1, shuffle=False)


def run_test_on_config(config: Config) -> None:
    assert torch.cuda.is_available(), "CUDA is not available"

    dataloader = get_dataloader(config)

    sampler = DifuscoSampler(config=config)

    batch = next(iter(dataloader))
    heatmaps = sampler.sample(batch)

    # Check output shape
    expected_samples = config.parallel_sampling * config.sequential_sampling
    assert heatmaps.shape[0] == expected_samples, "Incorrect number of samples"
    if config.task == "tsp":
        assert heatmaps.shape[1] == 50, "Incorrect number of nodes"
        assert heatmaps.shape[2] == 50, "Incorrect number of nodes"
    elif config.task == "mis":
        assert heatmaps.shape[1] == 796, "Incorrect number of nodes"

    # Check output range for categorical diffusion
    assert torch.all(heatmaps >= 0), "Heatmap values below 0"
    assert torch.all(heatmaps <= 1), "Heatmap values above 1"


def test_sampler_tsp_sampling(config_tsp: Config) -> None:
    run_test_on_config(config_tsp)


def test_sampler_mis_sampling(config_mis: Config) -> None:
    run_test_on_config(config_mis)
