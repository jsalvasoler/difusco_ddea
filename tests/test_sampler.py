from __future__ import annotations

from pathlib import Path

import pytest
import torch
from config.configs.mis_inference import config as mis_inference_config
from config.configs.tsp_inference import config as tsp_inference_config
from config.myconfig import Config
from problems.mis.mis_dataset import MISDataset
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from torch_geometric.loader import DataLoader

from difusco.sampler import DifuscoSampler

common = Config(
    data_path="data",
    logs_path="logs",
    results_path="results",
    models_path="models",
    np_eval=True,
    device="cuda",
)


@pytest.fixture(params=[(1, 1), (3, 1), (1, 3), (3, 3)])
def config_tsp(request: tuple[int, int]) -> Config:
    """Fixture to create a Config object for the tests."""
    parallel_sampling, sequential_sampling = request.param

    config = common.update(
        task="tsp",
        test_split="tsp/tsp50_test_concorde.txt",
        training_split="tsp/tsp50_train_concorde.txt",
        validation_split="tsp/tsp50_test_concorde.txt",
        ckpt_path="tsp/tsp50_categorical.ckpt",
        parallel_sampling=parallel_sampling,
        sequential_sampling=sequential_sampling,
    )
    return tsp_inference_config.update(config)


@pytest.fixture(params=[(1, 1), (3, 1), (1, 3), (3, 3)])
def config_mis(request: tuple[int, int]) -> Config:
    """Fixture to create a Config object for MIS tests."""
    parallel_sampling, sequential_sampling = request.param

    config = common.update(
        task="mis",
        test_split="mis/er_50_100/test",
        test_split_label_dir="mis/er_50_100/test_labels",
        training_split="mis/er_50_100/train",
        training_split_label_dir="mis/er_50_100/train_labels",
        validation_split="mis/er_50_100/test",
        validation_split_label_dir="mis/er_50_100/test_labels",
        ckpt_path="mis/mis_er_50_100_gaussian.ckpt",
        parallel_sampling=parallel_sampling,
        sequential_sampling=sequential_sampling,
        sparse_factor=-1,
    )
    return mis_inference_config.update(config)


@pytest.fixture
def config_mis_recombination() -> Config:
    config = common.update(
        task="mis",
        test_split="mis/er_50_100/test",
        test_split_label_dir="mis/er_50_100/test_labels",
        training_split="mis/er_50_100/train",
        training_split_label_dir="mis/er_50_100/train_labels",
        validation_split="mis/er_50_100/test",
        validation_split_label_dir="mis/er_50_100/test_labels",
        ckpt_path="mis/mis_er_50_100_gaussian.ckpt",
        parallel_sampling=2,
        sequential_sampling=2,
        mode="recombination",
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


def assert_heatmap_properties(heatmaps: torch.Tensor, config: Config) -> None:
    """Assert properties of the heatmaps based on the configuration."""
    expected_samples = config.parallel_sampling * config.sequential_sampling
    assert heatmaps.shape[0] == expected_samples, "Incorrect number of samples"

    if config.task == "tsp":
        assert heatmaps.shape[1] == 50, "Incorrect number of nodes"
        assert heatmaps.shape[2] == 50, "Incorrect number of nodes"
    elif config.task == "mis":
        assert heatmaps.shape[1] == 56, "Incorrect number of nodes"

    # Check output range for categorical diffusion
    assert torch.all(heatmaps >= 0), "Heatmap values below 0"
    assert torch.all(heatmaps <= 1), "Heatmap values above 1"


def run_test_on_config(config: Config) -> None:
    assert torch.cuda.is_available(), "CUDA is not available"

    dataloader = get_dataloader(config)

    sampler = DifuscoSampler(config=config)

    batch = next(iter(dataloader))
    heatmaps = sampler.sample(batch)

    assert_heatmap_properties(heatmaps, config)


def test_sampler_tsp_sampling(config_tsp: Config) -> None:
    run_test_on_config(config_tsp)


def test_sampler_mis_sampling(config_mis: Config) -> None:
    run_test_on_config(config_mis)


def test_sampler_mis_recombination(config_mis_recombination: Config) -> None:
    config = config_mis_recombination
    assert torch.cuda.is_available(), "CUDA is not available"

    dataloader = get_dataloader(config)

    sampler = DifuscoSampler(config=config)

    batch = next(iter(dataloader))
    heatmaps1 = sampler.sample(batch, features=None)
    assert_heatmap_properties(heatmaps1, config)

    features = torch.randn(56, 2).bool()
    heatmaps2 = sampler.sample(batch, features=features)
    assert_heatmap_properties(heatmaps2, config)

    # assert not allclose
    assert not torch.allclose(heatmaps1, heatmaps2), "Heatmaps are not different"


@pytest.mark.parametrize(("parallel_sampling", "sequential_sampling"), [(1, 1), (3, 1), (1, 3), (3, 3)])
def test_sampler_sparse_tsp500(parallel_sampling: int, sequential_sampling: int) -> None:
    config = common.update(
        task="tsp",
        test_split="tsp/tsp500_test_concorde.txt",
        training_split="tsp/tsp500_test_concorde.txt",
        validation_split="tsp/tsp500_test_concorde.txt",
        ckpt_path="tsp/tsp500_categorical.ckpt",
        parallel_sampling=parallel_sampling,
        sequential_sampling=sequential_sampling,
        sparse_factor=50,
    )
    config = tsp_inference_config.update(config)
    dataloader = get_dataloader(config)

    sampler = DifuscoSampler(config=config)

    batch = next(iter(dataloader))
    heatmaps = sampler.sample(batch)

    assert_heatmap_properties(heatmaps, config)
