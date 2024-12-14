from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ea.config import Config
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from torch_geometric.loader import DataLoader

from difusco.sampler import Sampler


@pytest.fixture(params=[(1, 1), (3, 1), (1, 3), (3, 3)])
def config(request: tuple[int, int]) -> Config:
    """Fixture to create a Config object for the tests."""
    parallel_sampling, sequential_sampling = request.param
    return Config(
        task="tsp",
        diffusion_type="categorical",
        learning_rate=0.0002,
        weight_decay=0.0001,
        lr_scheduler="cosine-decay",
        data_path="data",
        test_split="tsp/tsp50_test_concorde.txt",
        training_split="tsp/tsp50_train_concorde.txt",
        validation_split="tsp/tsp50_test_concorde.txt",
        models_path="models",
        ckpt_path="tsp/tsp50_categorical.ckpt",
        batch_size=32,
        num_epochs=50,
        diffusion_steps=2,
        validation_examples=8,
        diffusion_schedule="linear",
        inference_schedule="cosine",
        inference_diffusion_steps=50,
        device="cuda",
        parallel_sampling=parallel_sampling,
        sequential_sampling=sequential_sampling,
        sparse_factor=-1,
        n_layers=12,
        hidden_dim=256,
        aggregation="sum",
        use_activation_checkpoint=False,
        fp16=False,
    )


@pytest.fixture
def tsp_dataloader(config: Config) -> DataLoader:
    """Fixture to create a TSP dataloader for testing."""
    data_file = Path(config.data_path) / config.test_split
    dataset = TSPGraphDataset(data_file=data_file, sparse_factor=config.sparse_factor)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def test_sampler_tsp_sampling(config: Config, tsp_dataloader: DataLoader) -> None:
    assert torch.cuda.is_available(), "CUDA is not available"

    sampler = Sampler(
        task="tsp",
        config=config,
        device="cuda",
    )

    batch = next(iter(tsp_dataloader))
    heatmaps = sampler.sample(batch)

    # Check output shape
    expected_samples = config.parallel_sampling * config.sequential_sampling
    assert heatmaps.shape[0] == expected_samples, "Incorrect number of samples"
    assert heatmaps.shape[1] == 50, "Incorrect number of nodes"
    assert heatmaps.shape[2] == 50, "Incorrect number of nodes"

    # Check output range for categorical diffusion
    assert torch.all(heatmaps >= 0), "Heatmap values below 0"
    assert torch.all(heatmaps <= 1), "Heatmap values above 1"
