from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from difuscombination.gnn_encoder_difuscombination import GNNEncoderDifuscombination
from problems.mis.mis_dataset import MISDataset

if TYPE_CHECKING:
    from torch_geometric.data import Data as GraphData

from difusco.gnn_encoder import GNNEncoder


@pytest.fixture
def mis_sample() -> tuple[torch.Tensor, GraphData, torch.Tensor]:
    # load er_50_100
    dataset = MISDataset(data_dir="data/mis/er_50_100/test", data_label_dir="data/mis/er_50_100/test_labels")
    # Get a sample
    return dataset.__getitem__(0)


@pytest.fixture
def gnn_model() -> GNNEncoder:
    # take configuration from mis_inference
    from config.configs.mis_inference import config as mis_inf_config

    config = mis_inf_config.update(
        task="mis",
        device="cuda",
    )

    return GNNEncoder(
        n_layers=config.n_layers,
        hidden_dim=config.hidden_dim,
        out_channels=1,  # For Gaussian diffusion
        aggregation=config.aggregation,
        sparse=True,  # Use sparse version since it is MIS
        use_activation_checkpoint=config.use_activation_checkpoint,
        node_feature_only=True,  # For MIS we only use node features
    )


def get_random_difuscombination_x_sample(graph_data: GraphData) -> torch.tensor:
    # Stack two more features in the node feature dimension -> we expect failure
    x = torch.randn_like(graph_data.x, dtype=torch.float32)
    features = torch.cat(
        [
            x.unsqueeze(1),
            torch.randn_like(x.unsqueeze(1), dtype=torch.float32),
            torch.randn_like(x.unsqueeze(1), dtype=torch.float32),
        ],
        dim=1,
    )
    return x, features

def test_gnn_encoder_mis_inference(
    mis_sample: tuple[torch.Tensor, GraphData, torch.Tensor], gnn_model: GNNEncoder
) -> None:
    _, graph_data, _ = mis_sample

    # Create fake timesteps for testing
    batch_size = 1
    timesteps = torch.ones((batch_size,), dtype=torch.float32)

    # Run inference
    x = torch.randn_like(graph_data.x, dtype=torch.float32)
    edge_index = graph_data.edge_index

    # Forward pass
    output = gnn_model(x, timesteps, edge_index=edge_index)

    # Basic shape checks
    assert output.shape[0] == x.shape[0], "Output should have same number of nodes as input"
    assert output.shape[1] == 1, "Output should have 1 channel for Gaussian diffusion"


def test_gnn_encoder_difuscombination_mis(mis_sample: tuple[torch.Tensor, GraphData, torch.Tensor]) -> None:
    from config.configs.mis_inference import config as mis_inf_config

    config = mis_inf_config.update(
        task="mis",
        device="cuda",
        diffusion_type="gaussian",
        node_feature_only=True,
    )

    gnn_model = GNNEncoderDifuscombination(config)
    _, graph_data, _ = mis_sample

    timesteps = torch.ones((1,), dtype=torch.float32)
    x, features = get_random_difuscombination_x_sample(graph_data)
    _ = gnn_model(x, timesteps, features, edge_index=graph_data.edge_index)
