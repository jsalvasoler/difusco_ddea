"""Lightning module for training the DIFUSCO MIS model."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix

from difusco.mis.pl_mis_base_model import MISModelBase
from difuscombination.dataset import MISDatasetComb
from difuscombination.gnn_encoder_difuscombination import GNNEncoderDifuscombination

if TYPE_CHECKING:
    from config.myconfig import Config


class DifusCombinationMISModel(MISModelBase):
    def __init__(self, config: Config | None = None) -> None:
        super().__init__(param_args=config)

        self.train_dataset = MISDatasetComb(
            samples_file=os.path.join(config.data_path, config.training_samples_file),
            graphs_dir=os.path.join(config.data_path, config.training_graphs_dir),
            labels_dir=os.path.join(config.data_path, config.training_labels_dir),
        )

        self.test_dataset = MISDatasetComb(
            samples_file=os.path.join(config.data_path, config.test_samples_file),
            graphs_dir=os.path.join(config.data_path, config.test_graphs_dir),
            labels_dir=os.path.join(config.data_path, config.test_labels_dir),
        )

        self.validation_dataset = MISDatasetComb(
            samples_file=os.path.join(config.data_path, config.validation_samples_file),
            graphs_dir=os.path.join(config.data_path, config.validation_graphs_dir),
            labels_dir=os.path.join(config.data_path, config.validation_labels_dir),
        )

        self.config = config

        # Override self.model
        config.node_feature_only = True
        self.model = GNNEncoderDifuscombination(config)

    @staticmethod
    def process_batch(batch: tuple) -> tuple:
        _, graph_data, _ = batch
        node_labels = graph_data.x[:, 0]
        features = graph_data.x[:, 1:]
        edge_index = graph_data.edge_index

        edge_index = edge_index.to(node_labels.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        ).tocsr()

        return node_labels, edge_index, adj_mat, features

    @staticmethod
    def unpack_batch(batch: tuple) -> tuple:
        _, graph_data, point_indicator = batch
        node_labels = graph_data.x[:, 0]
        features = graph_data.x[:, 1:]
        edge_index = graph_data.edge_index
        return node_labels, edge_index, point_indicator, features
