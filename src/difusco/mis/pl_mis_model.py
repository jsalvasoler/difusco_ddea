"""Lightning module for training the DIFUSCO MIS model."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from problems.mis.mis_dataset import MISDataset
from scipy.sparse import coo_matrix

from difusco.mis.pl_mis_base_model import MISModelBase

if TYPE_CHECKING:
    from argparse import Namespace


class MISModel(MISModelBase):
    def __init__(self, param_args: Namespace | None = None) -> None:
        super().__init__(param_args=param_args)

        train_label_dir, test_label_dir, validation_label_dir = None, None, None

        if self.args.training_split_label_dir is not None:
            train_label_dir = os.path.join(self.args.data_path, self.args.training_split_label_dir)
        if self.args.test_split_label_dir is not None:
            test_label_dir = os.path.join(self.args.data_path, self.args.test_split_label_dir)
        if self.args.validation_split_label_dir is not None:
            validation_label_dir = os.path.join(self.args.data_path, self.args.validation_split_label_dir)

        self.train_dataset = MISDataset(
            data_dir=os.path.join(self.args.data_path, self.args.training_split),
            data_label_dir=train_label_dir,
        )

        self.test_dataset = MISDataset(
            data_dir=os.path.join(self.args.data_path, self.args.test_split),
            data_label_dir=test_label_dir,
        )

        self.validation_dataset = MISDataset(
            data_dir=os.path.join(self.args.data_path, self.args.validation_split),
            data_label_dir=validation_label_dir,
        )

    @staticmethod
    def process_batch(batch: tuple) -> tuple:
        _, graph_data, _ = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index

        edge_index = edge_index.to(node_labels.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        ).tocsr()

        return node_labels, edge_index, adj_mat, None

    @staticmethod
    def unpack_batch(batch: tuple) -> tuple:
        _, graph_data, point_indicator = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index

        return node_labels, edge_index, point_indicator, None
