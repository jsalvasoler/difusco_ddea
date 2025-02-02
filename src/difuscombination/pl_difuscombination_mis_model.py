"""Lightning module for training the DIFUSCO MIS model."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from problems.mis.mis_evaluation import mis_decode_np
from scipy.sparse import coo_matrix
from torch import nn
from torch.nn.functional import mse_loss, one_hot

from difusco.mis.pl_mis_model import MISModel
from difuscombination.dataset import MISDatasetComb
from difuscombination.gnn_encoder_difuscombination import GNNEncoderDifuscombination

if TYPE_CHECKING:
    from config.myconfig import Config


class DifusCombinationMISModel(MISModel):
    # TODO: extract the common code to some MIS Meta model, and then have a Difusco and Recombination chilfren
    def __init__(self, config: Config | None = None) -> None:
        # we need to update the config with fake datasets that allow the initialization of the MISModel
        config = config.update(
            training_split=config.training_graphs_dir,
            training_split_label_dir=None,
            test_split=config.test_graphs_dir,
            test_split_label_dir=None,
            validation_split=config.validation_graphs_dir,
            validation_split_label_dir=None,
        )

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
        self.model = GNNEncoderDifuscombination(config)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        return self.model(x, t, features, edge_index=edge_index)

    def categorical_training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        _, graph_data, point_indicator = batch
        t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
        node_labels = graph_data.x[:, 0]  # first dim are the labels
        features = graph_data.x[:, 1:]  # rest of the dims are the features
        edge_index = graph_data.edge_index

        # Sample from diffusion
        node_labels_onehot = one_hot(node_labels.long(), num_classes=2).float()
        node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)

        t = torch.from_numpy(t).long()
        t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()

        xt = self.diffusion.sample(node_labels_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        t = torch.from_numpy(t).float()
        t = t.reshape(-1)
        xt = xt.reshape(-1)
        edge_index = edge_index.to(node_labels.device).reshape(2, -1)

        # Denoise
        x0_pred = self.forward(
            xt.float().to(node_labels.device),
            t.float().to(node_labels.device),
            features,
            edge_index,
        )

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, node_labels)
        self.log("train/loss", loss)
        return loss

    def gaussian_training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        _, graph_data, point_indicator = batch
        t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
        node_labels = graph_data.x[:, 0]
        features = graph_data.x[:, 1:]
        edge_index = graph_data.edge_index
        device = node_labels.device

        # Sample from diffusion
        node_labels = node_labels.float() * 2 - 1
        node_labels = node_labels * (1.0 + 0.05 * torch.rand_like(node_labels))
        node_labels = node_labels.unsqueeze(1).unsqueeze(1)

        t = torch.from_numpy(t).long()
        t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()
        xt, epsilon = self.diffusion.sample(node_labels, t)

        t = torch.from_numpy(t).float()
        t = t.reshape(-1)
        xt = xt.reshape(-1)
        edge_index = edge_index.to(device).reshape(2, -1)
        epsilon = epsilon.reshape(-1)

        # Denoise
        epsilon_pred = self.forward(
            xt.float().to(device),
            t.float().to(device),
            features,
            edge_index,
        )
        epsilon_pred = epsilon_pred.squeeze(1)

        # Compute loss
        loss = mse_loss(epsilon_pred, epsilon.float())
        self.log("train/loss", loss)
        return loss

    @staticmethod
    def process_batch(batch: tuple) -> tuple:
        """
        Process the input batch and return node labels, edge index, and adjacency matrix.

        Args:
            batch: Input batch containing graph data

        Returns:
            tuple containing:
                - node_labels: Tensor of node labels
                - edge_index: Edge index tensor
                - adj_mat: Sparse adjacency matrix in CSR format
        """
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

    def test_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # noqa: ARG002
        split: str = "test",
    ) -> None:
        device = batch[-1].device
        node_labels, edge_index, adj_mat, features = self.process_batch(batch)

        stacked_predict_labels = []
        for _ in range(self.args.sequential_sampling):
            predict_labels = self.diffusion_sample(node_labels.shape[0], edge_index, device, features)
            predict_labels = predict_labels.cpu().detach().numpy()
            stacked_predict_labels.append(predict_labels)

        predict_labels = np.concatenate(stacked_predict_labels, axis=0)
        all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

        splitted_predict_labels = np.split(predict_labels, all_sampling)
        solved_solutions = [mis_decode_np(predict_labels, adj_mat) for predict_labels in splitted_predict_labels]
        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
        best_solved_cost = np.max(solved_costs)

        gt_cost = node_labels.cpu().numpy().sum()
        metrics = {
            f"{split}/gt_cost": gt_cost,
        }

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)

        self.test_outputs.append(metrics)
