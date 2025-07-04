"""Lightning module for training the DIFUSCO MIS model."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data
from problems.mis.mis_evaluation import mis_decode_np
from scipy.sparse import coo_matrix
from torch import nn
from torch.nn.functional import mse_loss, one_hot

from difusco.diffusion_schedulers import InferenceSchedule
from difusco.pl_meta_model import COMetaModel

if TYPE_CHECKING:
    from argparse import Namespace


class MISModelBase(COMetaModel):
    def __init__(self, param_args: Namespace | None = None) -> None:
        super().__init__(param_args=param_args, node_feature_only=True)

        # Parameters to be set by the child class

        self.train_dataset = None
        self.test_dataset = None
        self.validation_dataset = None

    @staticmethod
    def process_batch(batch: tuple) -> tuple:
        """
        Process the input batch and return node labels, edge index, adjacency matrix, and optional features.

        Args:
            batch: Input batch containing graph data

        Returns:
            tuple containing:
                - node_labels: Tensor of node labels
                - edge_index: Edge index tensor
                - adj_mat: Sparse adjacency matrix in CSR format
                - features: Features tensor (optional)
        """
        _, graph_data, _ = batch
        if len(graph_data.x.shape) != 2 or graph_data.x.shape[1] == 1:
            node_labels = graph_data.x
            features = None
        else:
            node_labels = graph_data.x[:, 0]
            features = graph_data.x[:, 1:]
        edge_index = graph_data.edge_index

        edge_index = edge_index.to(node_labels.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        ).tocsr()

        return node_labels, edge_index, adj_mat, features

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, edge_index: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """We pass kwargs since difuscombination needs extra arguments."""
        return self.model(x, t, edge_index=edge_index, **kwargs)

    def categorical_training_step(
        self,
        batch: tuple,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        node_labels, edge_index, point_indicator, features = self.unpack_batch(batch)
        device = node_labels.device

        # Sample from diffusion
        node_labels_onehot = one_hot(node_labels.long(), num_classes=2).float()
        node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)

        t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(
            int
        )
        t = torch.from_numpy(t).long()
        t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()

        xt = self.diffusion.sample(node_labels_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        t = torch.from_numpy(t).float()
        t = t.reshape(-1)
        xt = xt.reshape(-1)
        edge_index = edge_index.to(device).reshape(2, -1)

        # Denoise
        x0_pred = self.forward(
            xt.float().to(device),
            t.float().to(device),
            edge_index=edge_index,
            features=features.float().to(device) if features is not None else None,
        )

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, node_labels)
        self.log("train/loss", loss)
        return loss

    @abstractmethod
    def unpack_batch(self, batch: tuple) -> tuple:
        """
        Unpack the input batch and return node labels, edge index, point indicator, and optional features.

        Args:
            batch: Input batch containing graph data

        Returns:
            tuple containing:
                - node_labels: Tensor of node labels
                - edge_index: Edge index tensor
                - point_indicator: Point indicator tensor
                - features: Features tensor (optional)
        """

    def gaussian_training_step(
        self,
        batch: tuple,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        node_labels, edge_index, point_indicator, features = self.unpack_batch(batch)
        device = node_labels.device

        # Sample from diffusion
        node_labels = node_labels.float() * 2 - 1
        node_labels = node_labels * (1.0 + 0.05 * torch.rand_like(node_labels))
        node_labels = node_labels.unsqueeze(1).unsqueeze(1)

        t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(
            int
        )
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
            edge_index=edge_index,
            features=features.float().to(device) if features is not None else None,
        )
        epsilon_pred = epsilon_pred.squeeze(1)

        # Compute loss
        loss = mse_loss(epsilon_pred, epsilon.float())
        self.log("train/loss", loss)
        return loss

    def categorical_denoise_step(
        self,
        xt: torch.Tensor,
        t: np.ndarray | torch.Tensor,
        device: torch.device,
        edge_index: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        target_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(
                xt.float().to(device),
                t.float().to(device),
                features=features.float().to(device) if features is not None else None,
                edge_index=edge_index.long().to(device)
                if edge_index is not None
                else None,
            )
            x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
            return self.categorical_posterior(target_t, t, x0_pred_prob, xt)

    def gaussian_denoise_step(
        self,
        xt: torch.Tensor,
        t: np.ndarray | torch.Tensor,
        device: torch.device,
        edge_index: torch.Tensor | None = None,
        target_t: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            pred = self.forward(
                xt.float().to(device),
                t.float().to(device),
                features=features.float().to(device) if features is not None else None,
                edge_index=edge_index.long().to(device)
                if edge_index is not None
                else None,
            )
            pred = pred.squeeze(1)
            return self.gaussian_posterior(target_t, t, pred, xt)

    @torch.no_grad()
    def diffusion_sample(
        self,
        n_nodes: int,
        edge_index: torch.Tensor,
        device: torch.device,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Denoise to get a diffusion sample.

        If parallel_sampling is greater than 1, the output has shape (parallel_sampling, num_nodes),
        where num_nodes is the number of nodes in the graph. Otherwise, the output has shape (num_nodes,).
        """
        if self.args.parallel_sampling > 1:
            xt = torch.randn(
                (self.args.parallel_sampling, n_nodes), device=device, dtype=torch.float
            )
        else:
            xt = torch.randn(n_nodes, device=device, dtype=torch.float)

        if self.diffusion_type == "gaussian":
            pass
            # xt.requires_grad = True
        else:
            xt = (xt > 0).long()
        xt = xt.reshape(-1)

        if self.args.parallel_sampling > 1:
            edge_index = self.duplicate_edge_index(edge_index, n_nodes, device)
            if features is not None:
                features = features.repeat(self.args.parallel_sampling, 1)

        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(
            inference_schedule=self.args.inference_schedule,
            T=self.diffusion.T,
            inference_T=steps,
        )

        # Diffusion iterations
        for i in range(steps):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)

            if self.diffusion_type == "gaussian":
                xt = self.gaussian_denoise_step(
                    xt, t1, device, edge_index, target_t=t2, features=features
                )
            else:
                xt = self.categorical_denoise_step(
                    xt, t1, device, edge_index, target_t=t2, features=features
                )

        if self.diffusion_type == "gaussian":  # noqa: SIM108
            predict_labels = xt.float() * 0.5 + 0.5
        else:
            predict_labels = xt.float() + 1e-6

        return predict_labels

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if self.diffusion_type == "gaussian":
            return self.gaussian_training_step(batch, batch_idx)
        if self.diffusion_type == "categorical":
            return self.categorical_training_step(batch, batch_idx)
        error_message = f"Diffusion type {self.diffusion_type} not supported."
        raise ValueError(error_message)

    def test_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # noqa: ARG002
        split: str = "test",
    ) -> None:
        device = batch[-1].device
        node_labels, edge_index, adj_mat, features = self.process_batch(batch)

        stacked_predict_labels = []
        start_time = time.time()
        n_times = 0
        snapshots = []
        for _ in range(self.args.sequential_sampling):
            assert self.args.time_limit_inf_s is not None, "Time limit must be set"
            assert self.args.time_limit_inf_s > 0, "Time limit must be greater than 0"
            if (
                self.args.time_limit_inf_s is not None
                and time.time() - start_time > self.args.time_limit_inf_s
            ):
                break
            predict_labels = self.diffusion_sample(
                node_labels.shape[0], edge_index, device, features
            )
            predict_labels = predict_labels.cpu().detach().numpy()
            stacked_predict_labels.append(predict_labels)
            snapshots.append(time.time() - start_time)
            n_times += 1

        predict_labels = np.concatenate(stacked_predict_labels, axis=0)
        all_sampling = self.args.parallel_sampling * n_times

        splitted_predict_labels = np.split(predict_labels, all_sampling)
        solved_solutions = [
            mis_decode_np(predict_labels, adj_mat)
            for predict_labels in splitted_predict_labels
        ]
        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]

        for i in range(n_times):
            to_log = max(solved_costs[: (i + 1) * self.args.parallel_sampling])
            self.log(f"{split}/step_{i}/best_cost", to_log)
            self.log(f"{split}/step_{i}/time", snapshots[i])

        best_solved_cost = np.max(solved_costs)

        gt_cost = node_labels.cpu().numpy().sum()
        metrics = {
            f"{split}/gt_cost": gt_cost,
            f"{split}/inf_runtime": time.time() - start_time,
            f"{split}/n_generated_samples": len(stacked_predict_labels),
        }

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self.log(
            f"{split}/solved_cost",
            best_solved_cost,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.test_outputs.append(metrics)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        return self.test_step(batch, batch_idx, split="val")
