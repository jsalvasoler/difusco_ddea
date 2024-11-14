"""Lightning module for training the DIFUSCO MIS model."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_evaluation import mis_decode_np
from scipy.sparse import coo_matrix
from torch import nn
from torch.nn.functional import mse_loss, one_hot

from difusco.diffusion_schedulers import InferenceSchedule
from difusco.pl_meta_model import COMetaModel

if TYPE_CHECKING:
    from argparse import Namespace


class MISModel(COMetaModel):
    def __init__(self, param_args: Namespace | None = None) -> None:
        super().__init__(param_args=param_args, node_feature_only=True)

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

    def forward(self, x: torch.Tensor, t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.model(x, t, edge_index=edge_index)

    def categorical_training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        _, graph_data, point_indicator = batch
        t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
        node_labels = graph_data.x
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
        node_labels = graph_data.x
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
            edge_index,
        )
        epsilon_pred = epsilon_pred.squeeze(1)

        # Compute loss
        loss = mse_loss(epsilon_pred, epsilon.float())
        self.log("train/loss", loss)
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if self.diffusion_type == "gaussian":
            return self.gaussian_training_step(batch, batch_idx)
        if self.diffusion_type == "categorical":
            return self.categorical_training_step(batch, batch_idx)
        error_message = f"Diffusion type {self.diffusion_type} not supported."
        raise ValueError(error_message)

    def categorical_denoise_step(
        self,
        xt: torch.Tensor,
        t: np.ndarray | torch.Tensor,
        device: torch.device,
        edge_index: torch.Tensor | None = None,
        target_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
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
    ) -> torch.Tensor:
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            pred = self.forward(
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
            pred = pred.squeeze(1)
            return self.gaussian_posterior(target_t, t, pred, xt)

    def test_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # noqa: ARG002
        split: str = "test",
    ) -> None:
        device = batch[-1].device

        real_batch_idx, graph_data, point_indicator = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index

        stacked_predict_labels = []
        edge_index = edge_index.to(node_labels.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        ).tocsr()

        for _ in range(self.args.sequential_sampling):
            xt = torch.randn_like(node_labels.float())
            if self.args.parallel_sampling > 1:
                xt = xt.repeat(self.args.parallel_sampling, 1, 1)
                xt = torch.randn_like(xt)

            if self.diffusion_type == "gaussian":
                xt.requires_grad = True
            else:
                xt = (xt > 0).long()
            xt = xt.reshape(-1)

            if self.args.parallel_sampling > 1:
                edge_index = self.duplicate_edge_index(edge_index, node_labels.shape[0], device)

            batch_size = 1
            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=self.args.inference_schedule, T=self.diffusion.T, inference_T=steps
            )

            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
                t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

                if self.diffusion_type == "gaussian":
                    xt = self.gaussian_denoise_step(xt, t1, device, edge_index, target_t=t2)
                else:
                    xt = self.categorical_denoise_step(xt, t1, device, edge_index, target_t=t2)

            if self.diffusion_type == "gaussian":
                predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
            else:
                predict_labels = xt.float().cpu().detach().numpy() + 1e-6
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

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        return self.test_step(batch, batch_idx, split="val")
