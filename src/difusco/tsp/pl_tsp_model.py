# ruff: noqa: F841

"""Lightning module for training the DIFUSCO TSP model."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.utils.data
from problems.tsp.tsp_evaluation import TSPEvaluator, merge_tours
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from problems.tsp.tsp_operators import batched_two_opt_torch
from pytorch_lightning.utilities import rank_zero_info
from torch import nn
from torch.nn.functional import mse_loss, one_hot

from difusco.diffusion_schedulers import InferenceSchedule
from difusco.pl_meta_model import COMetaModel

if TYPE_CHECKING:
    from argparse import Namespace


class TSPModel(COMetaModel):
    def __init__(self, param_args: Namespace) -> None:
        super().__init__(param_args=param_args, node_feature_only=False)

        self.train_dataset = (
            TSPGraphDataset(
                data_file=os.path.join(self.args.data_path, self.args.training_split),
                sparse_factor=self.args.sparse_factor,
            )
            if self.args.training_split
            else None
        )

        self.test_dataset = (
            TSPGraphDataset(
                data_file=os.path.join(self.args.data_path, self.args.test_split),
                sparse_factor=self.args.sparse_factor,
            )
            if self.args.test_split
            else None
        )

        self.validation_dataset = (
            TSPGraphDataset(
                data_file=os.path.join(self.args.data_path, self.args.validation_split),
                sparse_factor=self.args.sparse_factor,
            )
            if self.args.validation_split
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(x, t, adj, edge_index)

    def categorical_training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """
        In the case of not self.sparse:
            batch = (sample_idx, points, adj_matrix, gt_tour)
            Dimensions:
                real_batch_idx: (batch_size,)
                points: (batch_size, num_points, 2)
                adj_matrix: (batch_size, num_points, num_points)
                gt_tour: (batch_size, num_points + 1)
        In the case of self.sparse:
            TODO: identify the dimensions of the input batch
        """
        edge_index = None
        if not self.sparse:
            _, points, adj_matrix, _ = batch
            t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
        else:
            _, graph_data, point_indicator, edge_indicator, _ = batch
            t = np.random.randint(
                1, self.diffusion.T + 1, point_indicator.shape[0]
            ).astype(int)
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))

        # Sample from diffusion
        adj_matrix_onehot = one_hot(adj_matrix.long(), num_classes=2).float()
        if self.sparse:
            adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

        xt = self.diffusion.sample(adj_matrix_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        if self.sparse:
            t = torch.from_numpy(t).float()
            t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
            xt = xt.reshape(-1)
            adj_matrix = adj_matrix.reshape(-1)
            points = points.reshape(-1, 2)
            edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)
        else:
            t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

        # Denoise
        x0_pred = self.forward(
            points.float().to(adj_matrix.device),
            xt.float().to(adj_matrix.device),
            t.float().to(adj_matrix.device),
            edge_index,
        )

        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, adj_matrix.long())
        self.log("train/loss", loss)
        return loss

    def gaussian_training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        if self.sparse:
            # TODO: Implement Gaussian diffusion with sparse graphs
            error_msg = (
                "DIFUSCO with sparse graphs are not supported for Gaussian diffusion"
            )
            raise ValueError(error_msg)
        _, points, adj_matrix, _ = batch

        adj_matrix = adj_matrix * 2 - 1
        adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
        # Sample from diffusion
        t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
        xt, epsilon = self.diffusion.sample(adj_matrix, t)

        t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
        # Denoise
        epsilon_pred = self.forward(
            points.float().to(adj_matrix.device),
            xt.float().to(adj_matrix.device),
            t.float().to(adj_matrix.device),
            None,
        )
        epsilon_pred = epsilon_pred.squeeze(1)

        # Compute loss
        loss = mse_loss(epsilon_pred, epsilon.float())
        self.log("train/loss", loss)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        if self.diffusion_type == "gaussian":
            return self.gaussian_training_step(batch, batch_idx)
        if self.diffusion_type == "categorical":
            return self.categorical_training_step(batch, batch_idx)
        error_msg = f"Unknown diffusion type {self.diffusion_type}"
        raise ValueError(error_msg)

    def categorical_denoise_step(
        self,
        points: torch.Tensor,
        xt: torch.Tensor,
        t: np.ndarray,
        device: torch.device,
        edge_index: torch.Tensor | None = None,
        target_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(
                points.float().to(device),
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            if not self.sparse:
                x0_pred_prob = (
                    x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                )
            else:
                x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(
                    dim=-1
                )

            return self.categorical_posterior(target_t, t, x0_pred_prob, xt)

    def gaussian_denoise_step(
        self,
        points: torch.Tensor,
        xt: torch.Tensor,
        t: np.ndarray,
        device: torch.device,
        edge_index: torch.Tensor | None = None,
        target_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            pred = self.forward(
                points.float().to(device),
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
            pred = pred.squeeze(1)
            return self.gaussian_posterior(target_t, t, pred, xt)

    @staticmethod
    def process_dense_batch(batch: tuple) -> tuple:
        """Process a batch of size 1 corresponding to a dense TSP instance"""
        real_batch_idx, points, adj_matrix, gt_tour = batch
        np_points = points.cpu().numpy()[0]
        np_gt_tour = gt_tour.cpu().numpy()[0]
        return real_batch_idx, None, None, points, adj_matrix, np_points, np_gt_tour

    @staticmethod
    def process_sparse_batch(batch: tuple) -> tuple:
        """Process a batch of size 1 corresponding to a sparse TSP instance"""
        real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
        route_edge_flags = graph_data.edge_attr
        points = graph_data.x
        edge_index = graph_data.edge_index
        num_edges = edge_index.shape[1]
        batch_size = point_indicator.shape[0]
        adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
        points = points.reshape((-1, 2))
        edge_index = edge_index.reshape((2, -1))
        np_points = points.cpu().numpy()
        np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
        np_edge_index = edge_index.cpu().numpy()
        return (
            real_batch_idx,
            edge_index,
            np_edge_index,
            points,
            adj_matrix,
            np_points,
            np_gt_tour,
        )

    def process_batch(self, batch: tuple) -> tuple:
        if not self.sparse:
            return self.process_dense_batch(batch)
        return self.process_sparse_batch(batch)

    @torch.no_grad()
    def diffusion_sample(
        self,
        points: torch.Tensor,
        edge_index: torch.Tensor,
        device: str,
    ) -> np.ndarray:
        """
        Denoise to get a diffusion sample.

        Output has shape (parallel_sampling, n, n), where n in the graph size.
        """
        if not self.sparse:
            xt_shape = (self.args.parallel_sampling, points.shape[1], points.shape[1])
        else:
            sparse_dim = (
                points.shape[0] // self.args.parallel_sampling * self.args.sparse_factor
            )
            xt_shape = (self.args.parallel_sampling, sparse_dim)

        xt = torch.randn(*xt_shape, device=device, dtype=torch.float)

        if self.diffusion_type == "gaussian":
            xt.requires_grad = True
        else:
            xt = (xt > 0).long()

        if self.sparse:
            xt = xt.reshape(-1)

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
                    points, xt, t1, device, edge_index, target_t=t2
                )
            else:
                xt = self.categorical_denoise_step(
                    points, xt, t1, device, edge_index, target_t=t2
                )

        if self.diffusion_type == "gaussian":
            adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
        else:
            adj_mat = xt.float().cpu().detach().numpy() + 1e-6

        return adj_mat

    def test_step(
        self,
        batch: tuple,
        batch_idx: int,  # noqa: ARG002
        split: str = "test",
    ) -> None:
        device = batch[-1].device

        (
            real_batch_idx,
            edge_index,
            np_edge_index,
            points,
            adj_matrix,
            np_points,
            np_gt_tour,
        ) = self.process_batch(batch)

        if self.args.parallel_sampling > 1:
            if not self.sparse:
                points = points.repeat(self.args.parallel_sampling, 1, 1)
            else:
                points = points.repeat(self.args.parallel_sampling, 1)
                edge_index = self.duplicate_edge_index(
                    edge_index, np_points.shape[0], device
                )

        stacked_tours = []
        for _ in range(self.args.sequential_sampling):
            adj_mat = self.diffusion_sample(points, edge_index, device)

            if self.args.save_numpy_heatmap:
                self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

            tours, merge_iterations = merge_tours(
                adj_mat,
                np_points,
                np_edge_index,
                sparse_graph=self.sparse,
                parallel_sampling=self.args.parallel_sampling,
            )

            # Refine using 2-opt
            solved_tours, ns = batched_two_opt_torch(
                np_points.astype("float64"),
                np.array(tours).astype("int64"),
                max_iterations=self.args.two_opt_iterations,
                device=device,
            )

            stacked_tours.append(solved_tours)

        solved_tours = np.concatenate(stacked_tours, axis=0)

        tsp_solver = TSPEvaluator(np_points)
        gt_cost = tsp_solver.evaluate(np_gt_tour)

        total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
        all_solved_costs = [
            tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)
        ]
        best_solved_cost = np.min(all_solved_costs)

        metrics = {
            f"{split}/gt_cost": gt_cost,
            f"{split}/2opt_iterations": ns,
            f"{split}/merge_iterations": merge_iterations,
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

    def run_save_numpy_heatmap(
        self,
        adj_mat: torch.Tensor,
        np_points: np.ndarray,
        real_batch_idx: torch.Tensor,
        split: Literal["val", "test"],
    ) -> None:
        if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
            msg = "Save numpy heatmap only support single sampling"
            raise NotImplementedError(msg)

        heatmap_path = os.path.join(
            self.logger.save_dir,
            self.args.wandb_logger_name,
            self.logger.version,
            "numpy_heatmap",
        )

        rank_zero_info(f"Saving heatmap to {heatmap_path}")
        os.makedirs(heatmap_path, exist_ok=True)

        real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
        np.save(
            os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat
        )
        np.save(
            os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"),
            np_points,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        return self.test_step(batch, batch_idx, split="val")
