from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from difusco.diffusion_schedulers import InferenceSchedule
from difusco.mis.pl_mis_model import MISModel

if TYPE_CHECKING:
    from argparse import Namespace


class HighDegreeSelection(MISModel):
    def __init__(self, param_args: Namespace | None = None) -> None:
        super().__init__(param_args=param_args)

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
                edge_index = self.duplicate_edge_index(
                    edge_index, node_labels.shape[0], device
                )

            batch_size = 1
            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=self.args.inference_schedule,
                T=self.diffusion.T,
                inference_T=steps,
            )

            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
                t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

                if self.diffusion_type == "gaussian":
                    xt = self.gaussian_denoise_step(
                        xt, t1, device, edge_index, target_t=t2
                    )
                else:
                    xt = self.categorical_denoise_step(
                        xt, t1, device, edge_index, target_t=t2
                    )

            if self.diffusion_type == "gaussian":
                predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
            else:
                predict_labels = xt.float().cpu().detach().numpy() + 1e-6
            stacked_predict_labels.append(predict_labels)

        predict_labels = np.concatenate(stacked_predict_labels, axis=0)
        all_sampling = self.args.sequential_sampling * self.args.parallel_sampling
        assert self.args.parallel_sampling == 1, (
            "High degree selection does not support parallel sampling"
        )

        splitted_predict_labels = np.split(predict_labels, all_sampling)
        solved_solutions = [
            high_degree_decode_np(predict_labels)
            for predict_labels in splitted_predict_labels
        ]
        node_labels = node_labels.cpu().numpy()

        accuracy = self.compute_accuracy(solved_solutions, node_labels)

        metrics = {
            f"{split}/accuracy": accuracy,
        }
        self.test_outputs.append(metrics)
        self.log(
            f"{split}/accuracy", accuracy, prog_bar=True, on_epoch=True, sync_dist=True
        )

    @staticmethod
    def compute_accuracy(
        solved_solutions: np.ndarray, node_labels: np.ndarray
    ) -> float:
        total_abs_diff = np.zeros_like(solved_solutions[0])
        for solution in solved_solutions:
            total_abs_diff += np.abs(node_labels - solution)

        return 1 - total_abs_diff.sum() / (total_abs_diff.size * len(solved_solutions))


def high_degree_decode_np(predictions: np.ndarray) -> np.ndarray:
    """Decode the labels to high degree nodes."""
    return (predictions > 0.5).astype(int)
