from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from ea.config import Config

from difusco.mis.pl_mis_model import MISModel
from difusco.tsp.pl_tsp_model import TSPModel


class Sampler:
    """A class that samples heatmaps from Difusco"""

    def __init__(
        self,
        config: Config,
    ) -> None:
        """
        Initialize the sampler

        Args:
            task: The task to sample for ("tsp" or "mis")
            ckpt_path: Path to the model checkpoint
            param_args: Arguments for model initialization
            device: Device to run inference on
        """
        self.task = config.task
        self.device = config.device

        ckpt_path = Path(config.models_path) / config.ckpt_path

        # Load the appropriate model
        if self.task == "tsp":
            self.model = TSPModel.load_from_checkpoint(ckpt_path, param_args=config, map_location=self.device)
        elif self.task == "mis":
            self.model = MISModel.load_from_checkpoint(ckpt_path, param_args=config, map_location=self.device)
        else:
            error_msg = f"Unknown task: {self.task}"
            raise ValueError(error_msg)

        self.model.eval()
        self.model.to(self.device)

    def sample(self, batch: tuple) -> torch.Tensor:
        """Sample heatmaps from Difusco"""
        if self.task == "tsp":
            return self.sample_tsp(batch)
        if self.task == "mis":
            return self.sample_mis(batch)
        error_msg = f"Unknown task: {self.task}"
        raise ValueError(error_msg)

    @torch.no_grad()
    def sample_mis(self, batch: tuple) -> torch.Tensor:
        """
        Sample heatmaps from Difusco

        Args:
            batch: Input batch in the format expected by the model

        Returns:
            Tensor containing the sampled heatmaps
        """
        node_labels, edge_index, adj_mat = self.model.process_batch(batch)
        heatmaps = None
        for _ in range(self.model.args.sequential_sampling):
            labels_pred = self.model.diffusion_sample(node_labels, edge_index, self.device)
            labels_pred = labels_pred.reshape((self.model.args.parallel_sampling, -1))
            heatmaps = labels_pred if heatmaps is None else torch.cat((heatmaps, labels_pred), dim=0)

        return torch.clamp(heatmaps, 0, 1)

    @torch.no_grad()
    def sample_tsp(self, batch: tuple) -> torch.Tensor:
        """
        Sample heatmaps from Difusco

        Args:
            batch: Input batch in the format expected by the model

        Returns:
            Tensor containing the sampled heatmaps
        """
        # Process batch based on task
        _, edge_index, _, points, adj_matrix, _, _ = self.model.process_batch(batch)

        # Handle parallel sampling if enabled

        if self.model.args.parallel_sampling > 1:
            if not self.model.sparse:
                points = points.repeat(self.model.args.parallel_sampling, 1, 1)
            else:
                points = points.repeat(self.model.args.parallel_sampling, 1)
                edge_index = self.model.duplicate_edge_index(
                    edge_index, points.shape[0] // self.model.args.parallel_sampling, self.device
                )

        heatmaps = None
        points = points.to(self.device)
        adj_matrix = adj_matrix.to(self.device)
        if self.model.sparse:
            edge_index = edge_index.to(self.device)
        for _ in range(self.model.args.sequential_sampling):
            adj_mat = self.model.diffusion_sample(
                points=points,
                adj_matrix=adj_matrix,
                edge_index=edge_index,
                device=self.device,
            )
            adj_mat = torch.from_numpy(adj_mat).to(self.device)
            heatmaps = adj_mat if heatmaps is None else torch.cat((heatmaps, adj_mat), dim=0)

        # clip heatmaps to be between 0 and 1
        return torch.clamp(heatmaps, 0, 1)
