from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from difuscombination.pl_difuscombination_mis_model import DifusCombinationMISModel

if TYPE_CHECKING:
    from config.myconfig import Config

from difusco.mis.pl_mis_model import MISModel
from difusco.tsp.pl_tsp_model import TSPModel


class DifuscoSampler:
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
            mode: Literal["difusco", "recombination"]
        """
        self.task = config.task
        self.device = config.device
        self.mode = config.mode if "mode" in config else "difusco"  # default to difusco

        ckpt_path = Path(config.models_path) / config.ckpt_path

        # Load the appropriate model
        if self.task == "tsp":
            assert self.mode == "difusco"
            self.model = TSPModel.load_from_checkpoint(ckpt_path, param_args=config, map_location=self.device)
        elif self.task == "mis":
            if self.mode == "difusco":
                self.model = MISModel.load_from_checkpoint(ckpt_path, param_args=config, map_location=self.device)
            elif self.mode == "difuscombination":
                self.model = DifusCombinationMISModel.load_from_checkpoint(
                    ckpt_path, param_args=config, map_location=self.device
                )
            else:
                error_msg = f"Unknown mode: {self.mode}"
                raise ValueError(error_msg)
        else:
            error_msg = f"Unknown task: {self.task}"
            raise ValueError(error_msg)

        self.model.eval()
        self.model.to(self.device)

    def sample(self, batch: tuple, features: torch.Tensor | None = None) -> torch.Tensor:
        """Sample heatmaps from Difusco"""
        if self.task == "tsp":
            return self.sample_tsp(batch=batch)
        if self.task == "mis":
            return self.sample_mis(batch=batch, features=features)
        error_msg = f"Unknown task: {self.task}"
        raise ValueError(error_msg)

    @torch.no_grad()
    def sample_mis(
        self,
        batch: tuple | None = None,
        edge_index: torch.Tensor | None = None,
        n_nodes: int | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample heatmaps from Difusco

        Args:
            batch: Input batch in the format expected by the model
            edge_index: Edge index tensor (only if batch is None)
            n_nodes: Number of nodes (only if batch is None)
            features: Features tensor (optional, only used with batch). Size: (n_nodes, 2)

        Returns:
            Tensor containing the sampled heatmaps
        """
        if batch is not None:
            node_labels, edge_index, _, sample_features = self.model.process_batch(batch)
            n_nodes = node_labels.shape[0]
            if features is None:
                features = sample_features
        elif edge_index is None or n_nodes is None:
            error_msg = "Must provide either batch or (edge_index and n_nodes)"
            raise ValueError(error_msg)

        edge_index = edge_index.to(self.device)

        heatmaps = None
        for _ in range(self.model.args.sequential_sampling):
            labels_pred = self.model.diffusion_sample(n_nodes, edge_index, self.device, features=features)
            labels_pred = labels_pred.reshape((self.model.args.parallel_sampling, -1))
            heatmaps = labels_pred if heatmaps is None else torch.cat((heatmaps, labels_pred), dim=0)

        return torch.clamp(heatmaps, 0, 1)

    @torch.no_grad()
    def sample_tsp(
        self, batch: tuple | None = None, edge_index: torch.Tensor | None = None, points: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Sample heatmaps from Difusco

        Args:
            batch: Input batch in the format expected by the model, or None if providing edge_index and points directly
            edge_index: Edge index tensor, required if batch is None
            points: Points tensor, required if batch is None

        Returns:
            Tensor containing the sampled heatmaps
        """
        # Process batch if provided, otherwise use direct inputs
        if batch is not None:
            _, edge_index, _, points, _, _, _ = self.model.process_batch(batch)
        elif points is None:
            error_msg = "Must provide either batch or both edge_index and points"
            raise ValueError(error_msg)
        if self.model.sparse:
            edge_index = edge_index.to(self.device)

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

        for _ in range(self.model.args.sequential_sampling):
            adj_mat = self.model.diffusion_sample(
                points=points,
                edge_index=edge_index,
                device=self.device,
            )
            adj_mat = torch.from_numpy(adj_mat).to(self.device)
            if self.model.sparse:
                adj_mat = adj_mat.reshape(self.model.args.parallel_sampling, -1)
            heatmaps = adj_mat if heatmaps is None else torch.cat((heatmaps, adj_mat), dim=0)

        # clip heatmaps to be between 0 and 1
        return torch.clamp(heatmaps, 0, 1)
