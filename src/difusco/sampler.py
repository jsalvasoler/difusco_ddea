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
        task: Literal["tsp", "mis"],
        config: Config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the sampler

        Args:
            task: The task to sample for ("tsp" or "mis")
            ckpt_path: Path to the model checkpoint
            param_args: Arguments for model initialization
            device: Device to run inference on
        """
        self.task = task
        self.device = device

        ckpt_path = Path(config.models_path) / config.ckpt_path

        # Load the appropriate model
        if task == "tsp":
            self.model = TSPModel.load_from_checkpoint(ckpt_path, param_args=config, map_location=device)
        elif task == "mis":
            self.model = MISModel.load_from_checkpoint(ckpt_path, param_args=config, map_location=device)
        else:
            error_msg = f"Unknown task: {task}"
            raise ValueError(error_msg)

        self.model.eval()
        self.model.to(device)

    def sample(self, batch: tuple) -> torch.Tensor:
        """Sample heatmaps from Difusco"""
        if self.task == "tsp":
            return self.sample_tsp(batch)
        if self.task == "mis":
            error_msg = "MIS sampling not yet implemented"
            raise NotImplementedError(error_msg)
        error_msg = f"Unknown task: {self.task}"
        raise ValueError(error_msg)

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
        for _ in range(self.model.args.sequential_sampling):
            adj_mat = self.model.diffusion_sample(
                points=points.to(self.device),
                adj_matrix=adj_matrix.to(self.device),
                edge_index=edge_index,
                device=self.device,
            )
            adj_mat = torch.from_numpy(adj_mat).to(self.device)
            heatmaps = adj_mat if heatmaps is None else torch.cat((heatmaps, adj_mat), dim=0)

        # clip heatmaps to be between 0 and 1
        return torch.clamp(heatmaps, 0, 1)
