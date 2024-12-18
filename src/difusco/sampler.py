from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import timeit
import numpy as np
import wandb
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from ea.ea_utils import save_results
from ea.evolutionary_algorithm import dataset_factory, instance_factory
from problems.mis.mis_heatmap_experiment import metrics_on_mis_heatmaps

if TYPE_CHECKING:
    from config.config import Config

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
            return self.sample_tsp(batch=batch)
        if self.task == "mis":
            return self.sample_mis(batch=batch)
        error_msg = f"Unknown task: {self.task}"
        raise ValueError(error_msg)

    @torch.no_grad()
    def sample_mis(
        self, batch: tuple | None = None, edge_index: torch.Tensor | None = None, n_nodes: int | None = None
    ) -> torch.Tensor:
        """
        Sample heatmaps from Difusco

        Args:
            batch: Input batch in the format expected by the model

        Returns:
            Tensor containing the sampled heatmaps
        """
        if batch is not None:
            node_labels, edge_index, _ = self.model.process_batch(batch)
            n_nodes = node_labels.shape[0]
        elif edge_index is None or n_nodes is None:
            error_msg = "Must provide either batch or (edge_index and n_nodes)"
            raise ValueError(error_msg)

        edge_index = edge_index.to(self.device)

        heatmaps = None
        for _ in range(self.model.args.sequential_sampling):
            labels_pred = self.model.diffusion_sample(n_nodes, edge_index, self.device)
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
        if self.model.sparse:
            edge_index = edge_index.to(self.device)
        for _ in range(self.model.args.sequential_sampling):
            adj_mat = self.model.diffusion_sample(
                points=points,
                edge_index=edge_index,
                device=self.device,
            )
            adj_mat = torch.from_numpy(adj_mat).to(self.device)
            heatmaps = adj_mat if heatmaps is None else torch.cat((heatmaps, adj_mat), dim=0)

        # clip heatmaps to be between 0 and 1
        return torch.clamp(heatmaps, 0, 1)


def run_difusco_initialization_experiments(config: Config) -> None:
    """Run experiments to evaluate Difusco initialization performance.
    
    Args:
        config: Configuration object containing experiment parameters
    """
    print(f"Running Difusco initialization experiments with config: {config}")
    
    # Initialize dataset and dataloader similar to EA
    dataset = dataset_factory(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize wandb if not in validation mode
    is_validation_run = config.validate_samples is not None
    if not is_validation_run:
        wandb.init(
            project=config.project_name,
            name=config.wandb_logger_name,
            entity=config.wandb_entity,
            config=config.__dict__,
            dir=config.logs_path,
        )
    
    # Initialize sampler
    sampler = DifuscoSampler(config)
    results = []
    
    for i, sample in tqdm(enumerate(dataloader)):
        start_time = timeit.default_timer()
        
        # Create problem instance to evaluate solutions
        instance = instance_factory(config, sample)
        
        # Sample solutions using Difusco
        heatmaps = sampler.sample(sample)
        
        # Convert heatmaps to solutions and evaluate
        if config.task == "tsp":
            instance_results = metrics_on_tsp_heatmaps(heatmaps, instance)
        else:  # MIS
            instance_results = metrics_on_mis_heatmaps(heatmaps, instance)
        
        results.append(instance_results)

        if not is_validation_run:
            wandb.log(instance_results, step=i)
            
        if is_validation_run and i >= config.validate_samples - 1:
            break
    
    # Compute and log aggregate results
    agg_results = {
        "avg_best_cost": sum(r["best_cost"] for r in results) / len(results),
        "avg_avg_cost": sum(r["avg_cost"] for r in results) / len(results),
        "avg_best_gap": sum(r["best_gap"] for r in results) / len(results),
        "avg_avg_gap": sum(r["avg_gap"] for r in results) / len(results),
        "avg_total_entropy": sum(r["total_entropy"] for r in results) / len(results),
        "avg_unique_solutions": sum(r["unique_solutions"] for r in results) / len(results),
        "avg_non_best_solutions": sum(r["non_best_solutions"] for r in results) / len(results),
        "avg_diff_to_nearest_int": sum(r["avg_diff_to_nearest_int"] for r in results) / len(results),
        "avg_diff_to_solution": sum(r["avg_diff_to_solution"] for r in results) / len(results),
        "avg_diff_rounded_to_solution": sum(r["avg_diff_rounded_to_solution"] for r in results) / len(results),
    }
    if not is_validation_run:
        wandb.log(agg_results)
        agg_results["wandb_id"] = wandb.run.id
        save_results(config, agg_results)
        wandb.finish()

if __name__ == "__main__":
    config = Config(
        task="mis",
    )