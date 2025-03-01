from __future__ import annotations

import os
import timeit
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import numpy as np
import torch
from evotorch.logging import StdOutLogger
from problems.mis.mis_ga import create_mis_ga
from problems.tsp.tsp_ga import create_tsp_ga
from torch_geometric.loader import DataLoader

from difusco.experiment_runner import Experiment, ExperimentRunner
from ea.ea_utils import LogFigures, dataset_factory, get_results_dict, instance_factory

if TYPE_CHECKING:
    from config.myconfig import Config
    from evotorch.algorithms import GeneticAlgorithm

    from ea.problem_instance import ProblemInstance


class EvolutionaryAlgorithm(Experiment):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def run_single_iteration(self, sample: tuple) -> dict:
        instance = instance_factory(self.config, sample)
        tmp_dir = Path(mkdtemp())
        ea = ea_factory(self.config, instance, sample=sample, tmp_dir=tmp_dir)

        if self.config.validate_samples:
            # only log ga for a particular instance if validate_samples is not None
            table_name = self._get_logger_table_name(instance_id=sample[0].item())
            custom_logger = LogFigures(
                table_name=table_name,
                instance_id=sample[0].item(),
                gt_cost=instance.get_gt_cost(),
                tmp_population_file=tmp_dir / "population.txt",
                searcher=ea,
            )
        _ = StdOutLogger(searcher=ea, interval=10, after_first_step=True)

        start_time = timeit.default_timer()
        ea.run(self.config.n_generations)
        end_time = timeit.default_timer()

        if self.config.validate_samples:
            try:
                custom_logger.save_evolution_figure()
            except Exception as e:  # noqa: BLE001
                print(f"Error saving evolution figure: {e}")

        cost = ea.status["pop_best_eval"]
        gt_cost = instance.get_gt_cost()

        diff = cost - gt_cost if ea.problem.objective_sense == "min" else gt_cost - cost
        gap = diff / gt_cost

        results = {"cost": cost, "gt_cost": gt_cost, "gap": gap, "runtime": end_time - start_time}
        return {k: v.item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v for k, v in results.items()}

    def _get_logger_table_name(self, instance_id: int) -> str:
        """Path will be logs_path/ea_logs/wandb_logger_name/id_timestamp.csv"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ea_logs_dir = os.path.join(self.config.logs_path, "ea_logs")
        os.makedirs(ea_logs_dir, exist_ok=True)
        table_name = os.path.join(ea_logs_dir, self.config.wandb_logger_name, f"{instance_id}_{timestamp}.csv")
        os.makedirs(os.path.dirname(table_name), exist_ok=True)
        return table_name

    def get_dataloader(self) -> DataLoader:
        return DataLoader(dataset_factory(self.config), batch_size=1, shuffle=False)

    def get_final_results(self, results: list[dict]) -> dict:
        agg_results = {
            "avg_cost": np.mean([r["cost"] for r in results]),
            "avg_gt_cost": np.mean([r["gt_cost"] for r in results]),
            "avg_gap": np.mean([r["gap"] for r in results]),
            "avg_runtime": np.mean([r["runtime"] for r in results]),
            "n_evals": len(results),
        }
        return get_results_dict(self.config, agg_results)

    def get_table_name(self) -> str:
        return "results/ea_results.csv"


def ea_factory(config: Config, instance: ProblemInstance, **kwargs) -> GeneticAlgorithm:
    if config.task == "mis":
        return create_mis_ga(instance, config, **kwargs)
    if config.task == "tsp":
        return create_tsp_ga(instance, config, **kwargs)
    error_msg = f"No evolutionary algorithm for task {config.task}."
    raise ValueError(error_msg)


def run_ea(config: Config) -> None:
    experiment = EvolutionaryAlgorithm(config)
    runner = ExperimentRunner(config, experiment)
    runner.main()
