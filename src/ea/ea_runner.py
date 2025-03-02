from __future__ import annotations

import os
import timeit
from argparse import ArgumentParser, Namespace
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


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run an evolutionary algorithm")

    general = parser.add_argument_group("general")
    general.add_argument("--task", type=str, required=True)
    general.add_argument("--data_path", type=str, required=True)
    general.add_argument("--logs_path", type=str, default=None)
    general.add_argument("--results_path", type=str, default=None)
    general.add_argument("--test_split", type=str, required=True)
    general.add_argument("--test_split_label_dir", type=str, default=None)

    wandb = parser.add_argument_group("wandb")
    wandb.add_argument("--project_name", type=str, default="difusco")
    wandb.add_argument("--wandb_entity", type=str, default=None)
    wandb.add_argument("--wandb_logger_name", type=str, default=None)

    ea_settings = parser.add_argument_group("ea_settings")
    ea_settings.add_argument("--device", type=str, default="cpu")
    ea_settings.add_argument("--pop_size", type=int, default=100)
    ea_settings.add_argument("--n_generations", type=int, default=100)
    ea_settings.add_argument("--max_two_opt_it", type=int, default=-1)
    ea_settings.add_argument("--initialization", type=str, default="random_feasible")
    ea_settings.add_argument("--recombination", type=str, default="classic")
    ea_settings.add_argument("--config_name", type=str, default=None)
    ea_settings.add_argument("--save_results", type=lambda x: x.lower() in ["true", "1", "yes", "y"], default=False)

    difusco_settings = parser.add_argument_group("difusco_settings")
    difusco_settings.add_argument("--models_path", type=str, default=".")
    difusco_settings.add_argument("--ckpt_path", type=str, default=None)
    difusco_settings.add_argument("--ckpt_path_difuscombination", type=str, default=None)
    difusco_settings.add_argument("--diffusion_type", type=str, default="categorical")
    difusco_settings.add_argument("--diffusion_schedule", type=str, default="linear")
    difusco_settings.add_argument("--inference_schedule", type=str, default="cosine")
    difusco_settings.add_argument("--diffusion_steps", type=int, default=1000)
    difusco_settings.add_argument("--inference_diffusion_steps", type=int, default=1000)
    difusco_settings.add_argument("--parallel_sampling", type=int, default=1)
    difusco_settings.add_argument("--sequential_sampling", type=int, default=1)
    difusco_settings.add_argument("--hidden_dim", type=int, default=256)
    difusco_settings.add_argument("--aggregation", type=str, default="sum")
    difusco_settings.add_argument("--n_layers", type=int, default=12)
    difusco_settings.add_argument("--use_activation_checkpoint", action="store_true")
    difusco_settings.add_argument("--fp16", action="store_true")
    difusco_settings.add_argument("--training_split", type=str, default=None)
    difusco_settings.add_argument("--validation_split", type=str, default=None)

    tsp_settings = parser.add_argument_group("tsp_settings")
    tsp_settings.add_argument("--sparse_factor", type=int, default=-1)

    mis_settings = parser.add_argument_group("mis_settings")
    mis_settings.add_argument("--tournament_size", type=int, default=2)
    mis_settings.add_argument("--deselect_prob", type=float, default=0.05)
    mis_settings.add_argument("--mutation_prob", type=float, default=0.25)
    mis_settings.add_argument("--opt_recomb_time_limit", type=int, default=15)
    mis_settings.add_argument(
        "--preserve_optimal_recombination", type=lambda x: x.lower() in ["true", "1", "yes", "y"], default=False
    )

    dev = parser.add_argument_group("dev")
    dev.add_argument("--profiler", action="store_true")
    dev.add_argument("--validate_samples", type=int, default=None)

    return parser


def validate_args(args: Namespace) -> None:
    assert args.task in ["tsp", "mis"]

    assert args.pop_size > 2, "Population size must be greater than 2."
    assert args.initialization in ["random_feasible", "difusco_sampling"]
    assert args.recombination in ["classic", "difuscombination", "optimal"]

    if args.task == "mis":
        assert args.recombination in [
            "classic",
            "optimal",
            "difuscombination",
        ], "Choose a valid recombination method for mis."
        assert args.initialization in [
            "random_feasible",
            "difusco_sampling",
        ], "Choose a valid initialization method for mis."
        assert args.tournament_size > 0, "Tournament size must be greater than 0 for mis."
        assert args.deselect_prob > 0, "Deselect probability must be greater than 0 for mis."

    if args.task == "tsp":
        assert args.max_two_opt_it > 0, "max_two_opt_it must be greater than 0 for tsp."

    for dir_path in [args.data_path, args.logs_path]:
        if dir_path:
            assert os.path.exists(dir_path), f"Path {dir_path} does not exist."

    for split in ["test_split"]:
        if not getattr(args, split):
            continue
        full_path = os.path.join(args.data_path, getattr(args, split))
        assert os.path.exists(full_path), f"Path {getattr(args, split)} does not exist."

    assert args.project_name == "difusco", "Project name must be of the form difusco."
    assert args.config_name is not None, "Config name must be provided."

    # Validate wandb logger name. Format example: tsp_diffusion_graph_categorical_tsp50_test
    if args.wandb_logger_name:
        assert args.wandb_logger_name.startswith(f"{args.task}_ea_"), "Wandb logger name must start with task_ea_"


def parse_args() -> tuple[Namespace, list[str]]:
    parser = get_arg_parser()
    args, extra = parser.parse_known_args()
    validate_args(args)
    return args, extra


class EvolutionaryAlgorithm(Experiment):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def run_single_iteration(self, sample: tuple) -> dict:
        instance = instance_factory(self.config, sample)
        tmp_dir = Path(mkdtemp())
        ea = ea_factory(self.config, instance, sample=sample, tmp_dir=tmp_dir)

        table_name = self._get_logger_table_name(instance_id=sample[0].item())
        if self.config.validate_samples:
            # only log ga for a particular instance if validate_samples is not None
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

        if self.config.save_results:
            df = ea.get_recombination_saved_results()
            table_name = table_name.replace(".csv", "_recombination.csv")
            df.to_csv(table_name, index=False)

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
        return f"{self.config.results_path}/ea_results.csv"


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
