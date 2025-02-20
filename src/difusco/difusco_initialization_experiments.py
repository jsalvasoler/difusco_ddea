from __future__ import annotations

import os
import timeit
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
from config.myconfig import Config
from ea.ea_utils import dataset_factory, instance_factory
from problems.mis.mis_heatmap_experiment import (
    get_feasible_solutions as get_feasible_solutions_mis,
)
from problems.mis.mis_heatmap_experiment import (
    metrics_on_mis_heatmaps,
)
from problems.tsp.tsp_heatmap_experiment import (
    get_feasible_solutions as get_feasible_solutions_tsp,
)
from problems.tsp.tsp_heatmap_experiment import (
    metrics_on_tsp_heatmaps,
)
from torch_geometric.loader import DataLoader

from difusco.experiment_runner import Experiment, ExperimentRunner
from difusco.sampler import DifuscoSampler

if TYPE_CHECKING:
    from problems.mis.mis_instance import MISInstance
    from problems.tsp.tsp_instance import TSPInstance


def parse_arguments() -> tuple[Namespace, list[str]]:
    parser = get_arg_parser()
    args, extra = parser.parse_known_args()
    return args, extra


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run an evolutionary algorithm")

    general = parser.add_argument_group("general")
    general.add_argument("--config_name", type=str, required=True)
    general.add_argument("--task", type=str, required=True)
    general.add_argument("--data_path", type=str, required=True)
    general.add_argument("--logs_path", type=str, default=None)
    general.add_argument("--results_path", type=str, default=None)
    general.add_argument("--test_split", type=str, required=True)
    general.add_argument("--test_split_label_dir", type=str, default=None)
    general.add_argument("--training_split", type=str, required=True)
    general.add_argument("--training_split_label_dir", type=str, default=None)
    general.add_argument("--validation_split", type=str, required=True)
    general.add_argument("--validation_split_label_dir", type=str, default=None)

    wandb = parser.add_argument_group("wandb")
    wandb.add_argument("--project_name", type=str, default="difusco")
    wandb.add_argument("--wandb_entity", type=str, default=None)
    wandb.add_argument("--wandb_logger_name", type=str, default=None)

    ea_settings = parser.add_argument_group("ea_settings")
    ea_settings.add_argument("--device", type=str, default="cpu")
    ea_settings.add_argument("--pop_size", type=int, default=100)

    difusco_settings = parser.add_argument_group("difusco_settings")
    difusco_settings.add_argument("--models_path", type=str, default=".")
    difusco_settings.add_argument("--ckpt_path", type=str, default=None)

    tsp_settings = parser.add_argument_group("tsp_settings")
    tsp_settings.add_argument("--sparse_factor", type=int, default=-1)

    mis_settings = parser.add_argument_group("mis_settings")
    mis_settings.add_argument("--np_eval", action="store_true")

    dev = parser.add_argument_group("dev")
    dev.add_argument("--profiler", type=bool, default=False)
    dev.add_argument("--validate_samples", type=int, default=None)
    dev.add_argument("--save_solutions", type=bool, default=False)
    dev.add_argument("--save_solutions_path", type=str, default=None)

    return parser


def get_feasible_solutions(heatmaps: torch.Tensor, instance: MISInstance | TSPInstance, config: Config) -> torch.Tensor:
    if config.task == "mis":
        return get_feasible_solutions_mis(heatmaps, instance)
    if config.task == "tsp":
        return get_feasible_solutions_tsp(heatmaps, instance)

    raise ValueError(f"Invalid task: {config.task}")


class DifuscoInitializationExperiment(Experiment):
    def __init__(self, config: Config) -> None:
        sampler_config = config.update(mode="difusco")
        self.sampler = DifuscoSampler(config=sampler_config)
        self.config = config
        self._validate_config()

    def run_single_iteration(self, sample: tuple[Any, ...]) -> dict:
        """Run a single Difusco iteration and return the results."""
        # Create problem instance to evaluate solutions
        instance = instance_factory(self.config, sample)

        # Sample solutions using Difusco
        start_time = timeit.default_timer()
        heatmaps = self.sampler.sample(sample)
        end_time = timeit.default_timer()
        sampling_time = end_time - start_time

        if self.config.save_solutions:
            instance_id = sample[0].item()
            torch.save(heatmaps, f"{self.config.save_solutions_path}/heatmaps_{instance_id}.pt")
            return {}

        # Convert heatmaps to solutions and evaluate
        if self.config.task == "tsp":
            instance_results = metrics_on_tsp_heatmaps(heatmaps, instance, self.config)
        else:  # MIS
            instance_results = metrics_on_mis_heatmaps(heatmaps, instance, self.config)
        instance_results["sampling_time"] = sampling_time

        return instance_results

    def get_dataloader(self) -> DataLoader:
        """Get the dataloader for the experiment."""
        dataset = dataset_factory(self.config)
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def get_final_results(self, results: list[dict]) -> dict:
        """Compute and return the final aggregated results."""
        if self.config.save_solutions:
            return {}

        def agg_results(results: list[dict], keys: list[str]) -> dict:
            return {f"avg_{key}": sum(r[key] for r in results) / len(results) for key in keys}

        aggregated_results = agg_results(
            results,
            [
                "best_cost",
                "avg_cost",
                "best_gap",
                "avg_gap",
                "total_entropy_heatmaps",
                "total_entropy_solutions",
                "unique_solutions",
                "non_best_solutions",
                "avg_diff_to_nearest_int",
                "avg_diff_to_solution",
                "avg_diff_rounded_to_solution",
                "sampling_time",
                "feasibility_heuristics_time",
            ],
        )

        return self._add_config_and_timestamp(aggregated_results)

    def get_table_name(self) -> str:
        """Get the name of the table to save results to."""
        return "results/init_experiments.csv"

    @staticmethod
    def _add_config_and_timestamp(results: dict[str, float | int | str], config: Config) -> dict:
        """Add configuration and timestamp to results."""
        data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        data.update(results)
        data.update(config.__dict__)
        return data

    def _validate_config(self) -> None:
        """Validate the configuration."""
        assert self.config.pop_size > 0, "pop_size must be greater than 0"
        assert (
            self.config.pop_size == self.config.parallel_sampling * self.config.sequential_sampling
        ), "Requirement: pop_size == parallel_sampling * sequential_sampling"

        if "categorical" in self.config.ckpt_path:
            assert self.config.diffusion_type == "categorical", "diffusion_type must be categorical"
        elif "gaussian" in self.config.ckpt_path:
            assert self.config.diffusion_type == "gaussian", "diffusion_type must be gaussian"

        if self.config.save_heatmaps:
            assert self.config.save_heatmaps_path is not None, "save_heatmaps_path must be provided"
            os.makedirs(self.config.save_heatmaps_path, exist_ok=True)


def main_init_experiments(config: Config) -> None:
    experiment = DifuscoInitializationExperiment(config)
    runner = ExperimentRunner(config, experiment)
    runner.main()


if __name__ == "__main__":
    from config.configs.mis_inference import config as mis_inference_config

    pop_size = 4
    config = Config(
        task="mis",
        data_path="/home/e12223411/repos/difusco/data",
        logs_path="/home/e12223411/repos/difusco/logs",
        results_path="/home/e12223411/repos/difusco/results",
        models_path="/home/e12223411/repos/difusco/models",
        test_split="mis/er_50_100/test",
        test_split_label_dir="mis/er_50_100/test_labels",
        training_split="mis/er_50_100/train",
        training_split_label_dir="mis/er_50_100/train_labels",
        validation_split="mis/er_50_100/test",
        validation_split_label_dir="mis/er_50_100/test_labels",
        ckpt_path="mis/mis_er_50_100_gaussian.ckpt",
        parallel_sampling=pop_size,
        sequential_sampling=1,
        diffusion_steps=2,
        inference_diffusion_steps=50,
        validate_samples=2,
        np_eval=True,
        profiler=False,
        pop_size=pop_size,
        save_heatmaps=True,
        save_heatmaps_path="/home/e12223411/repos/difusco/cache/mis/er_50_100/test",
    )
    config = mis_inference_config.update(config)
    main_init_experiments(config)
