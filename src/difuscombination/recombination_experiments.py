from __future__ import annotations

import os
import warnings
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from config.myconfig import Config
from ea.ea_runner import ea_factory
from evotorch.operators import CrossOver
from problems.mis.mis_instance import create_mis_instance
from torch_geometric.loader import DataLoader

import wandb
from difusco.experiment_runner import Experiment, ExperimentRunner
from difusco.sampler import DifuscoSampler
from difuscombination.dataset import MISDatasetComb
from difuscombination.pl_difuscombination_mis_model import DifusCombinationMISModel


def parse_arguments() -> tuple[Namespace, list[str]]:
    parser = get_arg_parser()
    args, extra = parser.parse_known_args()
    return args, extra


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run recombination experiments")

    general = parser.add_argument_group("general")
    general.add_argument("--config_name", type=str, required=True)
    general.add_argument("--task", type=str, required=True)
    general.add_argument("--data_path", type=str, required=True)
    general.add_argument("--logs_path", type=str, default=None)
    general.add_argument("--results_path", type=str, default=None)
    general.add_argument("--models_path", type=str, default=None)
    general.add_argument("--test_samples_file", type=str, required=True)
    general.add_argument("--test_graphs_dir", type=str, required=True)
    general.add_argument("--test_labels_dir", type=str, required=True)
    general.add_argument("--process_idx", type=int, required=False, default=0)
    general.add_argument("--num_processes", type=int, required=False, default=1)

    wandb = parser.add_argument_group("wandb")
    wandb.add_argument("--project_name", type=str, default="difusco")
    wandb.add_argument("--wandb_entity", type=str, default=None)
    wandb.add_argument("--wandb_logger_name", type=str, default=None)

    ea_settings = parser.add_argument_group("ea_settings")
    ea_settings.add_argument("--device", type=str, default="cuda")
    ea_settings.add_argument("--pop_size", type=int, default=100)

    recombination_settings = parser.add_argument_group("recombination_settings")
    recombination_settings.add_argument("--ckpt_path_difusco", type=str, default=None)
    recombination_settings.add_argument("--ckpt_path_difuscombination", type=str, default=None)

    dev = parser.add_argument_group("dev")
    dev.add_argument("--profiler", action="store_true", default=False)
    dev.add_argument("--validate_samples", type=int, default=None)

    return parser


class RecombinationExperiment(Experiment):
    def __init__(self, config: Config) -> None:
        self.config = config
        self._validate_config()

        # modify config state
        self._fake_paths_for_sampling_models()
        self._fake_attrs_for_ga()

        # we will just sample two solutions in parallel
        self.config.parallel_sampling = 2
        self.config.sequential_sampling = 1

        # samplers
        config_difuscombination = self.config.update(
            mode="difuscombination",
            ckpt_path=self.config.ckpt_path_difuscombination,
        )
        self.sampler_difuscombination = DifuscoSampler(config_difuscombination)

        config_difusco = self.config.update(
            mode="difusco",
            ckpt_path=self.config.ckpt_path_difusco,
        )
        self.sampler_difusco = DifuscoSampler(config_difusco)

    def _validate_config(self) -> None:
        """Validate the configuration."""
        # Validate paths exist
        assert os.path.exists(self.config.data_path), "data_path does not exist"
        for file_type in ["samples_file", "graphs_dir", "labels_dir"]:
            path = os.path.join(self.config.data_path, self.config[f"test_{file_type}"])
            assert os.path.exists(path), f"{path} does not exist"

        # Validate models exist
        assert os.path.exists(self.config.models_path), "models_path does not exist"
        for model_type in ["difusco", "difuscombination"]:
            path = os.path.join(self.config.models_path, self.config[f"ckpt_path_{model_type}"])
            assert os.path.exists(path), f"{path} does not exist"

        # warning if device is not cuda
        if self.config.device != "cuda":
            warnings.warn(
                f"WARNING: device is not cuda, using {self.config.device}. " "Performance will be degraded.",
                stacklevel=2,
            )

    def _fake_paths_for_sampling_models(self) -> None:
        # fake paths for plain difusco
        self.config.training_split = self.config.test_graphs_dir
        self.config.training_split_label_dir = None
        self.config.validation_split = self.config.test_graphs_dir
        self.config.validation_split_label_dir = None
        self.config.test_split = self.config.test_graphs_dir
        self.config.test_split_label_dir = self.config.test_labels_dir
        # fake paths for difuscombination
        self.config.training_samples_file = self.config.test_samples_file
        self.config.training_labels_dir = self.config.test_labels_dir
        self.config.training_graphs_dir = self.config.test_graphs_dir
        self.config.validation_samples_file = self.config.test_samples_file
        self.config.validation_labels_dir = self.config.test_labels_dir
        self.config.validation_graphs_dir = self.config.test_graphs_dir

    def _fake_attrs_for_ga(self) -> None:
        self.config.device = "cuda"
        self.config.pop_size = 2
        self.config.initialization = "random_feasible"
        self.config.recombination = "classic"
        self.config.tournament_size = 2
        self.config.deselect_prob = 0.05
        self.config.mutation_prob = 0.1
        self.config.opt_recomb_time_limit = 15
        self.config.preserve_optimal_recombination = False

    def run_single_iteration(self, sample: tuple) -> None:
        """
        Run a single iteration of the recombination experiment.

        A sample is: graph, 2 parent solutions, 1 child label

        We want to gather the following solutions:
        - get the label of the child solution (*)
        - (1) run trained difuscombination inference on the graph with 2 parent solutions
        - (2) run trained difuscombinatino inference on the graph with random noise parents
        - (3) run trained difuscombination inference on the graph with 2 heuristic construction parents
        - (4) run trained difusco inference on the graph
        - (5) run MIS GA classic recombination on the 2 parent solutions

        We then want to compute the gaps between all and *
        """
        graph, _, _, parents = DifusCombinationMISModel.process_batch(sample)
        parents = parents.to(self.config.device)
        instance = create_mis_instance(sample, device=self.config.device)
        n_nodes = instance.n_nodes

        results = {}

        def compute_gap(label_cost: float, cost: float) -> float:
            return (label_cost - cost) / label_cost

        def update_results(results: dict, idx: int, heatmaps: torch.Tensor) -> None:
            assert heatmaps.shape == (2, n_nodes)
            mean = heatmaps.sum(dim=1).mean().item()
            best = heatmaps.sum(dim=1).max().item()
            results[f"mean_cost_{idx}"] = mean
            results[f"best_cost_{idx}"] = best
            results[f"mean_gap_{idx}"] = compute_gap(label_cost, mean)
            results[f"best_gap_{idx}"] = compute_gap(label_cost, best)

        def from_heatmaps_to_solution(heatmaps: torch.Tensor) -> torch.Tensor:
            assert heatmaps.shape == (2, n_nodes)
            # Create a copy of the heatmaps to allow modifications
            solutions = heatmaps.clone().detach()
            solutions[0, :] = instance.get_feasible_from_individual(heatmaps[0, :])
            solutions[1, :] = instance.get_feasible_from_individual(heatmaps[1, :])
            return solutions

        # 0.
        # get the quality of the child solution (*)
        label_cost = graph.sum().item()
        results["label_cost"] = label_cost
        results["best_parent_cost"] = parents.float().sum(dim=0).max().item()
        results["mean_parent_cost"] = parents.float().sum(dim=0).mean().item()

        # 1.
        heatmaps = self.sampler_difuscombination.sample(sample)
        solutions = from_heatmaps_to_solution(heatmaps)
        update_results(results, 1, solutions)

        # 2.
        # generate random noise of size (n_nodes, 2) between 0 and 1
        random_noise_parents = torch.rand(n_nodes, 2)
        heatmaps = self.sampler_difuscombination.sample(sample, features=random_noise_parents)
        solutions = from_heatmaps_to_solution(heatmaps)
        update_results(results, 2, solutions)

        # 3.
        # generate feasible parents using the construction heuristic
        feasible_parents = torch.empty(n_nodes, 2, device=self.config.device)
        feasible_parents[:, 0] = (
            instance.get_feasible_from_individual(random_noise_parents[:, 0].to(self.config.device)).clone().detach()
        )
        feasible_parents[:, 1] = (
            instance.get_feasible_from_individual(random_noise_parents[:, 1].to(self.config.device)).clone().detach()
        )
        heatmaps = self.sampler_difuscombination.sample(sample, features=feasible_parents)
        solutions = from_heatmaps_to_solution(heatmaps)
        update_results(results, 3, solutions)

        # 4.
        heatmaps = self.sampler_difusco.sample(sample)
        solutions = from_heatmaps_to_solution(heatmaps)
        update_results(results, 4, solutions)

        # 5.
        ga = ea_factory(self.config, instance, sample=sample)
        crossover = ga._operators[0]  # noqa: SLF001
        assert isinstance(crossover, CrossOver)
        children = crossover._do_cross_over(parents[:, 0].unsqueeze(0), parents[:, 1].unsqueeze(0))  # noqa: SLF001
        solutions = children.values.float()
        update_results(results, 5, solutions)

        return results

    def get_dataloader(self) -> DataLoader:
        dataset = MISDatasetComb(
            samples_file=os.path.join(self.config.data_path, self.config.test_samples_file),
            graphs_dir=os.path.join(self.config.data_path, self.config.test_graphs_dir),
            labels_dir=os.path.join(self.config.data_path, self.config.test_labels_dir),
        )
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def get_final_results(self, results: list[dict]) -> dict:
        # results is a list of dicts, each with the same keys
        # we want to compute the mean of each key
        final = {f"final_{key}": np.mean([result[key] for result in results]) for key in results[0]}
        if wandb.run is not None:
            wandb.log(final)

        # we also add all the config parameters
        final.update(self.config.__dict__)
        return final

    def get_table_name(self) -> str:
        return "results/recombination_results.csv"


def main_recombination_experiments(config: Config) -> None:
    experiment = RecombinationExperiment(config)
    runner = ExperimentRunner(config, experiment)
    runner.run()


if __name__ == "__main__":
    # Example configuration for testing
    config = Config(
        task="mis",
        data_path="/share/joan.salva/repos/difusco/data/data",
        logs_path="/home/joan.salva/repos/difusco/logs",
        results_path="/home/joan.salva/repos/difusco/results",
        models_path="/home/joan.salva/repos/difusco/models",
        test_graphs_dir="mis/er_50_100/test",
        test_samples_file="difuscombination/mis/er_50_100/test",
        test_labels_dir="difuscombination/mis/er_50_100/test_labels",
        ckpt_path_difusco="mis/mis_er_50_100_gaussian.ckpt",
        ckpt_path_difuscombination="difuscombination/mis_er_50_100_gaussian_new.ckpt",
        parallel_sampling=2,
        sequential_sampling=1,
        diffusion_steps=2,
        inference_diffusion_steps=50,
        validate_samples=2,
        profiler=False,
        device="cuda",
        process_idx=0,
        num_processes=1,
    )
    from config.configs.mis_inference import config as mis_inference_config

    config = mis_inference_config.update(config)
    main_recombination_experiments(config)
