from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from config.configs.mis_inference import config as mis_inference_config
from config.myconfig import Config
from ea.ea_runner import ea_factory
from ea.ea_utils import dataset_factory, instance_factory
from evotorch.operators import CopyingOperator, CrossOver
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def parse_arguments() -> tuple[Namespace, list[str]]:
    parser = get_arg_parser()
    args, extra = parser.parse_known_args()
    return args, extra


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate training data for difuscombination")

    general = parser.add_argument_group("general")
    # Similar arguments as initialization experiments
    general.add_argument("--split", type=str, required=True)
    general.add_argument("--task", type=str, required=True)
    general.add_argument("--config_name", type=str, required=True)
    general.add_argument("--data_path", type=str, required=True)
    general.add_argument("--logs_path", type=str, default=None)
    general.add_argument("--results_path", type=str, default=None)
    general.add_argument("--test_split", type=str, required=True)
    general.add_argument("--test_split_label_dir", type=str, default=None)
    general.add_argument("--training_split", type=str, required=True)
    general.add_argument("--training_split_label_dir", type=str, default=None)
    general.add_argument("--validation_split", type=str, required=True)
    general.add_argument("--validation_split_label_dir", type=str, default=None)
    general.add_argument("--ckpt_path", type=str, required=True)
    general.add_argument("--num_graph_instances", type=int, required=True)
    general.add_argument("--num_samples_per_graph", type=int, required=True)
    general.add_argument("--pop_size", type=int, required=True)
    general.add_argument("--num_generations", type=int, required=True)
    general.add_argument("--device", type=str, required=False, default="cuda")
    general.add_argument("--process_idx", type=int, required=False, default=0)
    general.add_argument("--num_processes", type=int, required=False, default=1)
    # Add any additional arguments needed
    return parser


def validate_config(config: Config) -> None:
    # we consider the initial pop_size. Then one generation is recombination + mutation
    total_solutions_per_graph = config.pop_size + 2 * config.pop_size * config.num_generations
    assert (
        total_solutions_per_graph >= config.num_samples_per_graph
    ), "Not enough solutions per graph, increase pop_size or num_generations"

    # pop_size must be even
    assert config.pop_size % 2 == 0, "Population size must be even"

    # validate output_path
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    assert output_path.is_dir(), "Output path is not a directory"
    assert output_path.exists(), "Output path does not exist"

    # validate process_idx and num_processes
    assert 0 <= config.process_idx < config.num_processes, "Invalid process index"
    assert config.num_processes > 0, "Number of processes must be greater than 0"


def process_sample_generation(
    config: Config,
    sample: tuple[Any, ...],
    queue: mp.Queue,
    instance_file_name: str,
) -> None:
    """Generate training samples for one graph/instance."""
    try:
        # 1. Initialize DifuscoSampler and problem instance
        # 2. Generate 4 initial difusco samples
        # 3. Generate 4 mutations of those samples
        # 4. Generate 4 mutations of mutations
        # 5. Generate 4 recombinations
        # 6. Save all solutions and their relationships

        print(sample)

        instance = instance_factory(config, sample)
        ga = ea_factory(config, instance)
        problem = ga.problem

        if config.task == "mis":
            values = torch.zeros(config.pop_size, instance.n_nodes, dtype=torch.bool, device=config.device)
        else:  # tsp
            values = torch.zeros(config.pop_size, instance.n + 1, dtype=torch.int64, device=config.device)

        problem._fill(values)  # noqa: SLF001
        solutions = values

        crossover = ga._operators[0]  # noqa: SLF001
        assert isinstance(crossover, CrossOver)

        mutation = ga._operators[1]  # noqa: SLF001
        assert isinstance(mutation, CopyingOperator)

        # values keeps track of the population (size pop_size)
        for _ in range(config.num_generations):
            values = crossover._do_cross_over(values[: config.pop_size // 2], values[config.pop_size // 2 :])  # noqa: SLF001
            solutions = torch.cat([solutions, values.values], dim=0)
            values = mutation._do(values).values  # noqa: SLF001
            solutions = torch.cat([solutions, values], dim=0)

        num_solutions = solutions.shape[0]
        assert num_solutions == config.pop_size + 2 * config.pop_size * config.num_generations
        assert num_solutions >= config.num_samples_per_graph * 2, "Not enough solutions"
        # check that all the values in the solutions are in the range [0, n_nodes)
        assert torch.all(solutions < instance.n_nodes), "Solution contains invalid indices"

        if solutions.unique(dim=0).shape[0] >= config.num_samples_per_graph * 2:
            print("We managed to generate enough unique solutions")
            solutions = solutions.unique(dim=0)
            num_solutions = solutions.shape[0]
        else:
            print("Not enough unique solutions, some pairs will be repeated")

        sampled_solutions = solutions[torch.randperm(num_solutions)[: 2 * config.num_samples_per_graph]].to("cpu")

        # build solutions string
        solution_strs = []
        for i in range(config.num_samples_per_graph * 2):
            if config.task == "mis":
                indices = sampled_solutions[i].nonzero().flatten()
                solution_strs.append(" ".join(map(str, indices.tolist())))
            else:
                solution_strs.append(sampled_solutions[i].tolist())

        solutions_str = " | ".join(solution_strs)

        # save the solutions, together with the instance
        results = {
            "instance_file_name": instance_file_name,
            "solution_str": solutions_str,
        }

        queue.put(results)

    except Exception:  # noqa: BLE001
        queue.put({"error": traceback.format_exc()})


def run_training_data_generation(config: Config) -> None:
    """Generate training data for the difuscombination module.

    For each instance:
    - Generate 16 solutions:
        - 4 difusco samples (unique)
        - 4 mutations of difusco samples
        - 4 mutations of mutations
        - 4 recombinations of initial population
    """
    start_time = time.time()
    config = config.update(initialization="difusco_sampling", np_eval=True, algo="ga", device=config.device)

    dataset = dataset_factory(config, split=config.split)
    torch.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    assert (
        len(dataloader) >= config.num_graph_instances
    ), f"Not enough graphs in the dataset: {len(dataloader)} < {config.num_graph_instances}"

    config = config.update(parallel_sampling=config.pop_size, sequential_sampling=1)

    ctx = mp.get_context("spawn")
    output_path = Path(config.output_path)

    results = []

    for i, sample in tqdm(enumerate(dataloader)):
        if i - 1 == config.num_graph_instances - 1:
            break

        queue = ctx.Queue()
        idx = sample[0].item()

        if i % config.num_processes != config.process_idx:
            continue

        process = ctx.Process(
            target=process_sample_generation, args=(config, sample, queue, dataset.get_file_name_from_sample_idx(idx))
        )

        process.start()
        process.join(timeout=30 * 60)  # 30 minutes timeout
        if process.is_alive():
            process.terminate()
            raise TimeoutError(f"Process timed out for iteration {i}")

        if queue.empty():
            raise RuntimeError("No result returned from the process")

        instance_results = queue.get()

        if "error" in instance_results:
            raise RuntimeError(instance_results["error"])

        results.append(instance_results)

        if process.is_alive():
            process.terminate()
            process.join()

    # save the results by converting to pandas dataframe
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / f"difuscombination_samples_{timestamp}_{config.process_idx}.csv", index=False)

    metadata = {
        "timestamp": timestamp,
        "config": config.__dict__,
        "runtime": time.time() - start_time,
        "num_graph_instances": len(results),
        "runtime_per_instance": (time.time() - start_time) / len(results),
    }
    with open(output_path / f"difuscombination_samples_{timestamp}_{config.process_idx}.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    # Example configuration similar to initialization experiments
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
        split="train",
        pop_size=4,
        num_generations=2,
        num_samples_per_graph=4,
        num_graph_instances=3,
        output_path="data/difuscombination/mis/er_50_100/train",
    )
    config = mis_inference_config.update(config)
    run_training_data_generation(config)
