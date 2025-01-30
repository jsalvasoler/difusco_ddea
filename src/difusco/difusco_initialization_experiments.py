from __future__ import annotations

import multiprocessing as mp
import timeit
import traceback
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any

import wandb
from config.configs.mis_inference import config as mis_inference_config
from config.myconfig import Config
from config.mytable import TableSaver
from ea.ea_utils import dataset_factory, instance_factory
from problems.mis.mis_heatmap_experiment import metrics_on_mis_heatmaps
from problems.tsp.tsp_heatmap_experiment import metrics_on_tsp_heatmaps
from pyinstrument import Profiler
from torch.profiler import ProfilerActivity, profile
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from difusco.sampler import DifuscoSampler


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
    dev.add_argument("--profiler", action="store_true")
    dev.add_argument("--validate_samples", type=int, default=None)

    return parser


def process_difusco_iteration(config: Config, sample: tuple[Any, ...], queue: mp.Queue) -> None:
    """Run a single Difusco iteration and store the result in the queue."""

    def run_iteration() -> None:
        try:
            # Create sampler in the child process
            sampler = DifuscoSampler(config)

            # Create problem instance to evaluate solutions
            instance = instance_factory(config, sample)

            # Sample solutions using Difusco
            start_time = timeit.default_timer()
            heatmaps = sampler.sample(sample)
            end_time = timeit.default_timer()
            sampling_time = end_time - start_time

            # Convert heatmaps to solutions and evaluate
            if config.task == "tsp":
                instance_results = metrics_on_tsp_heatmaps(heatmaps, instance, config)
            else:  # MIS
                instance_results = metrics_on_mis_heatmaps(heatmaps, instance, config)
            instance_results["sampling_time"] = sampling_time
            queue.put(instance_results)
        except Exception:  # noqa: BLE001
            queue.put({"error": traceback.format_exc()})

    if config.profiler:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            run_iteration()
        print(p.key_averages().table(sort_by="cpu_time_total"))
    else:
        run_iteration()


def add_config_and_timestamp(config: Config, results: dict[str, float | int | str]) -> None:
    data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    data.update(results)
    data.update(config.__dict__)

    return data


def validate_config(config: Config) -> None:
    assert config.pop_size > 0, "pop_size must be greater than 0"
    assert (
        config.pop_size == config.parallel_sampling * config.sequential_sampling
    ), "Requirement: pop_size == parallel_sampling * sequential_sampling"

    if "categorical" in config.ckpt_path:
        assert config.diffusion_type == "categorical", "diffusion_type must be categorical"
    elif "gaussian" in config.ckpt_path:
        assert config.diffusion_type == "gaussian", "diffusion_type must be gaussian"


def run_difusco_initialization_experiments(config: Config) -> None:
    """Run experiments to evaluate Difusco initialization performance.

    Args:
        config: Configuration object containing experiment parameters
    """
    validate_config(config)
    print(f"Running Difusco initialization experiments with config: {config}")

    dataset = dataset_factory(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    is_validation_run = config.validate_samples is not None
    if not is_validation_run:
        wandb.init(
            project=config.project_name,
            name=config.wandb_logger_name,
            entity=config.wandb_entity,
            config=config.__dict__,
            dir=config.logs_path,
        )

    results = []
    ctx = mp.get_context("spawn")

    for i, sample in tqdm(enumerate(dataloader)):
        queue = ctx.Queue()
        process = ctx.Process(target=process_difusco_iteration, args=(config, sample, queue))

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
        if not is_validation_run:
            wandb.log(instance_results, step=i)
        else:
            print(instance_results)

        if process.is_alive():
            process.terminate()
            process.join()

        if is_validation_run and i >= config.validate_samples - 1:
            break

    def agg_results(results: list[dict], keys: list[str]) -> dict:
        return {f"avg_{key}": sum(r[key] for r in results) / len(results) for key in keys}

    # Compute and log aggregate results
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

    if not is_validation_run:
        wandb.log(aggregated_results)
        final_results = add_config_and_timestamp(config, aggregated_results)
        final_results["wandb_id"] = wandb.run.id

        table_saver = TableSaver("results/init_experiments.csv")
        table_saver.put(final_results)
        wandb.finish()


def main_init_experiments(config: Config) -> None:
    if config.profiler:
        with Profiler() as p:
            run_difusco_initialization_experiments(config)
        print(p.output_text(unicode=True, color=True))
    else:
        run_difusco_initialization_experiments(config)


if __name__ == "__main__":
    pop_size = 4
    config = Config(
        task="mis",
        data_path="/home/joan.salva/repos/difusco/data",
        logs_path="/home/joan.salva/repos/difusco/logs",
        results_path="/home/joan.salva/repos/difusco/results",
        models_path="/home/joan.salva/repos/difusco/models",
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
        pop_size=pop_size,
    )
    config = mis_inference_config.update(config)
    main_init_experiments(config)
    # pop_size = 50
    # config = Config(
    #     task="tsp",
    #     parallel_sampling=pop_size,
    #     sequential_sampling=1,
    #     diffusion_steps=50,
    #     inference_diffusion_steps=50,
    #     validate_samples=1,
    #     np_eval=True,
    #     pop_size=pop_size,
    # )
    # config = tsp_inference_config.update(config)
    # run_difusco_initialization_experiments(config)
