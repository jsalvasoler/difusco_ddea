from __future__ import annotations

import multiprocessing as mp
import timeit
from typing import Any

import torch
import wandb
from config.config import Config
from config.configs.mis_inference import config as mis_inference_config
from config.configs.tsp_inference import config as tsp_inference_config
from ea.ea_utils import dataset_factory, instance_factory, save_results
from problems.mis.mis_heatmap_experiment import metrics_on_mis_heatmaps
from problems.tsp.tsp_heatmap_experiment import metrics_on_tsp_heatmaps
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from difusco.sampler import DifuscoSampler


def process_difusco_iteration(config: Config, sample: tuple[Any, ...], queue: mp.Queue) -> None:
    """Run a single Difusco iteration and store the result in the queue."""
    try:
        print(f"Starting iteration in process: {mp.current_process().name}")
        
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
    except BaseException as e:  # noqa: BLE001
        print(f"Error in process {mp.current_process().name}: {str(e)}")
        queue.put({"error": str(e)})


def run_difusco_initialization_experiments(config: Config) -> None:
    """Run experiments to evaluate Difusco initialization performance.

    Args:
        config: Configuration object containing experiment parameters
    """
    print(f"Running Difusco initialization experiments with config: {config}")

    # Initialize dataset and dataloader similar to EA
    print("Loading dataset...")  # Added print statement
    dataset = dataset_factory(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Dataset loaded with {len(dataset)} samples")  # Added print statement
    print(f"Dataset factory called in process: {mp.current_process().name}")

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

    results = []
    ctx = mp.get_context("spawn")

    for i, sample in tqdm(enumerate(dataloader)):
        queue = ctx.Queue()
        process = ctx.Process(target=process_difusco_iteration, args=(config, sample, queue))

        process.start()
        try:
            process.join(timeout=30 * 60)  # 30 minutes timeout

            # Check if process is still running after timeout
            if process.is_alive():
                process.terminate()
                raise TimeoutError(f"Process timed out for iteration {i}")

            # Check if queue is empty
            if queue.empty():
                raise RuntimeError("No result returned from the process")

            instance_results = queue.get()

            # Check for errors in the process
            if "error" in instance_results:
                raise RuntimeError(instance_results["error"])

            results.append(instance_results)
            if not is_validation_run:
                wandb.log(instance_results, step=i)

        except (TimeoutError, RuntimeError) as e:
            print(f"Error in iteration {i}: {e}")
        finally:
            if process.is_alive():
                process.terminate()
                process.join()

        if is_validation_run and i >= config.validate_samples - 1:
            break

    def agg_results(results: list[dict], keys: list[str]) -> dict:
        return {f"avg_{key}": sum(r[key] for r in results) / len(results) for key in keys}

    # Compute and log aggregate results
    final_results = agg_results(
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
        wandb.log(final_results)
        final_results["wandb_id"] = wandb.run.id
        save_results(config, final_results)
        wandb.finish()
    print(final_results)


if __name__ == "__main__":
    # pop_size = 2
    # config = Config(
    #     task="mis",
    #     parallel_sampling=pop_size,
    #     sequential_sampling=1,
    #     diffusion_steps=2,
    #     inference_diffusion_steps=50,
    #     validate_samples=2,
    #     np_eval=True,
    #     pop_size=pop_size,
    # )
    # config = mis_inference_config.update(config)
    # run_difusco_initialization_experiments(config)
    pop_size = 2
    config = Config(
        task="tsp",
        parallel_sampling=pop_size,
        sequential_sampling=1,
        diffusion_steps=2,
        inference_diffusion_steps=50,
        validate_samples=2,
        np_eval=True,
        pop_size=pop_size,
    )
    config = tsp_inference_config.update(config)
    run_difusco_initialization_experiments(config)
