from __future__ import annotations

import timeit
from pathlib import Path

import torch
from config.configs.mis_inference import config as mis_inference_config
from config.myconfig import Config
from config.mytable import TableSaver
from problems.mis.mis_dataset import MISDataset
from torch_geometric.loader import DataLoader
import copy
from difusco.sampler import DifuscoSampler

# Base configuration for all experiments
common_config = mis_inference_config.update(
    data_path="data",
    logs_path="logs",
    results_path="results",
    models_path="models",
    device="cuda",
    mode="difusco",
    diffusion_type="gaussian",
)


def prepare_config(dataset: str) -> Config:
    config = copy.deepcopy(common_config)
    config.test_split = f"mis/{dataset}/test"
    config.test_split_label_dir = f"mis/{dataset}/test_labels"
    config.training_split = f"mis/{dataset}/train"
    config.training_split_label_dir = f"mis/{dataset}/train_labels"
    config.validation_split = f"mis/{dataset}/test"
    config.validation_split_label_dir = f"mis/{dataset}/test_labels"
    config.ckpt_path = f"mis/mis_{dataset}_gaussian.ckpt"
    return config


def setup_sampler_and_batch(config: Config, batch_size: int) -> tuple[DifuscoSampler, tuple]:
    """Helper function to setup a DifuscoSampler and batch for benchmarking."""
    data_file = Path(config.data_path)
    dataset = MISDataset(data_dir=data_file / config.test_split, data_label_dir=data_file / config.test_split_label_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)   # batch needs to be 1!!
    
    # here is where the batch size is set
    config.parallel_sampling = batch_size
    config.sequential_sampling = 1

    sampler = DifuscoSampler(config=config)
    batch = next(iter(dataloader))
    
    return sampler, batch


def benchmark_sampling(config: Config, batch_size: int, n_iterations: int = 50) -> float:
    """Run a single benchmark configuration and return average time in seconds."""
    sampler, batch = setup_sampler_and_batch(config, batch_size)
    
    def run_benchmark():
        sampler.sample(batch)
    
    return timeit.timeit(lambda: run_benchmark(), number=n_iterations)


def run_benchmarks(n_iterations: int = 50) -> None:
    """Run benchmarks for different MIS configurations and batch sizes."""
    batch_sizes = [4, 8, 16, 32, 64]
    configs = {
        "er_50_100": prepare_config("er_50_100"),
        "er_300_400": prepare_config("er_300_400"),
        "er_700_800": prepare_config("er_700_800"),
    }
    
    table_saver = TableSaver("benchmark_difusco_sampling_results.csv")
    print(f"\nBenchmarking DifuscoSampler on MIS datasets ({n_iterations} iterations):")
    
    for dataset_name, config in configs.items():
        print(f"\nTesting {dataset_name}:")
        for batch_size in batch_sizes:
            try:
                time = benchmark_sampling(config, batch_size, n_iterations)
                avg_time = time / n_iterations * 1000  # Convert to milliseconds
                print(f"  Batch size {batch_size:3d}: {avg_time:.3f} ms")
                
                # Save results
                table_saver.put(
                    {
                        "dataset": dataset_name,
                        "batch_size": batch_size,
                        "avg_time_ms": avg_time,
                        "device": config.device,
                        "mode": config.mode,
                        "graph_size": dataset_name.split("_")[1],  # Extract graph size (50, 300, 700)
                    }
                )
            except Exception as e:
                print(f"  Error with batch size {batch_size}: {str(e)}")


if __name__ == "__main__":
    run_benchmarks(5)
