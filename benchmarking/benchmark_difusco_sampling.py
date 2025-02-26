from __future__ import annotations

import copy
import timeit
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.profiler
from config.configs.mis_inference import config as mis_inference_config
from config.mytable import TableSaver
from problems.mis.mis_dataset import MISDataset
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.loader import DataLoader

from difusco.sampler import DifuscoSampler

if TYPE_CHECKING:
    from config.myconfig import Config


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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # batch needs to be 1!!

    # here is where the batch size is set
    config.parallel_sampling = batch_size
    config.sequential_sampling = 1

    sampler = DifuscoSampler(config=config)
    batch = next(iter(dataloader))

    return sampler, batch


def benchmark_sampling(config: Config, batch_size: int, n_iterations: int = 50) -> tuple[float, str]:
    """Run a single benchmark configuration and return average time in seconds and profiling results."""
    sampler, batch = setup_sampler_and_batch(config, batch_size)

    def run_benchmark() -> None:
        sampler.sample(batch)

    # Run regular timing benchmark
    total_time = timeit.timeit(lambda: run_benchmark(), number=n_iterations)

    # Run with profiler for detailed analysis
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Run a few iterations with profiling
        for _ in range(5):
            with record_function("sample_batch"):
                sampler.sample(batch)
            prof.step()

    # Generate detailed profiling report
    profile_report = str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    # Save detailed report to file
    report_filename = f"results/profile_report_{config.test_split.replace('/', '_')}_{batch_size}.txt"
    with open(report_filename, "w") as f:
        f.write(f"Profiling Report for {config.test_split} with batch size {batch_size}\n")
        f.write("=" * 80 + "\n")
        f.write(profile_report)
        f.write("\n\nMemory Stats:\n")
        f.write("=" * 80 + "\n")
        f.write(str(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=100)))

    # Return a shorter version for console output
    console_report = str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return total_time, console_report


def run_benchmarks(n_iterations: int = 50) -> None:
    """Run benchmarks for different MIS configurations and batch sizes."""
    batch_sizes = [4, 8, 16, 32, 64]
    configs = {
        # "er_50_100": prepare_config("er_50_100"),
        "er_300_400": prepare_config("er_300_400"),
        "er_700_800": prepare_config("er_700_800"),
    }

    table_saver = TableSaver("benchmark_difusco_sampling_results.csv")
    print(f"\nBenchmarking DifuscoSampler on MIS datasets ({n_iterations} iterations):")

    for dataset_name, config in configs.items():
        print(f"\nTesting {dataset_name}:")
        for batch_size in batch_sizes:
            try:
                time, profile_report = benchmark_sampling(config, batch_size, n_iterations)
                avg_time = time / n_iterations * 1000  # Convert to milliseconds
                print(f"  Batch size {batch_size:3d}: {avg_time:.3f} ms")
                print("\nProfiling Report:")
                print(profile_report)
                print("-" * 80)

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
            except Exception as e:  # noqa: BLE001
                print(f"  Error with batch size {batch_size}: {e!s}")


if __name__ == "__main__":
    run_benchmarks(5)
