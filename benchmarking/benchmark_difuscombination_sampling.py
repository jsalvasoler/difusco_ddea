from __future__ import annotations

import timeit
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.profiler
from config.configs.mis_inference import config as mis_inference_config
from config.mytable import TableSaver
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_ga import MISGaProblem
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.loader import DataLoader

from difusco.experiment_runner import Experiment, ExperimentRunner
from difusco.sampler import DifuscoSampler

# from denoising_diffusion_pytorch import Unet, GaussianDiffusion

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
    enable_profiling=False,  # Add profiling flag to config
    validate_samples=1,
    profiler=False,
)


def prepare_config(dataset: str) -> Config:
    return common_config.update(
        test_split=f"mis/{dataset}/test",
        test_split_label_dir=f"mis/{dataset}/test_labels",
        training_split=f"mis/{dataset}/train",
        training_split_label_dir=f"mis/{dataset}/train_labels",
        validation_split=f"mis/{dataset}/test",
        validation_split_label_dir=f"mis/{dataset}/test_labels",
        ckpt_path=f"difuscombination/mis_{dataset}_gaussian.ckpt",
        training_samples_file=f"difuscombination/mis/{dataset}/test",
        training_labels_dir=f"difuscombination/mis/{dataset}/test_labels",
        training_graphs_dir=f"mis/{dataset}/test",
        test_samples_file=f"difuscombination/mis/{dataset}/test",
        test_labels_dir=f"difuscombination/mis/{dataset}/test_labels",
        test_graphs_dir=f"mis/{dataset}/test",
        validation_samples_file=f"difuscombination/mis/{dataset}/test",
        validation_labels_dir=f"difuscombination/mis/{dataset}/test_labels",
        validation_graphs_dir=f"mis/{dataset}/test",
        mode="difuscombination",
    )


class BenchmarkExperiment(Experiment):
    def __init__(
        self, config: Config, batch_size: int, n_iterations: int = 50, experiment_name: str = "default"
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.sampler = None
        self.experiment_name = experiment_name

    def get_dataloader(self) -> DataLoader:
        data_file = Path(self.config.data_path)
        dataset = MISDataset(
            data_dir=data_file / self.config.test_split, data_label_dir=data_file / self.config.test_split_label_dir
        )
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def _run_profiling(self, sample: tuple[Any, ...]) -> None:
        """Run profiling analysis and save results."""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(self.n_iterations):
                with record_function("sample_batch"):
                    self.sampler.sample(sample)
                prof.step()

        # Save detailed profiling report
        report_filename = f"results/profile_report_{self.config.test_split.replace('/', '_')}_{self.batch_size}.txt"
        with open(report_filename, "w") as f:
            f.write(f"Profiling Report for {self.config.test_split} with batch size {self.batch_size}\n")
            f.write("=" * 80 + "\n")
            f.write(str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100)))
            f.write("\n\nMemory Stats:\n")
            f.write("=" * 80 + "\n")
            f.write(str(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=100)))

    def run_single_iteration(self, sample: tuple[Any, ...]) -> dict:
        self.config = self.config.update(
            parallel_sampling=1,
            sequential_sampling=1,
            batch_size=self.batch_size,
        )
        self.sampler = DifuscoSampler(config=self.config)
        # self.sampler = DDPMSampler(config=self.config)
        n_nodes = sample[2].item()
        batched_sample = MISGaProblem._duplicate_batch(n_times=self.batch_size, batch=sample)  # noqa: SLF001
        features = torch.randn(n_nodes * self.batch_size, 2).bool().float()
        # Run timing benchmark
        try:
            start_time = timeit.default_timer()
            for _ in range(self.n_iterations):
                self.sampler.sample(batched_sample, features=features)
            total_time = timeit.default_timer() - start_time
            torch.cuda.empty_cache()  # Clear CUDA cache

            avg_time = (total_time / self.n_iterations) * 1000  # Convert to milliseconds
            results = {
                "avg_time_ms": avg_time,
                "batch_size": self.batch_size,
                "dataset": self.config.test_split.split("/")[1],
                "device": self.config.device,
                "mode": self.config.mode,
                "experiment_name": self.experiment_name,
            }
            table_name = "results/benchmark_difusco_sampling_results.csv"
            table_saver = TableSaver(table_name)
            table_saver.put(results)
            print(results)
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"Error in run_single_iteration: {e!s}")
            print("Traceback:")
            print(traceback.format_exc())

        # Run profiling if enabled
        if self.config.enable_profiling:
            self._run_profiling(sample)

        return {}

    def get_final_results(self, results: list[dict]) -> dict:
        pass

    def get_table_name(self) -> str:
        pass


def run_benchmarks(n_iterations: int = 50, experiment_name: str = "default", enable_profiling: bool = False) -> None:
    """Run benchmarks for different MIS configurations and batch sizes.

    Args:
        n_iterations: Number of iterations to run for each benchmark
        experiment_name: Name of the experiment run to identify different benchmark sessions
        enable_profiling: Whether to enable detailed profiling analysis
    """
    batch_sizes = [4, 8, 16, 32, 64]
    configs = {
        "er_50_100": prepare_config("er_50_100"),
        # "er_300_400": prepare_config("er_300_400"),
        # "er_700_800": prepare_config("er_700_800"),
    }

    print(f"\nBenchmarking DifuscoSampler on MIS datasets ({n_iterations} iterations):")

    for dataset_name, config in configs.items():
        print(f"\nTesting {dataset_name}:")
        config.enable_profiling = enable_profiling  # Set profiling flag
        for batch_size in batch_sizes:
            try:
                experiment = BenchmarkExperiment(config, batch_size, n_iterations, experiment_name)
                runner = ExperimentRunner(config, experiment)
                runner.main()
            except Exception as e:  # noqa: BLE001
                import traceback

                print(f"  Error with batch size {batch_size}: {e!s}")
                print("Traceback:")
                print(traceback.format_exc())


if __name__ == "__main__":
    run_benchmarks(3, "weird experiment", enable_profiling=False)
