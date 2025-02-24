import timeit
from typing import Literal

import torch
from config.mytable import TableSaver
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_instance import create_mis_instance
from torch_geometric.loader import DataLoader

mis_dataset = MISDataset(
    data_dir="data/mis/er_50_100/test",
    data_label_dir="data/mis/er_50_100/test_labels",
)


def setup_instance(device: Literal["cpu", "cuda"] = "cpu", np_eval: bool = True) -> tuple:
    """Helper function to setup a MIS instance for benchmarking."""
    dataloader = DataLoader(mis_dataset, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))
    instance = create_mis_instance(batch, device=device, np_eval=np_eval)
    ind = torch.rand(instance.n_nodes).to(device)
    return instance, ind


def benchmark_mis_experiment(n_iterations: int = 1000) -> None:
    # Run benchmarks for different configurations
    configs = [
        ("NumPy (CPU)", "cpu", True),
        ("NumPy (CUDA)", "cuda", True),
        ("PyTorch (CPU)", "cpu", False),
        ("PyTorch (CUDA)", "cuda", False),
    ]

    def run_benchmark(device: str, np_eval: bool, n_iterations: int) -> float:
        """Run a single benchmark configuration and return average time in seconds."""
        setup_code = f"""
from __main__ import setup_instance
instance, ind = setup_instance(device='{device}', np_eval={np_eval})
"""
        stmt = "instance.get_feasible_from_individual(ind)"
        return timeit.timeit(stmt=stmt, setup=setup_code, number=n_iterations)

    print(f"\nBenchmarking results ({n_iterations} iterations):")
    for name, device, np_eval in configs:
        time = run_benchmark(device, np_eval, n_iterations)
        print(f"{name} implementation average time: {time/n_iterations*1000:.3f} ms")


def setup_instance_batched(device: Literal["cpu", "cuda"] = "cpu", np_eval: bool = True, batch_size: int = 32) -> tuple:
    """Helper function to setup a MIS instance for benchmarking."""
    dataloader = DataLoader(mis_dataset, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))
    instance = create_mis_instance(batch, device=device, np_eval=np_eval)
    ind = torch.rand(batch_size, instance.n_nodes).to(device)
    return instance, ind


def benchmark_mis_experiment_batched(n_iterations: int = 1000, batch_size: int = 32) -> None:
    configs = [
        ("NumPy", "cpu", True),
        ("NumPy", "cuda", True),
        ("PyTorch", "cpu", False),
        ("PyTorch", "cuda", False),
    ]

    def run_benchmark(device: str, np_eval: bool, n_iterations: int) -> float:
        """Run a single benchmark configuration and return average time in seconds."""
        setup_code = f"""
from __main__ import setup_instance_batched
instance, ind = setup_instance_batched(device='{device}', np_eval={np_eval}, batch_size={batch_size})
"""
        stmt = "instance.get_feasible_from_individual_batch(ind)"
        return timeit.timeit(stmt=stmt, setup=setup_code, number=n_iterations)

    table_saver = TableSaver("benchmark_results.csv")
    print(f"\nBenchmarking results ({n_iterations} iterations):")
    for name, device, np_eval in configs:
        time = run_benchmark(device, np_eval, n_iterations)
        avg_time = time / n_iterations * 1000
        print(f"{name}+{device}+{np_eval} average time: {avg_time:.3f} ms")

        # write results to file
        table_saver.put(
            {
                "name": name,
                "device": device,
                "np_eval": np_eval,
                "batch_size": batch_size,
                "avg_time": avg_time,
            }
        )


if __name__ == "__main__":
    # benchmark_mis_experiment(1000)
    for batch_size in [16, 32, 64, 128]:
        benchmark_mis_experiment_batched(50, batch_size)
