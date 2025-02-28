import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from config.configs.mis_inference import config as mis_instance_config
from config.myconfig import Config
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_instance import MISInstance
from problems.mis.solve_optimal_recombination import solve_constrained_mis, solve_wmis
from torch_geometric.loader import DataLoader

from difusco.sampler import DifuscoSampler

common_config = mis_instance_config.update(
    device="cpu",
    task="mis",
    data_path="data",
    models_path="models",
)


def print_dict(d: dict) -> None:
    print(json.dumps(d, indent=4))


def prepare_config(dataset: str) -> Config:
    return common_config.update(
        test_split=f"mis/{dataset}/test",
        test_split_label_dir=f"mis/{dataset}/test_labels",
        training_split=f"mis/{dataset}/train",
        training_split_label_dir=f"mis/{dataset}/train_labels",
        validation_split=f"mis/{dataset}/test",
        validation_split_label_dir=f"mis/{dataset}/test_labels",
        ckpt_path=f"mis/mis_{dataset}_gaussian.ckpt",
        cache_dir=f"cache/mis/{dataset}/test",
    )


def get_dataloader(config: Config) -> DataLoader:
    data_file = Path(config.data_path)
    dataset = MISDataset(data_dir=data_file / config.test_split, data_label_dir=data_file / config.test_split_label_dir)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def from_torch_to_numpy_indices(solution: torch.Tensor) -> np.array:
    return solution.cpu().numpy().nonzero()[0]


def run_experiment(
    recombination: Literal["solve_wmis", "solve_constrained_mis"],
    n_samples: int,
    dataset: str,
    pop_size: int = 16,
) -> None:
    """
    Recombination types:
    - "solve_wmis": Greedy recombination
    - "solve_constrained_mis": Random recombination
    """
    config = prepare_config(dataset)
    dataloader = get_dataloader(config)

    # we need pop_size samples
    config.parallel_sampling = pop_size
    config.sequential_sampling = 1
    config.mode = "difusco"

    sampler = DifuscoSampler(config)

    results = []

    for i, sample in enumerate(dataloader):
        if i == n_samples:
            break
        instance = MISInstance.create_from_batch_sample(sample, "cpu")
        heatmaps = sampler.sample(sample)
        solutions = instance.get_feasible_from_individual_batch(heatmaps)

        n_pairs = pop_size // 2
        instance_results = []

        costs = solutions.sum(dim=-1)
        best_cost = costs.max().item()

        for j in range(n_pairs):
            solution_1 = solutions[2 * j]
            solution_2 = solutions[2 * j + 1]
            solution_1_np = from_torch_to_numpy_indices(solution_1)
            solution_2_np = from_torch_to_numpy_indices(solution_2)

            if recombination == "solve_wmis":
                solve_result = solve_wmis(instance, solution_1_np, solution_2_np, time_limit=15)
            elif recombination == "solve_constrained_mis":
                solve_result = solve_constrained_mis(instance, solution_1_np, solution_2_np, time_limit=15)

            # Calculate parent costs and improvements
            parent_costs = [solve_result["parent_1_obj"], solve_result["parent_2_obj"]]
            best_parent = max(parent_costs)
            avg_parent = sum(parent_costs) / 2
            child_cost = solve_result["children_obj"]

            # Calculate improvements relative to parents
            best_improvement = (child_cost - best_parent) / best_parent * 100
            avg_improvement = (child_cost - avg_parent) / avg_parent * 100

            solve_result = {
                "runtime": solve_result["runtime"],
                "children_obj": child_cost,
                "parent_1_obj": solve_result["parent_1_obj"],
                "parent_2_obj": solve_result["parent_2_obj"],
                "best_parent_obj": best_parent,
                "avg_parent_obj": avg_parent,
                "best_improvement": best_improvement,
                "avg_improvement": avg_improvement,
                "improved": best_improvement > 0,
            }

            instance_results.append(solve_result)

        # process instance results
        # 1. overall best population improvement: from best parent to best child
        # 2. overall average population improvement: from avg parent to avg child
        # 3. percentage of times that we could improve

        children_best = max([result["children_obj"] for result in instance_results])
        children_avg = sum([result["children_obj"] for result in instance_results]) / len(instance_results)
        generation_improvement = (children_best - best_cost) / best_cost
        avg_generation_improvement = (children_avg - best_cost) / best_cost
        ratio_improved = sum([result["improved"] for result in instance_results]) / len(instance_results)

        instance_result = {
            "generation_improvement": generation_improvement,
            "avg_generation_improvement": avg_generation_improvement,
            "ratio_improved": ratio_improved,
            "improved": generation_improvement > 0,
            "runtime_avg": sum([result["runtime"] for result in instance_results]) / len(instance_results),
            "runtime_std": np.std([result["runtime"] for result in instance_results]),
        }

        print(f" -- instance {instance} --")
        print_dict(instance_result)
        results.append(instance_result)

    # finally average the results

    avg_generation_improvement = sum([result["generation_improvement"] for result in results]) / len(results)
    avg_avg_generation_improvement = sum([result["avg_generation_improvement"] for result in results]) / len(results)
    avg_ratio_improved = sum([result["ratio_improved"] for result in results]) / len(results)

    final_result = {
        "population_size": pop_size,
        "recombination": recombination,
        "dataset": dataset,
        "avg_generation_improvement": avg_generation_improvement,
        "avg_avg_generation_improvement": avg_avg_generation_improvement,
        "avg_ratio_improved": avg_ratio_improved,
        "total_improved": sum([result["improved"] for result in results]),
        "total_unimproved": sum([not result["improved"] for result in results]),
        "avg_improved": sum([result["improved"] for result in results]) / len(results),
        "avg_runtime_avg": sum([result["runtime_avg"] for result in results]) / len(results),
        "avg_runtime_std": np.std([result["runtime_std"] for result in results]),
        "num_samples": len(results),
    }

    print(" -- final result --")
    print_dict(final_result)

    final_result["sample_results"] = results

    return final_result


if __name__ == "__main__":
    datasets = ["er_50_100", "er_300_400", "er_700_800"]
    datasets = ["er_50_100"]
    for dataset in datasets:
        results = run_experiment(
            recombination="solve_constrained_mis",  # or solve_wmis
            n_samples=10,
            dataset=dataset,
            pop_size=16,
        )
        # TODO: save results to fil
