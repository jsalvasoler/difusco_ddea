from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from config.configs.mis_inference import config as mis_instance_config
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_instance import MISInstance
from problems.mis.solve_optimal_recombination import (
    solve_constrained_mis,
    solve_local_branching_mis,
    solve_wmis,
)
from torch_geometric.loader import DataLoader

from difusco.sampler import DifuscoSampler

if TYPE_CHECKING:
    import torch
    from config.myconfig import Config


def print_dict(d: dict) -> None:
    print(json.dumps(d, indent=4))


def prepare_config(dataset: str, base_path: str) -> Config:
    common_config = mis_instance_config.update(
        device="cpu",
        task="mis",
        data_path=f"{base_path}/data",
        models_path=f"{base_path}/models",
    )
    return common_config.update(
        test_split=f"mis/{dataset}/test",
        test_split_label_dir=f"mis/{dataset}/test_labels",
        training_split=f"mis/{dataset}/test",
        training_split_label_dir=f"mis/{dataset}/test_labels",
        validation_split=f"mis/{dataset}/test",
        validation_split_label_dir=f"mis/{dataset}/test_labels",
        ckpt_path=f"mis/mis_{dataset}_gaussian.ckpt",
        cache_dir=f"{base_path}/cache/mis/{dataset}/test",
    )


def get_dataloader(config: Config) -> DataLoader:
    data_file = Path(config.data_path)
    dataset = MISDataset(
        data_dir=data_file / config.test_split,
        data_label_dir=data_file / config.test_split_label_dir,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def from_torch_to_numpy_indices(solution: torch.Tensor) -> np.array:
    return solution.cpu().numpy().nonzero()[0]


def run_experiment(
    recombination: Literal[
        "solve_wmis", "solve_constrained_mis", "solve_local_branching_mis"
    ],
    n_samples: int,
    dataset: str,
    pop_size: int = 16,
    **kwargs,
) -> None:
    """
    Recombination types:
    - "solve_wmis": Greedy recombination
    - "solve_constrained_mis": Random recombination
    """
    base_path = kwargs.pop("base_path", ".")
    config = prepare_config(dataset, base_path)
    dataloader = get_dataloader(config)

    # we need pop_size samples
    config.parallel_sampling = pop_size
    config.sequential_sampling = 1
    config.mode = "difusco"

    sampler = DifuscoSampler(config)
    time_limit = kwargs.pop("time_limit", 15)

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

            kwargs["output_flag"] = 1
            kwargs["display_interval"] = 20

            if recombination == "solve_wmis":
                solve_result = solve_wmis(
                    instance,
                    solution_1_np,
                    solution_2_np,
                    time_limit=time_limit,
                    **kwargs,
                )
            elif recombination == "solve_constrained_mis":
                solve_result = solve_constrained_mis(
                    instance,
                    solution_1_np,
                    solution_2_np,
                    time_limit=time_limit,
                    **kwargs,
                )
            elif recombination == "solve_local_branching_mis":
                solve_result = solve_local_branching_mis(
                    instance,
                    solution_1_np,
                    solution_2_np,
                    time_limit=time_limit,
                    **kwargs,
                )
            else:
                raise ValueError(f"Invalid recombination type: {recombination}")

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
        children_avg = sum(
            [result["children_obj"] for result in instance_results]
        ) / len(instance_results)
        generation_improvement = (children_best - best_cost) / best_cost
        avg_generation_improvement = (children_avg - best_cost) / best_cost
        ratio_improved = sum([result["improved"] for result in instance_results]) / len(
            instance_results
        )

        instance_result = {
            "generation_improvement": generation_improvement,
            "avg_generation_improvement": avg_generation_improvement,
            "ratio_improved": ratio_improved,
            "improved": generation_improvement > 0,
            "runtime_avg": sum([result["runtime"] for result in instance_results])
            / len(instance_results),
            "runtime_std": np.std([result["runtime"] for result in instance_results]),
        }

        print(f" -- instance {instance} --")
        print_dict(instance_result)
        results.append(instance_result)

    # finally average the results

    avg_generation_improvement = sum(
        [result["generation_improvement"] for result in results]
    ) / len(results)
    avg_avg_generation_improvement = sum(
        [result["avg_generation_improvement"] for result in results]
    ) / len(results)
    avg_ratio_improved = sum([result["ratio_improved"] for result in results]) / len(
        results
    )

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
        "avg_runtime_avg": sum([result["runtime_avg"] for result in results])
        / len(results),
        "avg_runtime_std": np.std([result["runtime_std"] for result in results]),
        "num_samples": len(results),
    }

    print(" -- final result --")
    print_dict(final_result)

    final_result["sample_results"] = results

    return final_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", type=str, default=".", help="Base directory path"
    )
    parser.add_argument("--recombination", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--pop_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--lambda_penalty", type=float, default=0.5)  # for solve_mwis
    parser.add_argument(
        "--fix_selection", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument(
        "--fix_unselection", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument(
        "--k_factor", type=float, default=1.5
    )  # for solve_local_branching_mis
    parser.add_argument("--time_limit", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = run_experiment(**vars(args))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results["metadata"] = {
        "timestamp": timestamp,
        "args": vars(args),
    }
    filename = (
        f"{args.base_path}/results/{args.dataset}_{args.recombination}_"
        f"{args.fix_selection}_{args.fix_unselection}_{args.k_factor}_{timestamp}.json"
    )
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    # Example command:
    # python -m difuscombination.gurobi_optimal_recombination \
    #     --recombination solve_local_branching_mis \
    #     --n_samples 10 \
    #     --pop_size 16 \
    #     --dataset er_50_100 \
    #     --fix_selection True \
    #     --fix_unselection True \
    #     --k_factor 1.5 \
    #     --time_limit 15
