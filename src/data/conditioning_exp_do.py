from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from config.myconfig import Config
from config.mytable import TableSaver
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_instance import create_mis_instance
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from difusco.sampler import DifuscoSampler

N_SAMPLES = 20


def duplicate_batch(n_times: int, batch: tuple) -> tuple:
    """
    Duplicate a batch n_times.

    Args:
        n_times: Number of times to duplicate the batch
        batch: Input batch to duplicate

    Returns:
        tuple: Duplicated batch
    """
    # Duplicate the first tensor
    tensor0 = batch[0]  # e.g., shape: [1, ...]
    tensor0_dup = tensor0.repeat(n_times, 1)  # repeat along the first dimension

    # Handle the DataBatch in index 1
    # Convert it to a list of Data objects (should contain one element)
    data_list = batch[1].to_data_list()
    # Duplicate the single graph n times
    duplicated_data = [deepcopy(data_list[0]) for _ in range(n_times)]
    # Rebuild the DataBatch; this will automatically handle shifting node indices
    # and creating a new `batch` attribute
    new_data_batch = Batch.from_data_list(duplicated_data)

    # Duplicate the third tensor
    tensor2 = batch[2]  # e.g., shape: [1, ...]
    tensor2_dup = tensor2.repeat(n_times, 1)

    # Return the new batch as a tuple
    return (tensor0_dup, new_data_batch, tensor2_dup)


def load_solutions_from_csv(csv_path: str) -> dict:
    """
    Load solutions from a CSV file using pandas.

    Args:
        csv_path: Path to the CSV file containing solutions

    Returns:
        dict: Mapping from sample file name to list of solutions
    """
    # Read CSV file with pandas
    df = pd.read_csv(csv_path)

    solutions_dict = {}

    # Process each row
    for _, row in df.iterrows():
        sample_file_name = row["sample_file_name"]
        solutions_str = row["solutions"]

        # Parse the solutions string
        solution_lists = solutions_str.split(" | ")
        solutions = []
        for sol_str in solution_lists:
            try:
                # Convert string representation of list to actual list
                sol = ast.literal_eval(sol_str)
                solutions.append(np.array(sol, dtype=np.int64))
            except (SyntaxError, ValueError):
                # Skip malformed solutions
                continue

        solutions_dict[sample_file_name] = solutions

    return solutions_dict


def run_experiment_for_sample(
    sample: tuple, sampler: DifuscoSampler, solutions_dict: dict, sample_file_name: str
) -> dict | None:
    # Skip if we don't have solutions for this sample
    if sample_file_name not in solutions_dict:
        print(f"No solutions found for {sample_file_name}, skipping")
        return None

    solutions = solutions_dict[sample_file_name]
    if len(solutions) < 2:
        print(f"Not enough solutions for {sample_file_name}, skipping")
        return None

    # Make sure solutions are sorted by quality (length)
    solutions = sorted(solutions, key=len)

    # Create MIS instance
    instance = create_mis_instance(sample, device="cpu")
    print(f"Graph has {instance.n_nodes} nodes and {instance.edge_index.shape[1]//2} edges")

    # Create pairs of solutions for batch processing
    solution_pairs = []
    parents_1 = []
    parents_2 = []
    for i in range(0, len(solutions) - 1, 2):
        if i + 1 < len(solutions):
            solution_pairs.append((solutions[i], solutions[i + 1]))
            parent_1 = torch.zeros(instance.n_nodes, dtype=torch.float32)
            parent_2 = torch.zeros(instance.n_nodes, dtype=torch.float32)
            parent_1[solutions[i]] = 1.0
            parent_2[solutions[i + 1]] = 1.0
            parents_1.append(parent_1)
            parents_2.append(parent_2)

    parents_1 = torch.stack(parents_1)
    costs_1 = parents_1.sum(dim=1)
    parents_2 = torch.stack(parents_2)
    costs_2 = parents_2.sum(dim=1)
    print(f"created parents: {parents_1.shape}, {parents_2.shape}")

    num_pairings = parents_1.shape[0]

    features = torch.stack([parents_1, parents_2], dim=2)
    assert features.shape == (num_pairings, instance.n_nodes, 2), "Incorrect features shape"
    # we need to reshape the features to (num_pairings * n_nodes, 2)
    features = features.reshape(num_pairings * instance.n_nodes, 2)
    assert features.shape == (num_pairings * instance.n_nodes, 2), "Incorrect features shape"

    # make a batch with the sample
    batch = duplicate_batch(num_pairings, sample)

    heatmaps = sampler.sample(batch, features=features).to(device="cpu")
    assert heatmaps.shape == (
        num_pairings,
        2,
        instance.n_nodes,
    ), f"Incorrect heatmaps shape: {heatmaps.shape}, expected (num_pairings, 2, solution_length)"

    # split into two children by dropping dimension 1 -> (num_pairings, solution_length)
    heatmaps_child1 = heatmaps.select(1, 0)
    heatmaps_child2 = heatmaps.select(1, 1)

    heatmaps_child = torch.cat([heatmaps_child1, heatmaps_child2], dim=0)
    children = instance.get_feasible_from_individual_batch(heatmaps_child)
    print(f"created children: {children.shape}")
    children_costs = children.sum(dim=1)
    print(f"children costs: {children_costs}")

    results = {"label_cost": instance.get_gt_cost(), "parent_costs": [], "child_costs": []}
    for i in range(num_pairings):
        results["parent_costs"].append((int(costs_1[i].item()), int(costs_2[i].item())))
        results["child_costs"].append((int(children_costs[2 * i].item()), int(children_costs[2 * i + 1].item())))

    print(results)
    return results


def condition_difusco_with_solutions(config: Config) -> None:
    """
    Condition DifuscoSampler with pairs of solutions of increasing quality.

    Args:
        config: Configuration object with parameters
    """
    # Load solutions from CSV
    csv_path = f"results/conditioning_experiment_{config.dataset}.csv"
    print(f"Loading solutions from {csv_path}")
    solutions_dict = load_solutions_from_csv(csv_path)

    # Initialize dataset and dataloader
    data_dir = "/home/e12223411/repos/difusco/data"
    mis_dataset = MISDataset(
        data_dir=f"{data_dir}/mis/{config.dataset}/test",
        data_label_dir=f"{data_dir}/mis/{config.dataset}/test_labels",
    )

    dataloader = DataLoader(mis_dataset, batch_size=1, shuffle=False)

    # Initialize DifuscoSampler
    sampler_config: Config = config.update(
        test_samples_file=f"difuscombination/mis/{config.dataset}/test",
        test_labels_dir=f"difuscombination/mis/{config.dataset}/test_labels",
        test_graphs_dir=f"mis/{config.dataset}/test",
        training_samples_file=f"difuscombination/mis/{config.dataset}/test",
        training_labels_dir=f"difuscombination/mis/{config.dataset}/test_labels",
        training_graphs_dir=f"mis/{config.dataset}/test",
        validation_samples_file=f"difuscombination/mis/{config.dataset}/test",
        validation_labels_dir=f"difuscombination/mis/{config.dataset}/test_labels",
        validation_graphs_dir=f"mis/{config.dataset}/test",
        task="mis",
        device="cuda",
        ckpt_path=f"difuscombination/mis_{config.dataset}_gaussian_new.ckpt",
        parallel_sampling=2,  # Sample 2 times like in recombination operator
        sequential_sampling=1,
        sparse_factor=-1,
        mode="difuscombination",
    )
    sampler: DifuscoSampler = DifuscoSampler(config=sampler_config)

    table_saver = TableSaver(config.output_table)

    from tqdm import tqdm

    for i, sample in tqdm(enumerate(dataloader)):
        if i >= N_SAMPLES:
            break

        sample_file_name = mis_dataset.get_file_name_from_sample_idx(i)
        print(f"\nProcessing sample {i}: {sample_file_name}")

        results = run_experiment_for_sample(sample, sampler, solutions_dict, sample_file_name)
        if results:
            row = {
                "sample_file_name": sample_file_name,
                "label_cost": results["label_cost"],
                "parent_costs": results["parent_costs"],
                "child_costs": results["child_costs"],
            }
            table_saver.put(row)


def main(config: Config) -> None:
    condition_difusco_with_solutions(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run conditioning experiment.")
    parser.add_argument("--dataset", type=str, default="er_50_100", help="Dataset name")

    args = parser.parse_args()

    from config.configs.mis_inference import config as inference_config

    config = inference_config.update(
        dataset=args.dataset,
        device="cpu",  # Use CPU to avoid CUDA issues
        output_table=f"results/conditioning_results_{args.dataset}.csv",
        models_path="models",
        data_path="data",
        logs_path="logs",
        results_path="results",
    )
    main(config)
