from __future__ import annotations

import os
import time

import numpy as np
from config.myconfig import Config
from config.mytable import TableSaver
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_instance import create_mis_instance
from problems.mis.solve_optimal_recombination import get_lil_csr_matrix, maximum_weighted_independent_set


def solve_datasets(
    dataset_name: str, retrieve_kamis_labels: bool = False, time_limit: int = 60, n_instances: int | None = None
) -> None:
    """
    Solve MIS datasets using Gurobi.

    Args:
        dataset_name: Name of the dataset to solve
    """
    # Create config
    config = Config(dataset=dataset_name, table_name=f"results/gurobi_solve_{dataset_name}.csv")

    # Setup data paths
    data_dir = "/home/e12223411/repos/difusco/data"
    mis_dataset = MISDataset(
        data_dir=f"{data_dir}/mis/{config.dataset}/test",
        data_label_dir=f"{data_dir}/mis/{config.dataset}/test_labels",
    )

    # Setup table to save results
    table_saver = TableSaver(config.table_name)
    if os.path.exists(table_saver.table_name):
        already_solved = set(table_saver.get().sample_file_name.unique().tolist())
    else:
        already_solved = set()

    print(f"Solving dataset: {config.dataset}")
    print(f"Already solved: {len(already_solved)} samples")
    s = 0
    for i, batch in enumerate(mis_dataset):
        if n_instances is not None and i >= n_instances:
            break

        if retrieve_kamis_labels:
            x = batch[1].x
            cost = x.sum()
            print(f"Label cost: {cost}")
            s += cost
            continue

        print(f"Solving sample {i+1}/{len(mis_dataset)}")

        sample_file_name = mis_dataset.get_file_name_from_sample_idx(i)
        if sample_file_name in already_solved:
            print(f"Skipping {sample_file_name} because it is already solved")
            continue

        print(f"\nSample {i}: {sample_file_name}")

        # Get optimal solution (ground truth) cost
        label_cost = batch[1].x.sum()
        print(f"Label cost: {label_cost}")

        # Create MIS instance from batch
        instance = create_mis_instance(batch, device="cpu")
        print(f"Graph has {instance.n_nodes} nodes and {instance.edge_index.shape[1]//2} edges")

        # Get adjacency matrix in proper format for maximum_weighted_independent_set
        adj_matrix = get_lil_csr_matrix(instance.adj_matrix_np)

        # Set all weights to 1 for plain MIS
        weights = np.ones(instance.n_nodes)

        # Solve using maximum_weighted_independent_set
        start_time = time.time()

        # Call the maximum_weighted_independent_set function - just solve once
        mwis_result = maximum_weighted_independent_set(
            adj_matrix,
            weights,
            time_limit=time_limit,
            solver_params={
                "OutputFlag": 0,  # Suppress Gurobi output
                "MIPFocus": 1,  # Focus on finding feasible solutions quickly
                "Threads": 6,  # Use 4 threads
            },
        )

        runtime = time.time() - start_time

        if mwis_result is None:
            print(f"No solution found for instance {i}")
            continue

        print(f"MIS solution size: {len(mwis_result.x)}")
        print(f"MIS solution weight: {mwis_result.f}")
        print(f"Runtime: {runtime:.4f} seconds")

        # Save solution to table
        table_saver.put(
            {
                "sample_file_name": sample_file_name,
                "gurobi_cost": mwis_result.f,
                "runtime": runtime,
            }
        )

    if retrieve_kamis_labels:
        print(f"avg label cost: {s/len(mis_dataset)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve MIS datasets using Gurobi.")
    parser.add_argument("--dataset", type=str, default="er_50_100", help="Dataset name.")
    parser.add_argument("--retrieve_kamis_labels", action="store_true", help="Retrieve Kamis labels.")
    parser.add_argument("--time_limit", type=float, default=60, help="Time limit for Gurobi.")
    parser.add_argument("--n_instances", type=int, default=None, help="Number of instances to solve.")

    args = parser.parse_args()
    solve_datasets(args.dataset, args.retrieve_kamis_labels, args.time_limit, args.n_instances)
