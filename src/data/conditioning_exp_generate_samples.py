from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from config.myconfig import Config
from config.mytable import TableSaver
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_instance import create_mis_instance
from problems.mis.solve_optimal_recombination import maximum_weighted_independent_set

N_SAMPLES = 20
N_SOLUTIONS = 24


def get_lil_csr_matrix(adj_matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Convert adjacency matrix to lil format and zero out lower triangle."""
    adj_matrix_lil = sp.lil_matrix(adj_matrix)
    adj_matrix_lil[np.tril_indices(adj_matrix.shape[0])] = 0
    return adj_matrix_lil.tocsr()


def solve_plain_mis(config: Config) -> None:
    """Solve plain MIS using maximum_weighted_independent_set directly.

    Args:
        config: Configuration containing dataset name
    """
    data_dir = "/home/e12223411/repos/difusco/data"
    mis_dataset = MISDataset(
        data_dir=f"{data_dir}/mis/{config.dataset}/test",
        data_label_dir=f"{data_dir}/mis/{config.dataset}/test_labels",
    )

    table_saver = TableSaver(config.table_name)
    already_solved = set(table_saver.get().sample_file_name.unique().tolist())

    for i, batch in enumerate(mis_dataset):
        if i >= N_SAMPLES:
            break

        sample_file_name = mis_dataset.get_file_name_from_sample_idx(i)
        if sample_file_name in already_solved:
            print(f"Skipping {sample_file_name} because it is already solved")
            continue

        print(f"\nSample {i}:")
        label_cost = batch[1].x.sum()
        print(f"Label cost: {label_cost}")

        # Create MIS instance from batch
        instance = create_mis_instance(batch, device="cpu")
        print(f"Graph has {instance.n_nodes} nodes and {instance.edge_index.shape[1]//2} edges")

        # Get adjacency matrix in proper format for maximum_weighted_independent_set
        adj_matrix = get_lil_csr_matrix(instance.adj_matrix_np)

        # Set all weights to 1 for plain MIS
        weights = np.ones(instance.n_nodes)

        step = {
            "er_50_100": 1,
            "er_300_400": 2,
            "er_700_800": 3,
        }[config.dataset]

        desired_costs = [label_cost - i * step for i in range(N_SOLUTIONS)]
        solutions = []

        for desired_cost in desired_costs:
            if desired_cost < 1:
                continue
            print(f"Solving for desired cost: {desired_cost}")
            # Solve using maximum_weighted_independent_set directly
            start_time = time.time()

            # Call the maximum_weighted_independent_set function
            mwis_result = maximum_weighted_independent_set(
                adj_matrix,
                weights,
                desired_cost=desired_cost,
                time_limit=15,
                solver_params={"OutputFlag": 0, "SolutionLimit": 1, "MIPFocus": 1},  # Enable output for debugging
            )

            runtime = time.time() - start_time
            if mwis_result is None:
                continue
            print(f"MIS solution size: {len(mwis_result.x)}")
            print(f"MIS solution weight: {mwis_result.f}")
            print(f"Runtime: {runtime:.4f} seconds")
            print(f"Solution nodes: {mwis_result.x}")
            solutions.append(list(mwis_result.x))

        sol_str = " | ".join([str(sol) for sol in solutions])

        table_saver.put({"sample_file_name": sample_file_name, "solutions": sol_str})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run conditioning experiment.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    args = parser.parse_args()

    config = Config(dataset=args.dataset, table_name=f"results/conditioning_experiment_{args.dataset}.csv")
    solve_plain_mis(config)
