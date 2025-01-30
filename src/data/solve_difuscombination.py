from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import gurobipy as gp
import numpy as np
import pandas as pd
from config.myconfig import Config
from config.mytable import TableSaver
from ea.ea_utils import instance_factory
from gurobi_optimods.mwis import maximum_weighted_independent_set
from problems.mis.mis_dataset import MISDataset
from scipy.sparse import lil_matrix

if TYPE_CHECKING:
    from problems.mis.mis_instance import MISInstance


def solve_problem(instance: MISInstance, solution_1: np.array, solution_2: np.array) -> None:
    assert np.all(solution_1 < instance.n_nodes), "solution_1 contains invalid indices"
    assert np.all(solution_2 < instance.n_nodes), "solution_2 contains invalid indices"

    weights = np.full(instance.n_nodes, 0.5)
    only_1 = np.setdiff1d(solution_1, solution_2)
    weights[only_1] = 1
    only_2 = np.setdiff1d(solution_2, solution_1)
    weights[only_2] = 1
    both = np.intersect1d(solution_1, solution_2)
    weights[both] = 1.5

    # Convert to lil_matrix for efficient modification
    adj_matrix_lil = lil_matrix(instance.adj_matrix)

    # Make instance.adj_matrix an upper triangular matrix
    adj_matrix_lil[np.tril_indices(instance.n_nodes)] = 0

    # Convert back to csr_matrix
    adj_matrix = adj_matrix_lil.tocsr()

    env = gp.Env(empty=True)
    env.setParam("ThreadLimit", 8)
    env.setParam("TimeLimit", 60)

    start_time = time.time()
    with env:
        mwis = maximum_weighted_independent_set(adj_matrix, weights=weights)
        print(mwis.x.tolist())
        print(f"parent obj: {len(solution_1)}, {len(solution_2)}")
        print(f"children obj: {len(mwis.x)}")
        np_labels = np.zeros(instance.n_nodes)
        np_labels[mwis.x] = 1

    return {
        "runtime": round(time.time() - start_time, 4),
        "parent_1_obj": len(solution_1),
        "parent_2_obj": len(solution_2),
        "children_obj": len(mwis.x),
        "children_np_labels": np_labels,
    }


def solve_difuscombination() -> None:
    """
    We want to do the following:

    Input params:
    - file_path: path to the difuscombination_samples_{snapshot}.csv file
    - batch_size: number of graphs to solve (lines in the csv file)
    - batch_idx: which batch to solve -> i <= lines // batch_size + 1
    - output_dir: path to the directory where the solutions will be saved

    1. Load a specified difuscombination_samples_{snapshot}.csv file
    2. Use the batch_idx and batch_size to identify the batch
    3. For each graph in the batch, we want to:
        - Load the graph instance
        - Solve the problem for the 4 different input conditioners
        - Save the solution in output_dir with name:
          {graph_name}_0_1.txt, or {graph_name}_2_3.txt, ... {graph_name}_7_8.txt
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", type=str, required=True)
    arg_parser.add_argument("--data_label_dir", type=str, required=True)
    arg_parser.add_argument("--file_path", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, required=True)
    arg_parser.add_argument("--batch_idx", type=int, required=True)
    arg_parser.add_argument("--output_dir", type=str, required=True)
    args = arg_parser.parse_args()

    df = pd.read_csv(args.file_path)
    batch_size = args.batch_size
    batch_idx = args.batch_idx
    data_dir = args.data_dir
    data_label_dir = args.data_label_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    first_index = batch_idx * batch_size
    first_index_next = (batch_idx + 1) * batch_size

    config = Config(task="mis", np_eval=True, device="cpu")

    mis_dataset = MISDataset(data_dir=data_dir, data_label_dir=data_label_dir)

    table_saver = TableSaver(str(Path(output_dir) / f"results_{batch_idx}.csv"))

    for i in range(first_index, min(first_index_next, len(df))):
        print(f"Solving {i} / {len(df)}")
        instance_file_name = df.iloc[i]["instance_file_name"]
        solution_str = df.iloc[i]["solution_str"]

        instance_idx = mis_dataset.get_sample_idx_from_file_name(instance_file_name)

        sample = mis_dataset.__getitem__(instance_idx)
        instance = instance_factory(config, sample)

        def get_solutions_from_solution_str(solution_str: str) -> list[np.ndarray]:
            # every split is a str 93 298 306 | 11 34 37 45
            split = solution_str.split(" | ")
            return [np.array(list(map(int, s.split(" ")))) for s in split]

        solutions = get_solutions_from_solution_str(solution_str)
        n_solutions = len(solutions)
        n_samples = n_solutions // 2

        for j in range(n_samples):
            solution_1 = solutions[2 * j]
            solution_2 = solutions[2 * j + 1]

            result = solve_problem(instance, solution_1, solution_2)

            sample_file_name = f"{instance_file_name}___{2 * j}_{2 * j + 1}.txt"
            with open(os.path.join(output_dir, sample_file_name), "w") as f:
                # write result["children_np_labels"] to file
                np.savetxt(f, result["children_np_labels"], fmt="%d")

            del result["children_np_labels"]
            result["instance_file_name"] = instance_file_name
            result["sample_file_name"] = sample_file_name
            table_saver.put(result)


if __name__ == "__main__":
    import sys

    sys.argv = [
        "solve_difuscombination.py",  # script name
        "--data_dir",
        "/home/e12223411/repos/difusco/data/mis/er_50_100/test",
        "--data_label_dir",
        "/home/e12223411/repos/difusco/data/mis/er_50_100/test_labels",
        "--file_path",
        "/home/e12223411/repos/difusco/data/difuscombination/mis/er_50_100/test/difuscombination_samples_2025-01-30_14-36-45.csv",
        "--batch_size",
        "10",
        "--batch_idx",
        "0",
        "--output_dir",
        "/home/e12223411/repos/difusco/data/difuscombination/mis/er_50_100/test_labels",
    ]

    solve_difuscombination()
