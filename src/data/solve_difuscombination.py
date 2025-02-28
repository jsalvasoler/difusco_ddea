from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from config.myconfig import Config
from config.mytable import TableSaver
from ea.ea_utils import instance_factory
from problems.mis.mis_dataset import MISDataset
from problems.mis.solve_optimal_recombination import solve_wmis


def get_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", type=str, required=True)
    arg_parser.add_argument("--data_label_dir", type=str, required=True)
    arg_parser.add_argument("--file_path", type=str, required=True)
    arg_parser.add_argument("--num_batches", type=int, required=True)
    arg_parser.add_argument("--batch_idx", type=int, required=True)
    arg_parser.add_argument("--output_dir", type=str, required=True)
    return arg_parser


def solve_difuscombination(config: Config) -> None:
    """
    We want to do the following:

    Input params:
    - file_path: path to the difuscombination_samples_{snapshot}.csv file
    - num_batches: number of batches to split the file into (lines in the csv file)
    - batch_idx: which batch to solve -> i = 0 .. num_batches - 1
    - output_dir: path to the directory where the solutions will be saved

    1. Load a specified difuscombination_samples_{snapshot}.csv file
    2. Use the batch_idx and batch_size to identify the batch
    3. For each graph in the batch, we want to:
        - Load the graph instance
        - Solve the problem for the 4 different input conditioners
        - Save the solution in output_dir with name:
          {graph_name}_0_1.txt, or {graph_name}_2_3.txt, ... {graph_name}_7_8.txt
    """
    assert config.batch_idx in list(range(config.num_batches)), "batch_idx must be in range(num_batches)"

    if config.file_path.endswith("csv"):
        assert Path(config.file_path).exists(), "file_path must exist"
    else:
        # check if it is a directory
        assert os.path.isdir(config.file_path), "If not a csv file, file_path must be a directory"
        # in such case, find the csv file in the directory, with the latest timestamp
        csv_files = [f for f in os.listdir(config.file_path) if f.endswith(".csv")]
        assert len(csv_files) > 0, "No csv files found in the directory"
        csv_files.sort()
        config.file_path = os.path.join(config.file_path, csv_files[-1])

    df = pd.read_csv(config.file_path)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    batch_size = len(df) // config.num_batches
    remainder = len(df) % config.num_batches  # Number of extra elements to distribute

    if config.batch_idx < remainder:
        first_index = config.batch_idx * (batch_size + 1)
        first_index_next = first_index + (batch_size + 1)
    else:
        first_index = remainder * (batch_size + 1) + (config.batch_idx - remainder) * batch_size
        first_index_next = first_index + batch_size

    config = config.update(Config(task="mis", device="cpu"))

    mis_dataset = MISDataset(data_dir=config.data_dir, data_label_dir=None)

    table_saver = TableSaver(str(Path(config.output_dir) / f"results_{config.batch_idx}.csv"))

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

            result = solve_wmis(instance, solution_1, solution_2)

            sample_file_name = f"{instance_file_name}___{2 * j}_{2 * j + 1}.txt"
            with open(os.path.join(config.output_dir, sample_file_name), "w") as f:
                # write result["children_np_labels"] to file
                np.savetxt(f, result["children_np_labels"], fmt="%d")

            del result["children_np_labels"]
            result["instance_file_name"] = instance_file_name
            result["sample_file_name"] = sample_file_name
            table_saver.put(result)


if __name__ == "__main__":
    config = Config(
        data_dir="/home/e12223411/repos/difusco/data/mis/er_50_100/test",
        data_label_dir="/home/e12223411/repos/difusco/data/mis/er_50_100/test_labels",
        file_path="/home/e12223411/repos/difusco/data/difuscombination/mis/er_50_100/test/difuscombination_samples_2025-01-30_14-36-45.csv",
        num_batches=10,
        batch_idx=0,
        output_dir="/home/e12223411/repos/difusco/data/difuscombination/mis/er_50_100/test_labels",
    )

    solve_difuscombination(config)
