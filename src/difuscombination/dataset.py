from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from problems.mis.mis_dataset import MISDataset
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData

"""
A dataset for difuscombination.

One sample is a tuple of (graph, solution_1, solution_2), and its label is the children solution.
"""


class MISDatasetComb(Dataset):
    def __init__(
        self, samples_file: str | os.PathLike, graphs_dir: str | os.PathLike, labels_dir: str | os.PathLike
    ) -> None:
        self.graphs_dir = graphs_dir
        assert os.path.exists(os.path.dirname(self.graphs_dir)), f"Directory {graphs_dir} does not exist"
        self.labels_dir = labels_dir
        assert os.path.exists(os.path.dirname(self.labels_dir)), f"Directory {labels_dir} does not exist"

        self.mis_dataset = MISDataset(data_dir=graphs_dir, data_label_dir=None)

        self.samples_file = samples_file
        if str(self.samples_file).endswith(".csv"):
            assert os.path.exists(self.samples_file), f"File {self.samples_file} does not exist"
        else:
            # check that is a directory
            assert os.path.isdir(self.samples_file), f"File {self.samples_file} is not a directory"
            # check that it contains a csv file
            assert any(
                f.endswith(".csv") for f in os.listdir(self.samples_file)
            ), f"No csv file found in {self.samples_file}"
            # filter by those starting with difuscombination_samples_
            samples_files = [f for f in os.listdir(self.samples_file) if f.startswith("difuscombination_samples_")]
            # sort the csv files by timestamp, take the latest one
            samples_files = sorted(samples_files)
            self.samples_file = os.path.join(self.samples_file, samples_files[-1])
            assert os.path.exists(self.samples_file), f"File {self.samples_file} does not exist"
            print(f"Using samples file {self.samples_file}")

        start_time = time.time()

        self.label_files = sorted(
            [os.path.join(self.labels_dir, f) for f in os.listdir(self.labels_dir) if f.endswith(".txt")]
        )
        assert len(self.label_files) > 0, f"No files found in {self.labels_dir}"

        self.samples_df = pd.read_csv(self.samples_file)
        self.samples_df = self.samples_df.sort_values(by="instance_file_name")

        # check that unique instance_file_name is a subset of graph files
        file_names_in_df = self.samples_df["instance_file_name"].unique()
        graph_dir_files = set(os.listdir(self.graphs_dir))
        file_names_set = set(file_names_in_df)
        # assert file_names_set.issubset(graph_dir_files), f"Some instance files are not in {self.graphs_dir}"

        self._length = len(self.label_files)

        print(f'Loaded "{self.labels_dir}" with {self._length} examples in {time.time() - start_time:.2f}s')

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _get_graph_file_from_label_file(label_file: str) -> str:
        # we just need to remove everything after the double underscore
        file_name = os.path.basename(label_file).split("___")[0]
        assert file_name.endswith(".gpickle"), f"File {file_name} is not a gpickle file"
        return file_name

    @staticmethod
    def _get_solutions_from_solution_str(solution_str: str) -> list[np.ndarray]:
        # every split is a str 93 298 306 | 11 34 37 45
        split = solution_str.split(" | ")
        return [np.array(list(map(int, s.split(" ")))) for s in split]

    def _get_two_parent_solutions(self, file_name: str) -> tuple[np.ndarray, np.ndarray]:
        graph_file_name, sols_id_suffix = file_name.split("___")
        graph_file_name = os.path.basename(graph_file_name)
        # we need to get the line in the samples_df that has the same instance_file_name
        row = self.samples_df[self.samples_df["instance_file_name"] == graph_file_name]
        solution_strs = row["solution_str"].values[0]
        solutions = self._get_solutions_from_solution_str(solution_strs)
        first_idx, second_idx = sols_id_suffix.split(".")[0].split("_")
        return solutions[int(first_idx)], solutions[int(second_idx)]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, GraphData, torch.Tensor]:
        graph_file_name = self._get_graph_file_from_label_file(self.label_files[idx])
        graph_idx = self.mis_dataset.get_sample_idx_from_file_name(graph_file_name)
        num_nodes, _, edge_index = self.mis_dataset.get_example(graph_idx)
        solution_1, solution_2 = self._get_two_parent_solutions(self.label_files[idx])

        # convert solutions from numpy arrays of indices to binary masks
        solution_1_mask = np.zeros(num_nodes)
        solution_1_mask[solution_1] = 1
        solution_2_mask = np.zeros(num_nodes)
        solution_2_mask[solution_2] = 1

        # get node labels
        with open(self.label_files[idx]) as f:
            node_labels = [int(_) for _ in f.read().splitlines()]
        node_labels = np.array(node_labels, dtype=np.int64)
        assert node_labels.shape[0] == num_nodes, f"Node labels shape mismatch: {node_labels.shape[0]} != {num_nodes}"

        # Ensure all arrays are 2-dimensional
        node_labels = node_labels[:, np.newaxis]
        solution_1_mask = solution_1_mask[:, np.newaxis]
        solution_2_mask = solution_2_mask[:, np.newaxis]

        # Concatenate along axis=1
        features = torch.from_numpy(np.concatenate([node_labels, solution_1_mask, solution_2_mask], axis=1)).long()
        assert features.shape[0] == num_nodes, f"Features shape mismatch: {features.shape[0]} != {num_nodes}"
        assert features.shape[1] == 3, f"Features shape mismatch: {features.shape[1]} != 3"

        graph_data = GraphData(x=features, edge_index=torch.from_numpy(edge_index))

        point_indicator = np.array([num_nodes], dtype=np.int64)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
        )
