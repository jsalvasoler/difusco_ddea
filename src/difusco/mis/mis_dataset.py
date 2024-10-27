"""MIS (Maximal Independent Set) dataset."""

from __future__ import annotations

import glob
import os

# import pickle5 as pickle  # TODO: I changed this like wrt Edward Sun code. Check if it works.
import pickle

import numpy as np
import torch
from torch_geometric.data import Data as GraphData


class MISDataset(torch.utils.data.Dataset):
    def __init__(self, data_file: str | os.PathLike, data_label_dir: str | os.PathLike | None = None) -> None:
        self.data_file = data_file
        assert os.path.exists(os.path.dirname(self.data_file)), f"Data file {data_file} does not exist"
        self.file_lines = glob.glob(data_file)
        assert len(self.file_lines) > 0, f"No files found in {data_file}"
        self.data_label_dir = data_label_dir
        print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

    def __len__(self) -> int:
        return len(self.file_lines)

    def get_example(self, idx: int) -> tuple[int, np.ndarray, np.ndarray]:
        with open(self.file_lines[idx], "rb") as f:
            graph = pickle.load(f)  # noqa: S301

        num_nodes = graph.number_of_nodes()

        if self.data_label_dir is None:
            node_labels = [x[1] for x in graph.nodes(data="label")]
            if node_labels is not None and node_labels[0] is not None:
                node_labels = np.array(node_labels, dtype=np.int64)
            else:
                node_labels = np.zeros(num_nodes, dtype=np.int64)
        else:
            base_label_file = os.path.basename(self.file_lines[idx]).replace(".gpickle", "_unweighted.result")
            node_label_file = os.path.join(self.data_label_dir, base_label_file)
            with open(node_label_file) as f:
                node_labels = [int(_) for _ in f.read().splitlines()]
            node_labels = np.array(node_labels, dtype=np.int64)
            assert node_labels.shape[0] == num_nodes

        edges = np.array(graph.edges, dtype=np.int64)
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        # add self loop
        self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
        edges = np.concatenate([edges, self_loop], axis=0)
        edges = edges.T

        return num_nodes, node_labels, edges

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, GraphData, torch.Tensor]:
        num_nodes, node_labels, edge_index = self.get_example(idx)
        graph_data = GraphData(x=torch.from_numpy(node_labels), edge_index=torch.from_numpy(edge_index))

        point_indicator = np.array([num_nodes], dtype=np.int64)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
        )
