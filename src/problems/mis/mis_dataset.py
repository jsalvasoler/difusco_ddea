"""MIS (Maximal Independent Set) dataset."""

from __future__ import annotations

import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData


class MISDataset(Dataset):
    def __init__(self, data_dir: str | os.PathLike, data_label_dir: str | os.PathLike | None = None) -> None:
        self.data_dir = data_dir
        assert os.path.exists(os.path.dirname(self.data_dir)), f"Data file {data_dir} does not exist"

        start_time = time.time()
        self.sample_files = [
            os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".gpickle")
        ]
        self.sample_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        assert len(self.sample_files) > 0, f"No files found in {data_dir}"
        self.data_label_dir = data_label_dir
        print(f'Loaded "{data_dir}" with {len(self.sample_files)} examples in {time.time() - start_time:.2f}s')

    def __len__(self) -> int:
        return len(self.sample_files)

    def get_example(self, idx: int) -> tuple[int, np.ndarray, np.ndarray]:
        with open(self.sample_files[idx], "rb") as f:
            graph = pickle.load(f)  # noqa: S301

        num_nodes = graph.number_of_nodes()

        if self.data_label_dir is None:
            node_labels = [x[1] for x in graph.nodes(data="label")]
            if node_labels is not None and node_labels[0] is not None:
                node_labels = np.array(node_labels, dtype=np.int64)
            else:
                node_labels = np.zeros(num_nodes, dtype=np.int64)
        else:
            base_label_file = os.path.basename(self.sample_files[idx]).replace(".gpickle", "_unweighted.result")
            node_label_file = os.path.join(self.data_label_dir, base_label_file)
            with open(node_label_file) as f:
                node_labels = [int(_) for _ in f.read().splitlines()]
            node_labels = np.array(node_labels, dtype=np.int64)
            assert (
                node_labels.shape[0] == num_nodes
            ), f"Node labels shape mismatch: {node_labels.shape[0]} != {num_nodes}"

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

    def get_file_name_from_sample_idx(self, idx: int) -> str:
        return os.path.basename(self.sample_files[idx])

    def get_sample_idx_from_file_name(self, file_name: str) -> int:
        return self.sample_files.index(os.path.join(self.data_dir, file_name))
