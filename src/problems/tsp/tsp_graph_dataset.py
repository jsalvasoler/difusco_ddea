"""TSP (Traveling Salesman Problem) Graph Dataset"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData

if TYPE_CHECKING:
    import os


class TSPGraphDataset(Dataset):
    def __init__(self, data_file: os.PathLike, sparse_factor: float = -1) -> None:
        self.data_file = data_file
        self.sparse_factor = sparse_factor

        with open(data_file) as file:
            self.file_lines = file.read().splitlines()

        print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

    def __len__(self) -> int:
        return len(self.file_lines)

    def get_example(self, idx: int) -> tuple[np.array, np.array]:
        # Select sample
        line = self.file_lines[idx]

        # Clear leading/trailing characters
        line = line.strip()

        # Extract points
        points = line.split(" output ")[0]
        points = points.split(" ")
        points = np.array(
            [[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)]
        )

        # Extract tour
        tour = line.split(" output ")[1]
        tour = tour.split(" ")
        tour = np.array([int(t) for t in tour])
        tour -= 1

        return points, tour

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, GraphData, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        A TSP dataset item depends on the sparse factor.

        sparse_factor <= 0 returns a densely connected graph:
        - idx: (1,)
        - points: (n, 2)
        - adj_matrix: (n, n) -> ground truth adjacency matrix
        - tour: (n,) -> ground truth tour

        sparse_factor > 0 returns a sparse graph where each node is connected to its k nearest neighbors:
        - idx: (1,)
        - GraphData:
            - x: (n, 2)
            - edge_index: (2, m)
            - edge_attr: (m,)
        - point_indicator: (1,)
        - edge_indicator: (1,)
        - tour: (n,) -> ground truth tour
        """
        points, tour = self.get_example(idx)
        if self.sparse_factor <= 0:
            # Return a densely connected graph
            adj_matrix = np.zeros((points.shape[0], points.shape[0]))
            for i in range(tour.shape[0] - 1):
                adj_matrix[tour[i], tour[i + 1]] = 1
            # return points, adj_matrix, tour
            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)),
                torch.from_numpy(points).float(),
                torch.from_numpy(adj_matrix).float(),
                torch.from_numpy(tour).long(),
            )

        # Return a sparse graph where each node is connected to its k nearest neighbors
        # k = self.sparse_factor
        sparse_factor = self.sparse_factor
        kdt = KDTree(points, leaf_size=30, metric="euclidean")
        dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

        edge_index_0 = (
            torch.arange(points.shape[0])
            .reshape((-1, 1))
            .repeat(1, sparse_factor)
            .reshape(-1)
        )
        edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

        edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

        tour_edges = np.zeros(points.shape[0], dtype=np.int64)
        tour_edges[tour[:-1]] = tour[1:]
        tour_edges = torch.from_numpy(tour_edges)
        tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
        graph_data = GraphData(
            x=torch.from_numpy(points).float(),
            edge_index=edge_index,
            edge_attr=tour_edges,
        )

        point_indicator = np.array([points.shape[0]], dtype=np.int64)
        edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
            torch.from_numpy(edge_indicator).long(),
            torch.from_numpy(tour).long(),
        )

    def get_file_name_from_sample_idx(self, idx: int) -> str:
        """For TSP, we define file name as the string index of the sample"""
        return str(idx)
