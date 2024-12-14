from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from ea.problem_instance import ProblemInstance
from problems.mis.mis_evaluation import mis_decode_np, mis_decode_torch
from scipy.sparse import coo_matrix


class MISInstanceBase(ProblemInstance):
    def __init__(self, device: Literal["cpu", "cuda"] = "cpu") -> None:
        self.device = device

    @abstractmethod
    def get_degrees(self) -> torch.Tensor:
        """Returns a tensor of shape (n_nodes,) with the degrees of the nodes."""


class MISInstance(MISInstanceBase):
    def __init__(
        self,
        adj_matrix: torch.Tensor,
        n_nodes: int,
        gt_labels: np.array | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        super().__init__(device)
        self.adj_matrix = adj_matrix
        self.n_nodes = n_nodes
        self.gt_labels = gt_labels

    def get_gt_cost(self) -> float:
        return self.gt_labels.sum()

    def evaluate_individual(self, individual: torch.Tensor) -> float:
        """Individual is a BRKGA random key of shape (n_nodes,), values in [0, 1]."""
        return mis_decode_torch(individual, self.adj_matrix).sum()

    def evaluate_solution(self, solution: torch.Tensor) -> float:
        """Solution stored as a torch.Tensor of shape (n_nodes,), where 1 indicates a node in the MIS."""
        return solution.sum()

    def get_feasible_from_individual(self, individual: torch.Tensor) -> torch.Tensor:
        """Individual is a BRKGA random key of shape (n_nodes,), values in [0, 1]."""
        return mis_decode_torch(individual, self.adj_matrix)

    def get_degrees(self) -> torch.Tensor:
        return self.adj_matrix.sum(dim=1).to_dense().squeeze().to(self.device)


class MISInstanceNumPy(MISInstanceBase):
    def __init__(
        self,
        adj_matrix: coo_matrix,
        n_nodes: int,
        gt_labels: np.array | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        super().__init__(device)
        self.adj_matrix = adj_matrix
        self.n_nodes = n_nodes
        self.gt_labels = gt_labels

    def get_gt_cost(self) -> float:
        return self.gt_labels.sum()

    def evaluate_individual(self, individual: torch.Tensor) -> float:
        individual = individual.cpu().numpy()
        return mis_decode_np(individual, self.adj_matrix).sum()

    def evaluate_solution(self, solution: torch.Tensor) -> float:
        return solution.sum()

    def get_feasible_from_individual(self, individual: torch.Tensor) -> torch.Tensor:
        individual = individual.cpu().numpy()
        decoded = mis_decode_np(individual, self.adj_matrix)
        return torch.tensor(decoded, dtype=torch.bool, device=self.device)

    def get_degrees(self) -> torch.Tensor:
        return torch.tensor(self.adj_matrix.sum(axis=1), dtype=torch.int, device=self.device).squeeze()


def create_mis_instance(sample: tuple, device: Literal["cpu", "cuda"] = "cpu", np_eval: bool = True) -> MISInstance:
    _, graph_data, _ = sample
    edge_index = graph_data.edge_index

    if np_eval:
        edge_index_np = edge_index.cpu().numpy()
        adj_mat_sparse = coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        ).tocsr()
        return MISInstanceNumPy(adj_mat_sparse, graph_data.x.shape[0], graph_data.x, device)

    values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
    adj_mat_sparse = torch.sparse_coo_tensor(
        edge_index, values, (graph_data.x.shape[0], graph_data.x.shape[0]), device=device
    ).coalesce()
    assert adj_mat_sparse.is_sparse

    return MISInstance(adj_mat_sparse, graph_data.x.shape[0], graph_data.x, device)
