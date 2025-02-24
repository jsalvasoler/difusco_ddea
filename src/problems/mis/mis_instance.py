from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from ea.problem_instance import ProblemInstance
from problems.mis.mis_evaluation import (
    mis_decode_np,
    mis_decode_torch,
    mis_decode_torch_batched,
    precompute_neighbors_padded,
)
from scipy.sparse import coo_matrix, csr_matrix

from difusco.mis.pl_mis_model import MISModel


class MISInstanceBase(ProblemInstance):
    def __init__(self, device: Literal["cpu", "cuda"] = "cpu") -> None:
        self.device = device

    @abstractmethod
    def get_degrees(self) -> torch.Tensor:
        """Returns a tensor of shape (n_nodes,) with the degrees of the nodes."""

    @staticmethod
    @abstractmethod
    def create_from_batch_sample(sample: tuple, device: str) -> MISInstanceBase:
        """Create a MISInstance from a batch sample. The batch must have size 1."""

    @abstractmethod
    def get_feasible_from_individual(self, individual: torch.Tensor) -> torch.Tensor:
        """Returns a tensor of shape (n_nodes,) with the nodes in the MIS."""

    @abstractmethod
    def get_feasible_from_individual_batch(self, individual: torch.Tensor) -> torch.Tensor:
        """Returns a tensor of shape (batch_size, n_nodes) with the nodes in the MIS."""


class MISInstance(MISInstanceBase):
    def __init__(
        self,
        adj_matrix: torch.Tensor,
        n_nodes: int,
        edge_index: torch.Tensor | None,
        gt_labels: np.array | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        super().__init__(device)
        self.adj_matrix = adj_matrix
        self.n_nodes = n_nodes
        self.edge_index = edge_index
        self.gt_labels = gt_labels
        self.device = device
        neighbors_padded, degrees = precompute_neighbors_padded(adj_matrix.to_sparse_csr())
        self.neighbors_padded = neighbors_padded.to(device)
        self.degrees = degrees.to(device)

    @staticmethod
    def create_from_batch_sample(sample: tuple, device: str) -> MISInstance:
        """Create a MISInstance from a batch sample. The batch must have size 1."""

        node_labels, edge_index, _, _ = MISModel.process_batch(batch=sample)

        edge_index = edge_index.to(device)
        n_nodes = node_labels.shape[0]

        # Create sparse adjacency matrix
        values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
        adj_mat_sparse = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes), device=device).coalesce()
        assert adj_mat_sparse.is_sparse

        return MISInstance(adj_mat_sparse, n_nodes, edge_index, node_labels, device)

    def get_gt_cost(self) -> float:
        return self.gt_labels.sum() if self.gt_labels is not None else 0.0

    def evaluate_individual(self, individual: torch.Tensor) -> float:
        """Individual is a random key of shape (n_nodes,), values in [0, 1]."""
        return mis_decode_torch(individual, self.adj_matrix).sum()

    def evaluate_solution(self, solution: torch.Tensor) -> float:
        """Solution stored as a torch.Tensor of shape (n_nodes,), where 1 indicates a node in the MIS."""
        return solution.sum()

    def get_feasible_from_individual(self, individual: torch.Tensor) -> torch.Tensor:
        """Individual is a random key of shape (n_nodes,), values in [0, 1]."""
        return mis_decode_torch(individual, self.adj_matrix)

    def get_feasible_from_individual_batch(self, individual: torch.Tensor) -> torch.Tensor:
        """Individual is a random key of shape (batch_size, n_nodes), values in [0, 1]."""
        return mis_decode_torch_batched(individual, self.neighbors_padded, self.degrees)

    def get_degrees(self) -> torch.Tensor:
        return self.adj_matrix.sum(dim=1).to_dense().squeeze().to(self.device)


class MISInstanceNumPy(MISInstanceBase):
    def __init__(
        self,
        adj_matrix: csr_matrix,
        n_nodes: int,
        edge_index: torch.Tensor,
        gt_labels: np.array | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        super().__init__(device)
        self.adj_matrix = adj_matrix
        self.n_nodes = n_nodes
        self.edge_index = edge_index
        self.gt_labels = gt_labels

    @staticmethod
    def create_from_batch_sample(sample: tuple, device: str) -> MISInstanceNumPy:
        """Create a MISInstanceNumPy from a batch sample. The batch must have size 1."""

        node_labels, edge_index, _, _ = MISModel.process_batch(batch=sample)
        edge_index = edge_index.to(device)
        edge_index_np = edge_index.cpu().numpy()
        n_nodes = node_labels.shape[0]

        adj_mat_sparse = coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        ).tocsr()

        return MISInstanceNumPy(adj_mat_sparse, n_nodes, edge_index, node_labels, device)

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

    def get_feasible_from_individual_batch(self, individual: torch.Tensor) -> torch.Tensor:
        individual = individual.cpu().numpy()
        decoded = [mis_decode_np(individual[i], self.adj_matrix) for i in range(individual.shape[0])]
        decoded = np.array(decoded)
        return torch.tensor(decoded, dtype=torch.bool, device=self.device)

    def get_degrees(self) -> torch.Tensor:
        return torch.tensor(self.adj_matrix.sum(axis=1), dtype=torch.int, device=self.device).squeeze()


def create_mis_instance(sample: tuple, device: Literal["cpu", "cuda"] = "cpu", np_eval: bool = True) -> MISInstance:
    if np_eval:
        return MISInstanceNumPy.create_from_batch_sample(sample, device)

    return MISInstance.create_from_batch_sample(sample, device)
