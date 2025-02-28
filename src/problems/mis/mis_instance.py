from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from ea.problem_instance import ProblemInstance
from problems.mis.mis_evaluation import (
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
        n_nodes: int,
        edge_index: torch.Tensor,
        gt_labels: np.array | None = None,
        adj_matrix_np: csr_matrix | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        super().__init__(device)
        self.n_nodes = n_nodes
        self.edge_index = edge_index.to(device)
        self.gt_labels = gt_labels
        self.device = device

        if adj_matrix_np is None:
            edge_index_np = edge_index.cpu().numpy()
            # create sparse numpy adjacency matrix
            # save it because it is needed for every optimal recombination
            self.adj_matrix_np = coo_matrix(
                (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
            ).tocsr()
        else:
            self.adj_matrix_np = adj_matrix_np

        # Create sparse pytorch adjacency matrix
        # TODO: check if we can simplify this
        values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
        adj_matrix_sparse = (
            torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes), device=device).coalesce().to_sparse_csr()
        )

        neighbors_padded, degrees = precompute_neighbors_padded(adj_matrix_sparse)
        self.neighbors_padded = neighbors_padded.to(device)
        self.degrees = degrees.to(device)

    @staticmethod
    def create_from_batch_sample(sample: tuple, device: str) -> MISInstance:
        """Create a MISInstance from a batch sample. The batch must have size 1."""

        node_labels, edge_index, adj_matrix_np, _ = MISModel.process_batch(batch=sample)

        edge_index = edge_index.to(device)
        n_nodes = node_labels.shape[0]

        return MISInstance(n_nodes, edge_index, node_labels, adj_matrix_np, device)

    def get_gt_cost(self) -> float:
        if self.gt_labels is None:
            raise ValueError("Ground truth labels are not available")
        return self.gt_labels.sum().item()

    def evaluate_individual(self, individual: torch.Tensor) -> float:
        """Individual is a random key of shape (n_nodes,), values in [0, 1]."""
        return mis_decode_torch_batched(individual, self.neighbors_padded, self.degrees).sum()

    def evaluate_solution(self, solution: torch.Tensor) -> float:
        """Solution stored as a torch.Tensor of shape (n_nodes,), where 1 indicates a node in the MIS."""
        return solution.sum()

    def get_feasible_from_individual(self, individual: torch.Tensor) -> torch.Tensor:
        """Individual is a random key of shape (n_nodes,), values in [0, 1]."""
        return mis_decode_torch_batched(individual, self.neighbors_padded, self.degrees)

    def get_feasible_from_individual_batch(self, individual: torch.Tensor) -> torch.Tensor:
        """Individual is a random key of shape (batch_size, n_nodes), values in [0, 1]."""
        return mis_decode_torch_batched(individual, self.neighbors_padded, self.degrees)

    def get_degrees(self) -> torch.Tensor:
        return self.degrees

    def __repr__(self) -> str:
        n_edges = self.edge_index.shape[1] // 2  # Divide by 2 since edges are bidirectional
        return f"MISInstance(n_nodes={self.n_nodes}, n_edges={n_edges}, device='{self.device}')"


def create_mis_instance(sample: tuple, device: Literal["cpu", "cuda"] = "cpu") -> MISInstance:
    return MISInstance.create_from_batch_sample(sample, device)
