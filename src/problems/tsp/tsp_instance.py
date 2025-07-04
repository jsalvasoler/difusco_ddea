from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.sparse
import torch
from ea.problem_instance import ProblemInstance
from problems.tsp.tsp_evaluation import (
    adj_mat_to_tour,
    cython_merge,
    evaluate_tsp_route_torch,
)
from problems.tsp.tsp_operators import (
    batched_two_opt_torch,
    edge_recombination_crossover,
)

from difusco.tsp.pl_tsp_model import TSPModel

if TYPE_CHECKING:
    import numpy as np


class TSPInstance(ProblemInstance):
    """
    Class representing a TSP instance. Responsible for evaluating TSP individuals. Makes sure that all tensors stay
    on the same device as the argument tensors.
    """

    def __init__(
        self,
        points: torch.Tensor,
        edge_index: torch.Tensor | None,
        gt_tour: torch.Tensor,
    ) -> None:
        self.sparse = edge_index is not None

        self.points = points
        self.np_points = points.cpu().numpy()
        self.edge_index = edge_index
        self.np_edge_index = (
            edge_index.cpu().numpy() if edge_index is not None else None
        )
        self.device = points.device
        self.n = points.shape[0]
        self.gt_tour = gt_tour
        self.dist_mat = torch.cdist(points, points)
        self.gt_cost = self.evaluate_tsp_route(self.gt_tour)

    @staticmethod
    def create_from_batch_sample(
        sample: tuple, device: str, sparse_factor: int
    ) -> TSPInstance:
        """Create a TSPInstance from a batch sample. The batch must have size 1, i.e. a single sample."""
        check_idx = 3 if sparse_factor > 0 else 1
        assert sample[check_idx].shape[0] == 1, "Batch must have size 1"

        if sparse_factor <= 0:
            _, edge_index, _, points, _, _, gt_tour = TSPModel.process_dense_batch(
                sample
            )
            points = points[0].to(device)
        else:
            _, edge_index, _, points, _, _, gt_tour = TSPModel.process_sparse_batch(
                sample
            )
            edge_index = edge_index.to(device)
            points = points.to(device)

        gt_tour = torch.from_numpy(gt_tour).to(device)

        return TSPInstance(points, edge_index, gt_tour)

    def get_gt_cost(self) -> float:
        return self.gt_cost

    def evaluate_tsp_route(self, route: torch.Tensor) -> float:
        return evaluate_tsp_route_torch(self.dist_mat, route)

    def two_opt_mutation(
        self, routes: torch.Tensor, max_iterations: int
    ) -> torch.Tensor:
        """Routes is a tensor of shape (n_solutions, n + 1)"""
        tours, _ = batched_two_opt_torch(
            self.points, routes, max_iterations=max_iterations, device=self.device
        )
        return tours

    def get_tour_from_adjacency_np_heatmap(self, heatmap: np.ndarray) -> torch.Tensor:
        """
        If sparse, heatmap is an np.array of shape (n, n).
        If dense, heatmap is an np.array of shape (n, n_edges).

        Returns the tour of size n + 1, with the last value being the first value.
        """
        if self.sparse:
            # convert to sparse adjacency matrix
            adj_mat = (
                scipy.sparse.coo_matrix(
                    (heatmap, (self.np_edge_index[0], self.np_edge_index[1])),
                ).toarray()
                + scipy.sparse.coo_matrix(
                    (heatmap, (self.np_edge_index[1], self.np_edge_index[0])),
                ).toarray()
            )
        else:
            adj_mat = heatmap

        solved_adj_mat, _ = cython_merge(self.np_points, adj_mat)
        tour = adj_mat_to_tour(solved_adj_mat)
        return torch.tensor(tour, device=self.device)

    def edge_recombination_crossover(
        self, parents1: torch.Tensor, parents2: torch.Tensor
    ) -> torch.Tensor:
        """
        Edge recombination crossover for a batch of parents.

        Args:
            parents1: Tensor of size (batch_size, n + 1), first parent tours.
            parents2: Tensor of size (batch_size, n + 1), second parent tours.

        Returns:
            offspring: Tensor of size (batch_size, n + 1), containing offspring tours.
        """
        return edge_recombination_crossover(parents1, parents2)

    def evaluate_individual(self, ind: torch.Tensor) -> float:
        # individual has size self.n ** 2, we reshape it to a matrix
        heatmap = ind.view(self.n, self.n).cpu().numpy()

        # use cython_merge to get the adj matrix of the tour
        adj_mat, _ = cython_merge(self.np_points, heatmap)

        # convert adj matrix to tour
        tour = adj_mat_to_tour(adj_mat)
        tour.append(0)

        tour = torch.tensor(tour, device=ind.device)

        # Evaluate the tour using TSPTorchEvaluator
        return self.evaluate_tsp_route(tour)

    def evaluate_solution(self, solution: torch.Tensor) -> float:
        return self.evaluate_tsp_route(solution)

    def get_feasible_from_individual(self, individual: torch.Tensor) -> torch.Tensor:
        pass

    def is_valid_tour(self, tour: torch.Tensor) -> bool:
        """A tour is a tensor of size (n + 1,)"""
        if tour.shape[0] != self.n + 1:
            return False

        # start and end city must be the same
        if not (tour[0] == tour[-1]).item():
            return False

        # check no duplicate cities
        if tour.unique(dim=0).shape[0] != self.n:
            return False

        # check that all elements are int64
        if tour.dtype != torch.int64:
            return False

        # check that all elements are in the range [0, n)
        if not (tour >= 0).all():
            return False
        if not (tour <= self.n).all():
            return False

        return True


def create_tsp_instance(sample: tuple, device: str, sparse_factor: int) -> TSPInstance:
    """Create a TSPInstance from a sample. A sample is a batch of size 1"""
    return TSPInstance.create_from_batch_sample(sample, device, sparse_factor)
