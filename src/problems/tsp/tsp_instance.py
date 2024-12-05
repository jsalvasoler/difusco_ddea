import numpy as np
import torch
from ea.problem_instance import ProblemInstance
from problems.tsp.tsp_evaluation import TSPTorchEvaluator, adj_mat_to_tour, cython_merge
from problems.tsp.tsp_operators import batched_two_opt_torch, edge_recombination_crossover


class TSPInstance(ProblemInstance):
    """
    Class representing a TSP instance. Responsible for evaluating TSP individuals. Makes sure that all tensors stay
    on the same device as the argument tensors.
    """

    def __init__(self, points: torch.Tensor, gt_tour: torch.Tensor) -> None:
        self.points = points
        self.np_points = points.cpu().numpy()
        self.device = points.device

        self.n = points.shape[0]
        self.gt_tour = gt_tour
        self.tsp_evaluator = TSPTorchEvaluator(points=self.points)

        self._gt_cost = self.evaluate_tsp_route(gt_tour)

    @property
    def gt_cost(self) -> float:
        return self._gt_cost

    def evaluate_tsp_route(self, route: torch.Tensor) -> float:
        return self.tsp_evaluator.evaluate(route)

    def two_opt_mutation(self, routes: torch.Tensor, max_iterations: int) -> torch.Tensor:
        """Routes is a tensor of shape (n_solutions, n + 1)"""
        tours, _ = batched_two_opt_torch(self.points, routes, max_iterations=max_iterations, device=self.device)
        return tours

    def get_tour_from_adjacency_np_heatmap(self, heatmap: np.ndarray) -> torch.Tensor:
        """
        Heatmap is a tensor of shape (n, n).

        Returns the tour of size n + 1, with the last value being the first value.
        """
        adj_mat, _ = cython_merge(self.np_points, heatmap)
        tour = adj_mat_to_tour(adj_mat)
        return torch.tensor(tour, device=self.device)

    def edge_recombination_crossover(self, parents1: torch.Tensor, parents2: torch.Tensor) -> torch.Tensor:
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
        tour = self.adj_mat_to_tour(adj_mat)
        tour.append(0)

        tour = torch.tensor(tour, device=ind.device)

        # Evaluate the tour using TSPTorchEvaluator
        return self.evaluate_tsp_route(tour)

    def evaluate_solution(self, solution: torch.Tensor) -> float:
        return self.evaluate_tsp_route(solution)

    def get_feasible_from_individual(self, individual: torch.Tensor) -> torch.Tensor:
        pass


def create_tsp_instance(sample: tuple, device: str, sparse_factor: int) -> TSPInstance:
    """Create a TSPInstance from a sample. A sample is a batch of size 1"""

    if sparse_factor <= 0:
        _, points, _, tour = sample
        points, tour = points[0], tour[0]
        return TSPInstance(points.to(device), tour.to(device))

    _, points, _, _, tour = sample
    points, tour = points[0], tour[0]
    return TSPInstance(points.to(device), tour.to(device))
