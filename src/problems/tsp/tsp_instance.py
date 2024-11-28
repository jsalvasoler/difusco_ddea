import numpy as np
import torch
from ea.problem_instance import ProblemInstance
from problems.tsp.tsp_evaluation import TSPTorchEvaluator, cython_merge


class TSPInstance(ProblemInstance):
    """
    Class representing a TSP instance. Responsible for evaluating TSP individuals. Makes sure that all tensors stay
    on the same device as the argument tensors.
    """

    def __init__(self, points: torch.Tensor, gt_tour: torch.Tensor) -> None:
        self.points = points
        self.np_points = points.cpu().numpy()

        self.n = points.shape[0]
        self.gt_tour = gt_tour
        self.tsp_evaluator = TSPTorchEvaluator(points=self.points)

        self._gt_cost = self.evaluate_tsp_route(gt_tour)

    @property
    def gt_cost(self) -> float:
        return self._gt_cost

    def evaluate_tsp_route(self, route: torch.Tensor) -> float:
        return self.tsp_evaluator.evaluate(route)

    def evaluate_individual(self, ind: torch.Tensor) -> float:
        # individual has size self.n ** 2, we reshape it to a matrix
        heatmap = ind.view(self.n, self.n).cpu().numpy()

        # use cython_merge to get the adj matrix of the tour
        adj_mat, _ = cython_merge(self.np_points, heatmap)

        # convert adj matrix to tour
        tour = [0]
        while len(tour) < self.n:
            current_node = tour[-1]
            next_node = np.argmax(adj_mat[current_node])
            tour.append(next_node)
            adj_mat[:, current_node] = 0  # Prevent revisiting the same node
        tour.append(0)

        tour = torch.tensor(tour, device=ind.device)

        # Evaluate the tour using TSPTorchEvaluator
        return self.evaluate_tsp_route(tour)

    def evaluate_solution(self, solution: torch.Tensor) -> float:
        pass

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
