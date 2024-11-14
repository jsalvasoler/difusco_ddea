import torch

from difusco.tsp.utils import TSPTorchEvaluator


class TSPInstance:
    """
    Class representing a TSP instance. Responsible for evaluating TSP individuals. Makes sure that all tensors stay
    on the same device as the argument tensors.
    """

    def __init__(self, points: torch.Tensor, gt_tour: torch.Tensor) -> None:
        self.points = points
        self.n = points.shape[0]
        self.gt_tour = gt_tour
        self.tsp_evaluator = TSPTorchEvaluator(points=self.points)

        self.gt_cost = self.evaluate_tsp_individual(gt_tour)

    def evaluate_tsp_individual(self, route: torch.Tensor) -> float:
        return self.tsp_evaluator.evaluate(route)


def create_tsp_instance(sample: tuple, device: str, sparse_factor: int) -> TSPInstance:
    if sparse_factor <= 0:
        _, points, adj_matrix, tour = sample
        return TSPInstance(points.to(device), tour.to(device))

    _, points, graph_data, adj_matrix, tour = sample
    return TSPInstance(points.to(device), tour.to(device))
