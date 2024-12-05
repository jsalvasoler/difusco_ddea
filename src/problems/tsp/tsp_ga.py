from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from evotorch import Problem

if TYPE_CHECKING:
    from ea.config import Config
    from evotorch.algorithms import GeneticAlgorithm
    from problems.tsp.tsp_instance import TSPInstance


class TSPGAProblem(Problem):
    def __init__(self, instance: TSPInstance, config: Config) -> None:
        self.instance = instance
        super().__init__(
            objective_func=instance.evaluate_solution,
            objective_sense="min",
            solution_length=instance.n + 1,
            device=config.device,
            dtype=torch.bool,
        )

    def _fill(self, values: torch.Tensor) -> None:
        """
        Values is a tensor of shape (n_solutions, solution_length).
        """
        for i in range(values.shape[0]):
            if i == 0:
                random_heatmap = np.ones((self.instance.n, self.instance.n))
            else:
                random_heatmap = np.random.rand(self.instance.n, self.instance.n)
            values[i] = self.instance.get_tour_from_adjacency_np_heatmap(random_heatmap)


def create_tsp_ga(instance: TSPInstance, config: Config) -> GeneticAlgorithm:
    problem = TSPGAProblem(instance, config)


