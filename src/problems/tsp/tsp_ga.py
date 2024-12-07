from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import CopyingOperator, CrossOver
from torch import no_grad

if TYPE_CHECKING:
    from ea.config import Config
    from problems.tsp.tsp_instance import TSPInstance


class TSPGAProblem(Problem):
    def __init__(self, instance: TSPInstance, config: Config) -> None:
        self.instance = instance
        super().__init__(
            objective_func=instance.evaluate_solution,
            objective_sense="min",
            solution_length=instance.n + 1,
            device=config.device,
            dtype=torch.int64,
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


class TSPTwoOptMutation(CopyingOperator):
    def __init__(self, problem: TSPGAProblem, instance: TSPInstance, max_iterations: int = 5) -> None:
        super().__init__(problem)
        self._instance = instance
        self._max_iterations = max_iterations

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        data[:] = self._instance.two_opt_mutation(data, max_iterations=self._max_iterations)
        return result


class TSPGACrossover(CrossOver):
    def __init__(self, problem: Problem, instance: TSPInstance, tournament_size: int = 4) -> None:
        super().__init__(problem, tournament_size=tournament_size)
        self._instance = instance

    @no_grad()
    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        """
        Parents are two solutions of shape (batch_size, n_nodes).

        Crossover creates two children:
         - children1: forces the selection of common nodes between parents1 and parents2.
         - children2: forces the selection of the remaining nodes, but penalizes the common nodes.
        """
        # Perform edge recombination crossover twice
        children1 = self._instance.edge_recombination_crossover(parents1, parents2)
        children2 = self._instance.edge_recombination_crossover(parents2, parents1)

        # Combine children into final result
        children = torch.cat([children1, children2], dim=0)

        return self._make_children_batch(children)


def create_tsp_ga(instance: TSPInstance, config: Config) -> GeneticAlgorithm:
    problem = TSPGAProblem(instance, config)

    return GeneticAlgorithm(
        problem=problem,
        popsize=config.pop_size,
        re_evaluate=False,
        operators=[
            TSPGACrossover(problem, instance, tournament_size=4),
            TSPTwoOptMutation(problem, instance, max_iterations=config.max_two_opt_it),
        ],
    )
