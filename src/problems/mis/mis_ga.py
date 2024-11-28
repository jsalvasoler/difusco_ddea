from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import CrossOver
from torch import no_grad

if TYPE_CHECKING:
    from ea.config import Config
    from problems.mis.mis_instance import MISInstance


class MISGaProblem(Problem):
    def __init__(self, instance: MISInstance, config: Config) -> None:
        self.instance = instance
        super().__init__(
            objective_func=instance.evaluate_solution,
            objective_sense="max",
            solution_length=instance.n_nodes,
            device=config.device,
            dtype=torch.bool,
        )

    def _fill(self, values: torch.Tensor) -> None:
        """
        Values is a tensor of shape (n_solutions, solution_length).

        Initialization heuristic: (randomized) construction heuristic based on node degree.
        """

        degrees = self.instance.get_degrees()
        inversed_normalized_degrees = 1 - degrees / degrees.max()

        for i in range(values.shape[0]):
            std = 0.2 * i / values.shape[0]  # scales from 0 to 0.2
            noise = torch.randn(self.solution_length, device=self.device) * std

            values[i] = self.instance.get_feasible_from_individual(
                inversed_normalized_degrees + noise,
            )


class MISGACrossover(CrossOver):
    def __init__(self, problem: Problem, instance: MISInstance, tournament_size: int = 4) -> None:
        super().__init__(problem, tournament_size=tournament_size)
        self.instance = instance

    @no_grad()
    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        """
        Parents are two solutions of shape (num_pairings, n_nodes).

        Crossover creates two children:
         - children1: forces the selection of common nodes between parents1 and parents2.
         - children2: forces the selection of the remaining nodes, but penalizes the common nodes.
        """
        num_pairings = parents1.shape[0]
        device = parents1.device

        # Find common nodes between parents
        common_nodes = (parents1 & parents2).bool()  # Element-wise AND

        # Random values between 0 and 0.5, 1 if node forced to selection
        priority1 = torch.rand(num_pairings, self.problem.solution_length, device=device, dtype=torch.float32) * 0.5
        priority1[common_nodes] = 1
        # Random values between 0.5 and 1, 0 if node penalized for selection
        priority2 = (
            torch.rand(num_pairings, self.problem.solution_length, device=device, dtype=torch.float32) * 0.5 + 0.5
        )
        priority2[common_nodes] = 0

        children1 = parents1.clone()
        children2 = parents2.clone()

        for i in range(num_pairings):
            # Get feasible solutions based on priorities
            children1[i] = self.instance.get_feasible_from_individual(priority1[i])
            children2[i] = self.instance.get_feasible_from_individual(priority2[i])

        # Combine children into final result
        children = torch.cat([children1, children2], dim=0)

        return self._make_children_batch(children)


def create_mis_ga(instance: MISInstance, config: Config) -> GeneticAlgorithm:
    problem = MISGaProblem(instance, config)

    return GeneticAlgorithm(
        problem=problem, popsize=config.pop_size, re_evaluate=False, operators=[MISGACrossover(problem, instance)]
    )
