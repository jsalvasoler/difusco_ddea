from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import CopyingOperator, CrossOver
from torch import no_grad

from difusco.sampler import DifuscoSampler

if TYPE_CHECKING:
    from config.myconfig import Config
    from problems.mis.mis_instance import MISInstance


class MISGaProblem(Problem):
    def __init__(self, instance: MISInstance, config: Config) -> None:
        self.instance = instance
        self.config = config
        self.config.task = "mis"
        super().__init__(
            objective_func=instance.evaluate_solution,
            objective_sense="max",
            solution_length=instance.n_nodes,
            device=config.device,
            dtype=torch.bool,
        )

    def _fill(self, values: torch.Tensor) -> None:
        if self.config.initialization == "random_feasible":
            return self._fill_random_feasible_initialization(values)
        if self.config.initialization == "difusco_sampling":
            if self.config.device != "cuda":
                error_msg = "Difusco sampling is only supported on CUDA"
                raise ValueError(error_msg)
            return self._fill_difusco_sampling(values)
        error_msg = f"Invalid initialization method: {self.config.initialization}"
        raise ValueError(error_msg)

    def _fill_random_feasible_initialization(self, values: torch.Tensor) -> None:
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

    def _fill_difusco_sampling(self, values: torch.Tensor) -> None:
        """
        Values is a tensor of shape (n_solutions, solution_length).
        Uses Difusco to sample initial solutions.
        """
        sampler = DifuscoSampler(self.config)
        popsize = self.config.pop_size
        assert popsize == values.shape[0], "Population size must match the number of solutions"
        assert (
            self.config.parallel_sampling * self.config.sequential_sampling == popsize
        ), "Population size must match the number of solutions"

        # Sample node scores using Difusco
        node_scores = sampler.sample_mis(
            batch=None,
            n_nodes=self.instance.n_nodes,
            edge_index=self.instance.edge_index,
        )

        # Convert scores to feasible solutions
        for i in range(popsize):
            values[i] = self.instance.get_feasible_from_individual(node_scores[i])


class MISGAMutation(CopyingOperator):
    """
    Gaussian mutation operator.

    Follows the algorithm description in:

        Sean Luke, 2013, Essentials of Metaheuristics, Lulu, second edition
        available for free at http://cs.gmu.edu/~sean/book/metaheuristics/
    """

    def __init__(self, problem: Problem, instance: MISInstance, deselect_prob: float = 0.05) -> None:
        """
        Mutation operator for the Maximum Independent Set problem. With probability deselect_prob, a selected node is
        unselected and gets a probabilty of zero. Solution is then made feasible.

        Args:
            problem: The problem object to work with.
            instance: The instance object to work with.
            deselect_prob: The probability of deselecting a selected node.
        """

        super().__init__(problem)
        self._instance = instance
        self._deselect_prob = deselect_prob

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        deselect_mask = torch.rand(data.shape, device=data.device, dtype=torch.float32) <= self._deselect_prob
        priorities = torch.rand(data.shape, device=data.device, dtype=torch.float32)
        priorities[deselect_mask.bool() & data.bool()] = 0

        for i in range(data.shape[0]):
            if deselect_mask[i].sum() > 0:
                data[i] = self._instance.get_feasible_from_individual(priorities[i])

        return result


class MISGACrossover(CrossOver):
    def __init__(self, problem: Problem, instance: MISInstance, tournament_size: int = 4) -> None:
        super().__init__(problem, tournament_size=tournament_size)
        self._instance = instance

    @no_grad()
    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        """
        Parents are two solutions of shape (num_pairings, n_nodes).
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
            children1[i] = self._instance.get_feasible_from_individual(priority1[i])
            children2[i] = self._instance.get_feasible_from_individual(priority2[i])

        # Combine children into final result
        children = torch.cat([children1, children2], dim=0)

        return self._make_children_batch(children)


def create_mis_ga(instance: MISInstance, config: Config) -> GeneticAlgorithm:
    problem = MISGaProblem(instance, config)

    return GeneticAlgorithm(
        problem=problem,
        popsize=config.pop_size,
        re_evaluate=False,
        operators=[
            MISGACrossover(problem, instance),
            MISGAMutation(problem, instance, deselect_prob=0.2),
        ],
    )
