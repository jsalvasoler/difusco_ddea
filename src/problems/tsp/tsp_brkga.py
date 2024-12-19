import numpy as np
import torch
from config.myconfig import Config
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import CrossOver, GaussianMutation
from problems.tsp.tsp_instance import TSPInstance


def create_tsp_problem(instance: TSPInstance, config: Config) -> Problem:
    return Problem(
        objective_func=instance.evaluate_individual,
        objective_sense="min",
        solution_length=instance.n**2,
        device=config.device,
        bounds=(0, 1),
    )


class MatrixQuadrantCrossover(CrossOver):
    def __init__(self, problem: Problem, tournament_size: int = 4) -> None:
        super().__init__(problem, tournament_size=tournament_size)

    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        n = int(self.problem.solution_length ** (1 / 2))
        num_pairings = parents1.shape[0]

        parents1_mat = parents1.view(num_pairings, n, n)
        parents2_mat = parents2.view(num_pairings, n, n)

        # draw a random number between 0 and n for each pair
        s_values = np.random.randint(0, n, size=num_pairings)

        # quadrants are as follows:
        # - top-left: 0:s, 0:s
        # - top-right: 0:s, s:n
        # - bottom-left: s:n, 0:s
        # - bottom-right: s:n, s:n

        # create children
        children_1_mat = parents1_mat.clone()
        children_2_mat = parents2_mat.clone()

        # if s >= n / 2, we swap the top-left quadrant
        for i in range(num_pairings):
            s = s_values[i]

            # if s >= n / 2, we swap the top-left quadrant
            if s >= n / 2:
                children_1_mat[i, :s, :s] = parents2_mat[i, :s, :s]
                children_2_mat[i, :s, :s] = parents1_mat[i, :s, :s]
            else:
                children_1_mat[i, s:, s:] = parents2_mat[i, s:, s:]
                children_2_mat[i, s:, s:] = parents1_mat[i, s:, s:]

        children1 = children_1_mat.view(num_pairings, -1)
        children2 = children_2_mat.view(num_pairings, -1)

        # Stack the children tensors one on top of the other
        children = torch.cat([children1, children2], dim=0)
        assert children.shape[0] == 2 * num_pairings

        # Write the children solutions into a new SolutionBatch, and return the new batch
        return self._make_children_batch(children)


def create_tsp_brkga(instance: TSPInstance, config: Config) -> GeneticAlgorithm:
    problem = create_tsp_problem(instance, config)

    return GeneticAlgorithm(
        problem=problem,
        popsize=config.pop_size,
        operators=[
            MatrixQuadrantCrossover(problem, tournament_size=4),
            GaussianMutation(problem, stdev=0.1),
        ],
        re_evaluate=False,
    )
