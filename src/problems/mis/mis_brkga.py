from __future__ import annotations

from typing import TYPE_CHECKING

from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, OnePointCrossOver

if TYPE_CHECKING:
    from ea.config import Config
    from problems.mis.mis_instance import MISInstance


def create_mis_brkga(instance: MISInstance, config: Config) -> GeneticAlgorithm:
    problem = Problem(
        objective_func=instance.evaluate_individual,
        objective_sense="max",
        solution_length=instance.n_nodes,
        bounds=(0, 1),
        num_actors=config.n_parallel_evals,
        device=config.device,
    )

    return GeneticAlgorithm(
        problem=problem,
        popsize=config.pop_size,
        operators=[
            OnePointCrossOver(problem, tournament_size=4),
            GaussianMutation(problem, stdev=0.1),
        ],
        re_evaluate=False,
    )
