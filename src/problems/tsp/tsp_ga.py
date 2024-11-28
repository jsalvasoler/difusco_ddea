from __future__ import annotations

from typing import TYPE_CHECKING

from evotorch import Problem

if TYPE_CHECKING:
    from ea.config import Config
    from evotorch.algorithms import GeneticAlgorithm
    from problems.tsp.tsp_instance import TSPInstance


def create_tsp_ga(instance: TSPInstance, config: Config) -> GeneticAlgorithm:
    problem = Problem(  # noqa: F841
        objective_func=instance.evaluate_individual,
        objective_sense="min",
        solution_length=instance.n**2,
        device=config.device,
        bounds=(0, 1),
    )
