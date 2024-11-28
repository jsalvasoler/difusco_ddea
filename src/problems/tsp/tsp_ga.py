from __future__ import annotations

from typing import TYPE_CHECKING

from evotorch.algorithms import GeneticAlgorithm
from problems.tsp.tsp_instance import TSPInstance

if TYPE_CHECKING:
    from ea.config import Config


def create_tsp_ga(instance: TSPInstance, config: Config) -> GeneticAlgorithm:
    pass
