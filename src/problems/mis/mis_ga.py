from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ea.config import Config
    from evotorch.algorithms import GeneticAlgorithm
    from problems.mis.mis_instance import MISInstance

def create_mis_ga(instance: MISInstance, config: Config) -> GeneticAlgorithm:
    pass
