from abc import ABC, abstractmethod

import torch


class ProblemInstance(ABC):
    @property
    @abstractmethod
    def gt_cost(self) -> float:
        pass

    @abstractmethod
    def evaluate_individual(self, individual: torch.Tensor) -> float:
        pass

    @abstractmethod
    def evaluate_solution(self, solution: torch.Tensor) -> float:
        pass

    @abstractmethod
    def get_feasible_from_individual(self, individual: torch.Tensor) -> torch.Tensor:
        pass
