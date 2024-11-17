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
