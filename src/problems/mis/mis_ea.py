from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from ea.problem_instance import ProblemInstance
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, OnePointCrossOver
from problems.mis.mis_evaluation import mis_decode_np, mis_decode_torch
from scipy.sparse import coo_matrix

if TYPE_CHECKING:
    from ea.config import Config


class MISInstance(ProblemInstance):
    def __init__(self, adj_matrix: torch.Tensor, n_nodes: int, gt_labels: np.array | None = None) -> None:
        self.adj_matrix = adj_matrix
        self.n_nodes = n_nodes
        self.gt_labels = gt_labels

    @property
    def gt_cost(self) -> float:
        return self.gt_labels.sum()

    def evaluate_individual(self, individual: torch.Tensor) -> float:
        return mis_decode_torch(individual, self.adj_matrix).sum()


class MISInstanceNumPy(ProblemInstance):
    def __init__(self, adj_matrix: coo_matrix, n_nodes: int, gt_labels: np.array | None = None) -> None:
        self.adj_matrix = adj_matrix
        self.n_nodes = n_nodes
        self.gt_labels = gt_labels

    @property
    def gt_cost(self) -> float:
        return self.gt_labels.sum()

    def evaluate_individual(self, individual: torch.Tensor) -> float:
        individual = individual.cpu().numpy()
        return mis_decode_np(individual, self.adj_matrix).sum()


def create_mis_ea(instance: MISInstance, config: Config) -> GeneticAlgorithm:
    problem = Problem(
        objective_func=instance.evaluate_individual,
        objective_sense="max",
        solution_length=instance.n_nodes,
        bounds=(0, 1),
        num_actors=config.n_parallel_evals,
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


def create_mis_instance(sample: tuple, device: Literal["cpu", "cuda"] = "cpu", np_eval: bool = True) -> MISInstance:
    _, graph_data, _ = sample
    edge_index = graph_data.edge_index

    if np_eval:
        edge_index_np = edge_index.cpu().numpy()
        adj_mat_sparse = coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        ).tocsr()
        return MISInstanceNumPy(adj_mat_sparse, graph_data.x.shape[0], graph_data.x)

    values = torch.ones(edge_index.shape[1], dtype=torch.float32)
    adj_mat_sparse = torch.sparse_coo_tensor(
        edge_index, values, (graph_data.x.shape[0], graph_data.x.shape[0]), device=device
    ).coalesce()
    assert adj_mat_sparse.is_sparse

    return MISInstance(adj_mat_sparse, graph_data.x.shape[0], graph_data.x)
