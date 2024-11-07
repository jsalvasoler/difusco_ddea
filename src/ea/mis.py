from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, OnePointCrossOver

from difusco.mis.utils import mis_decode_np

if TYPE_CHECKING:
    from argparse import Namespace

    import torch


class MISInstance:
    def __init__(self, adj_matrix: sp.coo_matrix, n_nodes: int, gt_labels: np.array | None = None) -> None:
        self.adj_matrix = adj_matrix
        self.n_nodes = n_nodes
        self.gt_labels = gt_labels

    def evaluate_mis_individual(self, ind: torch.Tensor) -> float:
        return mis_decode_np(ind.cpu().numpy(), self.adj_matrix).sum()


def create_mis_ea(instance: MISInstance, args: Namespace) -> GeneticAlgorithm:
    problem = Problem(
        objective_func=instance.evaluate_mis_individual,
        objective_sense="max",
        solution_length=instance.n_nodes,
        bounds=(0, 1),
    )

    return GeneticAlgorithm(
        problem=problem,
        popsize=args.pop_size,
        operators=[
            OnePointCrossOver(problem, tournament_size=4),
            GaussianMutation(problem, stdev=0.1),
        ],
        re_evaluate=False,
    )


def create_mis_instance(sample: tuple) -> MISInstance:
    _, graph_data, _ = sample

    edge_index_np = graph_data.edge_index.cpu().reshape(2, -1).numpy()
    adj_mat = sp.coo_matrix(
        (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
    ).tocsr()

    return MISInstance(adj_mat, graph_data.x.shape[0], graph_data.x)
