import os

import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch.operators import GaussianMutation, OnePointCrossOver
from scipy import sparse as sp

from difusco.mis.mis_dataset import MISDataset
from difusco.mis.utils import mis_decode_np


def read_mis_instance() -> tuple:
    resource_dir = "tests/resources"
    dataset = MISDataset(
        data_dir=os.path.join(resource_dir, "er_example_dataset"),
        data_label_dir=os.path.join(resource_dir, "er_example_dataset_annotations"),
    )

    _, graph_data, _ = dataset.__getitem__(0)
    node_labels = graph_data.x
    edge_index = graph_data.edge_index

    edge_index = edge_index.to(node_labels.device).reshape(2, -1)
    edge_index_np = edge_index.cpu().numpy()
    adj_mat = sp.coo_matrix(
        (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
    ).tocsr()

    return adj_mat, node_labels


# The solution representation of the MIS problem is a vector x of length |V|, where each element is 0 <= x_i <= 1


class MISInstance:
    def __init__(self, adj_matrix: sp.coo_matrix, n_nodes: int) -> None:
        self._adj_matrix = adj_matrix
        self._n_nodes = n_nodes

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def adj_matrix(self) -> sp.coo_matrix:
        return self._adj_matrix

    def evaluate_mis_individual(self, ind: torch.Tensor) -> float:
        return mis_decode_np(ind.cpu().numpy(), self._adj_matrix).sum()


def main() -> None:
    adj_matrix, node_labels = read_mis_instance()
    instance = MISInstance(adj_matrix, node_labels.shape[0])

    problem = Problem(
        objective_func=instance.evaluate_mis_individual,
        objective_sense="max",
        solution_length=instance.n_nodes,
        bounds=(0, 1),
    )

    ga = GeneticAlgorithm(
        problem=problem,
        popsize=5,
        operators=[
            OnePointCrossOver(problem, tournament_size=4),
            GaussianMutation(problem, stdev=0.1),
        ],
        re_evaluate=False,
    )

    _ = StdOutLogger(ga)  # Report the evolution's progress to standard output
    ga.run(100)  # Run the algorithm for 100 generations
    print("Solution with best fitness ever:", instance.evaluate_mis_individual(ga.status["best"].values))
    print("Current population's best:", instance.evaluate_mis_individual(ga.status["pop_best"].values))


def test_problem_evaluation() -> None:
    adj_matrix, node_labels = read_mis_instance()
    instance = MISInstance(adj_matrix, node_labels.shape[0])

    problem = Problem(
        objective_func=instance.evaluate_mis_individual,
        objective_sense="max",
        solution_length=instance.n_nodes,
    )

    # create ind as a random tensor with size n_nodes
    ind = torch.rand(instance.n_nodes)
    obj = instance.evaluate_mis_individual(ind)
    assert obj == problem._objective_func(ind)  # noqa: SLF001

    # evaluate the labels (ground truth) of the instance
    obj_gt = instance.evaluate_mis_individual(node_labels)
    assert obj_gt == problem._objective_func(node_labels)  # noqa: SLF001

    assert obj_gt == node_labels.sum()
    assert obj_gt >= obj


if __name__ == "__main__":
    main()
    # test_problem_evaluation()
