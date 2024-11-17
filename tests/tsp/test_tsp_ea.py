import os
from unittest.mock import patch

import numpy as np
import torch
from ea.config import Config
from ea.evolutionary_algorithm import dataset_factory
from evotorch import Problem
from problems.tsp.tsp_ea import (
    MatrixQuadrantCrossover,
    TSPInstance,
    create_tsp_ea,
    create_tsp_instance,
    create_tsp_problem,
)
from problems.tsp.tsp_evaluation import TSPEvaluator
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from scipy.spatial import distance_matrix
from torch_geometric.loader import DataLoader


def get_tsp_sample() -> TSPInstance:
    resource_dir = "tests/resources"
    dataset = TSPGraphDataset(
        data_file=os.path.join(resource_dir, "tsp50_example_dataset_two_samples.txt"), sparse_factor=-1
    )

    item = dataset.__getitem__(0)
    _, points, _, tour = item
    # make a batch of size one for points and tour
    points = torch.stack([points])
    tour = torch.stack([tour])
    return 0, points, 0, tour


def test_create_tsp_instance() -> None:
    sample = get_tsp_sample()
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)

    assert instance.gt_cost > 0
    assert instance.points.shape == (50, 2)
    assert instance.n == 50
    assert instance.gt_tour.shape == (51,)
    assert instance.gt_tour.min() == 0
    assert instance.gt_tour.max() == 49

    # Compare with hand-calculated cost
    tour = instance.gt_tour.cpu().numpy()
    points = sample[1][0].cpu().numpy()
    dist_mat = distance_matrix(points, points)

    calc_cost = 0
    for i in range(len(tour) - 1):
        calc_cost += dist_mat[tour[i], tour[i + 1]]

    assert abs(calc_cost - instance.gt_cost) < 1e-4

    # Compare with TSPEvaluator
    tsp_evaluator = TSPEvaluator(points)

    assert abs(tsp_evaluator.evaluate(tour) - instance.gt_cost) < 1e-4


@patch("numpy.random.randint", return_value=np.array([2]))
def test_matrix_quadrant_crossover_one_pair(mock_randint) -> None:  # noqa: ANN001
    prob = Problem(
        objective_func=lambda x: x.sum(),
        objective_sense="min",
        solution_length=4**2,
        device="cpu",
    )

    crossover = MatrixQuadrantCrossover(prob, tournament_size=4)

    parents1 = torch.zeros(4**2).view(1, -1)
    parents2 = torch.ones(4**2).view(1, -1)

    children = crossover._do_cross_over(parents1, parents2)  # noqa: SLF001
    # Check if the random value was patched correctly
    assert mock_randint.called
    assert mock_randint.call_args[0] == (0, 4)

    assert children.values_shape == (2, 4**2)
    children_1 = children.values[0].view(4, 4).numpy()
    children_2 = children.values[1].view(4, 4).numpy()

    expected_children_1 = np.array(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    expected_children_2 = np.array(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )

    assert (children_1 == expected_children_1).all()
    assert (children_2 == expected_children_2).all()


@patch("numpy.random.randint", return_value=np.array([2, 2]))
def test_matrix_quadrant_crossover_two_pairs(mock_randint) -> None:  # noqa: ANN001
    prob = Problem(
        objective_func=lambda x: x.sum(),
        objective_sense="min",
        solution_length=4**2,
        device="cpu",
    )

    crossover = MatrixQuadrantCrossover(prob, tournament_size=4)

    parents1 = torch.zeros(4**2)
    parents2 = torch.ones(4**2)

    # stack two pairs of parents
    parents1 = torch.stack([parents1, parents1])
    parents2 = torch.stack([parents2, parents2])

    children = crossover._do_cross_over(parents1, parents2)  # noqa: SLF001
    # Check if the random value was patched correctly
    assert mock_randint.called
    assert mock_randint.call_args[0] == (0, 4)

    assert children.values_shape == (4, 4**2)
    children_10 = children.values[0].view(4, 4).numpy()
    children_11 = children.values[1].view(4, 4).numpy()
    children_20 = children.values[2].view(4, 4).numpy()
    children_21 = children.values[3].view(4, 4).numpy()

    expected_children_1 = np.array(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    expected_children_2 = np.array(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )

    assert (children_10 == expected_children_1).all()
    assert (children_11 == expected_children_1).all()
    assert (children_20 == expected_children_2).all()
    assert (children_21 == expected_children_2).all()


def test_tsp_problem_evaluation() -> None:
    instance = TSPInstance(
        points=torch.from_numpy(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])),
        gt_tour=torch.tensor([0, 1, 2, 3, 0]),
    )
    problem = Problem(
        objective_func=instance.evaluate_individual,
        objective_sense="min",
        solution_length=4**2,
        device="cpu",
    )

    # create the adjacency matrix of 0 -> 1 -> 2 -> 3 -> 0
    ind = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    ind = torch.from_numpy(ind).view(-1)
    obj = instance.evaluate_individual(ind)
    assert obj == problem._objective_func(ind) == 4  # noqa: SLF001

    # create the adjacency matrix of 0 -> 2 -> 1 -> 3 -> 0
    ind = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
    ind = torch.from_numpy(ind).view(-1)
    obj = instance.evaluate_individual(ind)
    assert obj == problem._objective_func(ind) == 2 * 2**0.5 + 2  # noqa: SLF001


def test_problem_evaluation_on_tsp_instance() -> None:
    sample = get_tsp_sample()
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)
    problem = create_tsp_problem(instance, config=Config(device="cpu"))

    # create random array of size n**2, clip it to [0, 1]
    ind = torch.rand(instance.n**2)

    # evaluate the individual
    obj = instance.evaluate_individual(ind)
    assert obj == problem._objective_func(ind)  # noqa: SLF001

    # evaluate the ground truth tour
    obj_gt = instance.evaluate_tsp_route(instance.gt_tour)
    assert obj_gt <= obj


def test_tsp_ga_runs() -> None:
    sample = get_tsp_sample()
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)

    ga = create_tsp_ea(instance, config=Config(pop_size=10, device="cpu"))
    ga.run(num_generations=2)

    status = ga.status
    assert status["iter"] == 2


def test_tsp_ga_runs_with_dataloader() -> None:
    dataset = dataset_factory(
        config=Config(
            data_path="tests/resources",
            test_split="tsp50_example_dataset_two_samples.txt",
            test_split_label_dir=None,
            task="tsp",
        )
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for sample in dataloader:
        instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)
        ga = create_tsp_ea(instance, config=Config(pop_size=10, device="cpu"))
        ga.run(num_generations=2)

        status = ga.status
        assert status["iter"] == 2
        break
