from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from config.configs.tsp_inference import config as tsp_inference_config
from config.myconfig import Config
from evotorch import Problem
from problems.tsp.tsp_evaluation import (
    TSPEvaluator,
    TSPTorchEvaluator,
    evaluate_tsp_route_np,
    evaluate_tsp_route_torch,
)
from problems.tsp.tsp_ga import (
    TSPGACrossover,
    TSPGAProblem,
    TSPTwoOptMutation,
    create_tsp_ga,
)
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from problems.tsp.tsp_instance import TSPInstance, create_tsp_instance
from scipy.spatial import distance_matrix
from torch_geometric.loader import DataLoader


@pytest.fixture
def batch_sample_size_one() -> tuple:
    resource_dir = "tests/resources"
    dataset = TSPGraphDataset(
        data_file=os.path.join(resource_dir, "tsp50_example_dataset_two_samples.txt"),
        sparse_factor=-1,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return next(iter(dataloader))


def test_create_tsp_instance(batch_sample_size_one: tuple) -> None:
    sample = batch_sample_size_one
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)

    assert instance.get_gt_cost() > 0
    assert not instance.sparse
    assert instance.points.shape == (50, 2)
    assert instance.np_points.shape == (50, 2)
    assert instance.edge_index is None  # dense graph
    assert instance.dist_mat.shape == (50, 50)
    assert instance.n == 50
    assert instance.gt_tour.shape == (51,)
    assert instance.gt_tour.min() == 0
    assert instance.gt_tour.max() == 49


def test_tsp_instance_eval_methods(batch_sample_size_one: tuple) -> None:
    sample = batch_sample_size_one
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)

    # Compare with hand-calculated cost
    tour = instance.gt_tour.cpu().numpy()
    points = sample[1][0].cpu().numpy()
    dist_mat = distance_matrix(points, points)

    calc_cost = 0
    for i in range(len(tour) - 1):
        calc_cost += dist_mat[tour[i], tour[i + 1]]

    assert abs(calc_cost - instance.get_gt_cost()) < 1e-4

    # Compare with TSPEvaluator
    tsp_evaluator = TSPEvaluator(points)

    assert abs(tsp_evaluator.evaluate(tour) - instance.get_gt_cost()) < 1e-4

    # Compare with evaluate_tsp_route_np
    assert (
        abs(
            evaluate_tsp_route_np(tsp_evaluator.dist_mat, tour) - instance.get_gt_cost()
        )
        < 1e-4
    )

    # Compare with TSPTorchEvaluator
    points = sample[1][0]
    tour = torch.tensor(tour)
    tsp_torch_evaluator = TSPTorchEvaluator(points)
    assert abs(tsp_torch_evaluator.evaluate(tour) - instance.get_gt_cost()) < 1e-4

    # Compare with evaluate_tsp_route_torch
    assert (
        abs(
            evaluate_tsp_route_torch(tsp_torch_evaluator.dist_mat, tour)
            - instance.get_gt_cost()
        )
        < 1e-4
    )


def test_tsp_problem_evaluation() -> None:
    instance = TSPInstance(
        points=torch.from_numpy(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        ),
        edge_index=None,
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


def test_valid_tour(batch_sample_size_one: tuple) -> None:
    sample = batch_sample_size_one
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)

    assert instance.is_valid_tour(instance.gt_tour)

    # create a tour with a duplicate city
    instance.n = 5
    tour = torch.tensor([0, 1, 2, 3, 3, 0])
    assert not instance.is_valid_tour(tour)

    # create a tour with wrong dtype
    tour = torch.tensor([0, 1, 2, 3.4, 7, 0], dtype=torch.float)
    assert not instance.is_valid_tour(tour)

    # create a tour with a city not in the instance
    tour = torch.tensor([0, 1, 2, 3, 7, 0])
    assert not instance.is_valid_tour(tour)


def test_tsp_ga_fill_random_feasible(batch_sample_size_one: tuple) -> None:
    sample = batch_sample_size_one
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)
    problem = TSPGAProblem(
        instance, Config(pop_size=10, device="cpu", initialization="random_feasible")
    )

    values = torch.zeros(10, instance.n + 1, dtype=torch.int64)
    problem._fill(values)  # noqa: SLF001 for testing

    assert values.shape == (10, instance.n + 1)
    assert values.dtype == torch.int64

    assert (values.sum(dim=1) > 0).all()

    evaluations = [
        instance.evaluate_tsp_route(values[i]) for i in range(values.shape[0])
    ]
    assert all(eval_ > 0 for eval_ in evaluations)

    # make sure that the routes are valid and that they match the problem's evaluation
    for i in range(values.shape[0]):
        assert instance.is_valid_tour(values[i])
        assert evaluations[i] == problem._objective_func(values[i])  # noqa: SLF001


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available, skipping test that requires GPU",
)
def test_tsp_ga_fill_difusco_sampling(batch_sample_size_one: tuple) -> None:
    sample = batch_sample_size_one
    instance = create_tsp_instance(sample, device="cuda", sparse_factor=-1)

    config = Config(
        parallel_sampling=2,
        sequential_sampling=1,
        diffusion_steps=2,
        inference_diffusion_steps=2,
        pop_size=2,
        initialization="difusco_sampling",
        ckpt_path="tsp/tsp50_categorical.ckpt",
        training_split="tsp/tsp50_train_concorde.txt",
        test_split="tsp/tsp50_test_concorde.txt",
        validation_split="tsp/tsp50_test_concorde.txt",
        task="tsp",
        device="cuda",
        sparse_factor=-1,
    )

    config = tsp_inference_config.update(config)

    problem = TSPGAProblem(instance, config)

    values = torch.zeros(2, instance.n + 1, dtype=torch.int64)
    problem._fill(values)  # noqa: SLF001 for testing

    assert values.shape == (2, instance.n + 1)
    assert values.dtype == torch.int64

    assert (values.sum(dim=1) > 0).all()

    evaluations = [
        instance.evaluate_tsp_route(values[i]) for i in range(values.shape[0])
    ]
    assert all(eval_ > 0 for eval_ in evaluations)

    # make sure that the routes are valid and that they match the problem's evaluation
    for i in range(values.shape[0]):
        assert instance.is_valid_tour(values[i])
        assert evaluations[i] == problem._objective_func(values[i])  # noqa: SLF001


def test_tsp_ga_mutation(batch_sample_size_one: tuple) -> None:
    sample = batch_sample_size_one
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)

    ga = create_tsp_ga(
        instance,
        config=Config(
            pop_size=10,
            device="cpu",
            max_two_opt_it=2,
            initialization="random_feasible",
        ),
    )

    mutation = ga._operators[1]  # noqa: SLF001
    assert isinstance(mutation, TSPTwoOptMutation)

    parents = deepcopy(ga.population)
    children = mutation._do(ga.population)  # noqa: SLF001
    assert children.values.shape == ga.population.values.shape
    assert children.values.dtype == torch.int64

    # make sure that no individual has worsened
    for i in range(children.values.shape[0]):
        assert instance.evaluate_tsp_route(
            children.values[i]
        ) <= instance.evaluate_tsp_route(parents.values[i])
        assert instance.is_valid_tour(children.values[i])


def test_tsp_ga_crossover_works(batch_sample_size_one: tuple) -> None:
    sample = batch_sample_size_one
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)

    ga = create_tsp_ga(
        instance,
        config=Config(
            pop_size=10,
            device="cpu",
            max_two_opt_it=2,
            initialization="random_feasible",
        ),
    )

    crossover = ga._operators[0]  # noqa: SLF001
    assert isinstance(crossover, TSPGACrossover)

    parents = deepcopy(ga.population)
    children = crossover._do_cross_over(parents.values[:5], parents.values[5:])  # noqa: SLF001
    assert children.values.shape == parents.values.shape
    assert children.values.dtype == torch.int64

    # make sure that children :5 and children 5: are different
    assert (children.values[:5] != children.values[5:]).any()

    # make sure that children and parents are different
    assert (children.values != parents.values).any()

    evaluations_1 = [
        instance.evaluate_tsp_route(children.values[i])
        for i in range(children.values.shape[0])
    ]

    # make sure that children can be evaluated
    for i in range(children.values.shape[0]):
        assert evaluations_1[i] > 0
        assert instance.is_valid_tour(children.values[i])


def test_tsp_ga_runs(batch_sample_size_one: tuple) -> None:
    sample = batch_sample_size_one
    instance = create_tsp_instance(sample, device="cpu", sparse_factor=-1)

    ga = create_tsp_ga(
        instance,
        config=Config(
            pop_size=10,
            device="cpu",
            max_two_opt_it=5,
            initialization="random_feasible",
        ),
    )

    ga.run(num_generations=0)

    # manually compute evaluations
    population = ga.population.values
    evaluations_0 = [
        instance.evaluate_tsp_route(population[i]) for i in range(population.shape[0])
    ]

    ga.run(num_generations=1)
    evaluations_1 = [
        instance.evaluate_tsp_route(ga.population.values[i])
        for i in range(ga.population.values.shape[0])
    ]

    assert ga.status["pop_best_eval"] == min(evaluations_1)

    assert sorted(evaluations_1) != sorted(evaluations_0)
