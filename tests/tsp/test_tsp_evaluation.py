import time

import numpy as np
import torch
from problems.tsp.tsp_evaluation import (
    TSPEvaluator,
    TSPTorchEvaluator,
    adj_mat_to_tour,
    cython_merge,
    cython_merge_get_tour,
)

from tests.resources.tsp_merge_python import merge_python


def test_merge_python_is_same_as_cython() -> None:
    # Example usage with simple coordinates
    coords = np.array([[0, 0], [1, 1], [2, 0], [1, -1]], dtype=float)
    # set random seed for reproducibility
    np.random.seed(0)
    adj_mat = np.random.rand(4, 4)

    # Symmetrize adj_mat to make it suitable for a tour
    adj_mat = (adj_mat + adj_mat.T) / 2

    A_1, iterations_1 = merge_python(coords, adj_mat)
    print("\nResulting Adjacency Matrix:")
    print(A_1)
    print(f"Total merge iterations: {iterations_1}")

    tour_1 = adj_mat_to_tour(A_1)
    print("\nExtracted Tour:")
    print(tour_1)

    # now use merge_cython
    A_2, iterations_2 = cython_merge(coords, adj_mat)
    print("\nResulting Adjacency Matrix:")
    print(A_2)
    print(f"Total merge iterations: {iterations_2}")

    tour_2 = adj_mat_to_tour(A_2)
    print("\nExtracted Tour:")
    print(tour_2)

    assert iterations_1 == iterations_2
    assert tour_1 == tour_2
    assert np.array_equal(A_1, A_2)


def test_cython_merge_solves_correctly() -> None:
    # Example usage with simple coordinates
    coords = np.array([[0, 0], [1, 1], [2, 0], [1, -1]], dtype=float)
    # set random seed for reproducibility
    np.random.seed(0)
    adj_mat = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=float)

    # add 0.001 everywhere except where there are ones
    adj_mat[adj_mat == 0] = 0.001

    A, iterations = merge_python(coords, adj_mat)
    print("\nResulting Adjacency Matrix:")
    print(A)
    print(f"Total merge iterations: {iterations}")

    tour = adj_mat_to_tour(A)
    print("\nExtracted Tour:")
    print(tour)
    assert tour in ([0, 1, 2, 3, 0], [0, 3, 2, 1, 0])


def test_numpy_heatmaps() -> None:
    file = "tests/resources/test-heatmap-size50.npy"
    heatmap = np.load(file)
    print(heatmap.shape)
    print(heatmap)
    print(heatmap.max())
    print(heatmap.min())
    print(heatmap.mean())
    assert heatmap.shape == (1, 50, 50)


def test_numpy_points() -> None:
    file = "tests/resources/test-points-size50.npy"
    points = np.load(file)
    print(points.shape)
    print(points)
    print(points.max())
    print(points.min())
    print(points.mean())
    assert points.shape == (50, 2)


def test_tsp_evaluator() -> None:
    file = "tests/resources/test-points-size50.npy"
    points = np.load(file)
    tsp_eval = TSPEvaluator(points)
    route = list(range(50))
    cost = tsp_eval.evaluate(route)

    # calculated cost
    calc_cost = 0
    for i in range(len(route) - 1):
        calc_cost += tsp_eval.dist_mat[route[i], route[i + 1]]

    assert cost == calc_cost


def test_tsp_evaluator_with_np_array() -> None:
    file = "tests/resources/test-points-size50.npy"
    points = np.load(file)
    tsp_eval = TSPEvaluator(points)
    route = np.array(list(range(50)))
    cost = tsp_eval.evaluate(route)

    # calculated cost
    calc_cost = 0
    for i in range(len(route) - 1):
        calc_cost += tsp_eval.dist_mat[route[i], route[i + 1]]

    assert cost == calc_cost


def test_tsp_torch_evaluator() -> None:
    file = "tests/resources/test-points-size50.npy"
    points = np.load(file)
    points = torch.tensor(points)
    tsp_eval = TSPTorchEvaluator(points)
    route = torch.tensor(list(range(50)))
    cost = tsp_eval.evaluate(route)

    # calculated cost
    calc_cost = 0
    for i in range(len(route) - 1):
        calc_cost += tsp_eval.dist_mat[route[i], route[i + 1]]

    assert cost == calc_cost


def test_cython_merge_get_tour() -> None:
    # Example usage with simple coordinates
    coords = np.array([[0, 0], [1, 1], [2, 0], [1, -1]], dtype=float)
    # set random seed for reproducibility
    np.random.seed(0)
    adj_mat = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=float)

    tour, merge_iterations = cython_merge_get_tour(coords, adj_mat)
    print(tour)
    print(merge_iterations)
    assert (tour == [0, 1, 2, 3, 0]).all() or (tour == [0, 3, 2, 1, 0]).all()


def test_cython_merge_get_tour_vs_python() -> None:
    N = 500

    points = np.random.rand(N, 2)

    # create a random adj_mat of size 50
    adj_mat = np.random.rand(N, N)
    adj_mat = (adj_mat + adj_mat.T) / 2
    adj_mat[adj_mat == 0] = 0.001

    # benchmark cython_merge_get_tour
    start_time = time.time()
    tour1, _ = cython_merge_get_tour(points, adj_mat)
    end_time = time.time()
    print(f"Cython merge get tour time: {end_time - start_time} seconds")

    # benchmark python merge_python
    start_time = time.time()
    adj_mat, _ = cython_merge(points, adj_mat)
    tour2 = adj_mat_to_tour(adj_mat)
    end_time = time.time()
    print(f"Python merge get tour time: {end_time - start_time} seconds")

    assert (tour1 == tour2).all()

    assert tour1[-1] == tour2[-1] == 0


if __name__ == "__main__":
    test_cython_merge_get_tour_vs_python()
