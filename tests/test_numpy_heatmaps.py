import numpy as np

# import tsp utils
from difusco.tsp.utils import TSPEvaluator


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


if __name__ == "__main__":
    # test_numpy_heatmaps()
    # test_numpy_points()
    test_tsp_evaluator()
    test_tsp_evaluator_with_np_array()
