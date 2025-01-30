from itertools import permutations

import numpy as np
import pytest
import torch
from problems.tsp.tsp_instance import TSPInstance
from problems.tsp.tsp_operators import (
    batched_two_opt_torch,
    build_edge_lists,
    edge_recombination_crossover,
    select_from_edge_lists,
)


@pytest.fixture
def parent_tensors() -> dict:
    batch_size = 4  # Number of parent pairs (output will have double this size)
    n = 5  # Number of cities

    # Create permutations for cities 1 through n-1
    parent1 = torch.stack([torch.randperm(n - 1) for _ in range(batch_size // 2)])
    parent2 = torch.stack([torch.randperm(n - 1) for _ in range(batch_size // 2)])

    # Add 0s at start and end
    parent1 = torch.cat(
        [
            torch.zeros(batch_size // 2, 1, dtype=torch.int),
            parent1 + torch.ones(n - 1, dtype=torch.int),
            torch.zeros(batch_size // 2, 1, dtype=torch.int),
        ],
        dim=-1,
    )
    parent2 = torch.cat(
        [
            torch.zeros(batch_size // 2, 1, dtype=torch.int),
            parent2 + torch.ones(n - 1, dtype=torch.int),
            torch.zeros(batch_size // 2, 1, dtype=torch.int),
        ],
        dim=-1,
    )

    assert parent1.shape == (batch_size // 2, n + 1)
    assert parent2.shape == (batch_size // 2, n + 1)

    assert (parent1[:, :-1] <= n - 1).all()
    assert (parent2[:, :-1] <= n - 1).all()

    return {"parent1": parent1, "parent2": parent2, "batch_size": batch_size, "n": n}


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_batched_two_opt_torch(input_type: str) -> None:
    # Example usage with simple coordinates
    coords = np.array([[0, 0], [1, 1], [2, 0], [1, -1]], dtype=float)
    dist = np.linalg.norm(coords[:, None] - coords, axis=-1)
    # set random seed for reproducibility
    np.random.seed(0)
    adj_mat = np.random.rand(4, 4)
    # Symmetrize adj_mat to make it suitable for a tour
    adj_mat = (adj_mat + adj_mat.T) / 2

    # Compute optimal tour by enumerating all permutations
    def get_optimal_tour(dist: np.ndarray) -> tuple:
        min_cost = np.inf
        min_tour = None
        for perm in permutations(range(1, 4)):
            # compute cost
            tour = [0, *list(perm)]
            cost = sum(dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + dist[tour[-1], tour[0]]
            if cost < min_cost:
                min_cost = cost
                min_tour = tour
        return [*min_tour, 0], min_cost

    min_tour, min_cost = get_optimal_tour(dist)

    print("\nOptimal Tour:")
    print(min_tour)
    print(f"Cost: {min_cost}")

    batched_tours = np.array([[0, 1, 2, 3, 0], [3, 2, 1, 0, 3], [0, 1, 3, 2, 0], [0, 2, 3, 1, 0]], dtype=int)

    if input_type == "torch":
        coords = torch.from_numpy(coords)
        batched_tours = torch.from_numpy(batched_tours)

    solved_tours, iterations = batched_two_opt_torch(coords, batched_tours, max_iterations=1000)
    print("\nSolved Tours:")
    print(solved_tours)
    print(f"Total 2-opt iterations: {iterations}")

    # Convert solved_tours to numpy for comparison if it's a torch tensor
    if isinstance(solved_tours, torch.Tensor):
        assert solved_tours.device.type == "cpu"
        solved_tours = solved_tours.cpu().numpy()

    # assert that all tours equal the min_tour
    edges_min_cost = np.array([min_tour[i : i + 2] for i in range(len(min_tour) - 1)])
    edges_min_cost += edges_min_cost[:, ::-1]
    for tour in solved_tours:
        edges = np.array([tour[i : i + 2] for i in range(len(tour) - 1)])
        edges += edges[:, ::-1]
        assert sorted(edges.tolist()) == sorted(edges_min_cost.tolist()), f"{edges} != {edges_min_cost}"


def test_build_edge_lists(parent_tensors: dict) -> None:
    parent1 = parent_tensors["parent1"]
    parent2 = parent_tensors["parent2"]
    batch_size = parent_tensors["batch_size"]
    n = parent_tensors["n"]

    assert parent1.shape == (batch_size // 2, n + 1)
    assert parent2.shape == (batch_size // 2, n + 1)

    # assert all values in 0 ... n - 1
    assert (parent1[:, :-1] <= n - 1).all()
    assert (parent2[:, :-1] <= n - 1).all()

    edge_lists = build_edge_lists(parent1, parent2)

    print(edge_lists)
    assert edge_lists.shape == (batch_size // 2, n, 4)

    parent1 = parent1.numpy()[:, :-1]
    parent2 = parent2.numpy()[:, :-1]
    for ind_idx in range(parent1.shape[0]):
        for node in range(n - 1):
            parent1_ind = parent1[ind_idx].tolist()
            parent2_ind = parent2[ind_idx].tolist()

            # find where the node is in the parent tours
            node_idx_in_parent1 = parent1_ind.index(node)
            node_idx_in_parent2 = parent2_ind.index(node)
            node_edges = [
                parent1_ind[node_idx_in_parent1 - 1],
                parent1_ind[(node_idx_in_parent1 + 1) % n],
                parent2_ind[node_idx_in_parent2 - 1],
                parent2_ind[(node_idx_in_parent2 + 1) % n],
            ]
            assert sorted(node_edges) == edge_lists[ind_idx, node, :].flatten().tolist()


def test_select_from_edge_lists(parent_tensors: dict) -> None:
    parent1 = parent_tensors["parent1"]
    parent2 = parent_tensors["parent2"]
    batch_size = parent_tensors["batch_size"]
    n = parent_tensors["n"]

    edge_lists = build_edge_lists(parent1, parent2)

    visited = torch.zeros(batch_size // 2, n, dtype=torch.bool)
    first_selection = torch.zeros(batch_size // 2, dtype=int)
    visited[torch.arange(batch_size // 2), first_selection] = True
    selection = select_from_edge_lists(edge_lists, visited, first_selection)

    assert selection.shape == (batch_size // 2,)
    assert (selection >= 1).all()  # cannot select zero as it is already visited
    assert (selection <= n - 1).all()


def test_edge_recombination(parent_tensors: dict) -> None:
    parent1 = parent_tensors["parent1"]
    parent2 = parent_tensors["parent2"]
    batch_size = parent_tensors["batch_size"]
    n = parent_tensors["n"]

    offspring = edge_recombination_crossover(parent1, parent2)

    assert offspring.shape == (batch_size // 2, n + 1)
    assert (offspring[:, :-1] <= n - 1).all()
    assert (offspring[:, :-1] >= 0).all()


def test_edge_recombination_small_example() -> None:
    # n = 9

    instance = TSPInstance(
        points=torch.rand(9, 2),
        edge_index=None,  # dense graph
        gt_tour=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 0]),  # irrelevant for the example
    )

    A = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 0])
    B = torch.tensor([3, 0, 1, 7, 6, 5, 8, 2, 4, 3])
    assert instance.is_valid_tour(A)
    assert instance.is_valid_tour(B)

    # put tensors in a batch
    A = A.unsqueeze(0).to(dtype=torch.int64)
    B = B.unsqueeze(0).to(dtype=torch.int64)

    # make sure edge_list is correct
    edge_lists = build_edge_lists(A, B)
    assert edge_lists.shape == (1, 9, 4)
    assert edge_lists.dtype == torch.int64

    expected_edge_lists = {
        0: [8, 1, 3],
        1: [0, 2, 7],
        2: [1, 3, 8, 4],
        3: [2, 4, 0],
        4: [3, 5, 2],
        5: [4, 6, 8],
        6: [5, 7],
        7: [6, 8, 1],
        8: [7, 0, 5, 2],
    }

    edge_lists_np = edge_lists.numpy()[0]
    # check that sorted and unduplicated lists are equal
    for i in range(edge_lists_np.shape[0]):
        assert sorted(set(edge_lists_np[i])) == sorted(expected_edge_lists[i]), f"Edge list {i} does not match"

    # We start from same node as parent1, which is 0
    visited = torch.zeros(1, 9, dtype=torch.bool)

    expected_selections = [(1, 3), (4,), (2, 5), (6,), (7,), (1, 8), (2,), (8,)]
    assume_selections = [3, 4, 5, 6, 7, 1, 2, 8]
    assert len(expected_selections) == len(assume_selections)
    assert all(ass in exp for (ass, exp) in zip(assume_selections, expected_selections))

    current_node = torch.tensor([0])
    for i in range(len(expected_selections)):
        visited[0, current_node] = True
        selection = select_from_edge_lists(edge_lists, visited, current_node)
        assert selection.shape == (1,)
        assert selection.item() in expected_selections[i], f"step {i} failed"
        current_node = torch.tensor(assume_selections[i])

    assert visited.sum() == 9 - 1
