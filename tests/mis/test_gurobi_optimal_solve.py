import numpy as np
import pytest
import torch
from problems.mis.mis_instance import MISInstance
from problems.mis.solve_optimal_recombination import (
    LocalBranching,
    get_lil_csr_matrix,
    maximum_weighted_independent_set,
    solve_wmis,
)

from tests.mis.test_mis_ea import read_mis_instance


def test_gurobi_optimal_solve() -> None:
    instance, _ = read_mis_instance()

    ind = torch.rand(2, instance.n_nodes)
    sols = instance.get_feasible_from_individual_batch(ind)
    solution_1, solution_2 = sols[0].numpy().nonzero()[0], sols[1].numpy().nonzero()[0]

    mwis_result = solve_wmis(instance, solution_1, solution_2, time_limit=1)
    assert mwis_result["children_np_labels"].shape == (instance.n_nodes,)

    # since we pass the best solution as starting point, the objective should be at least as good as the best solution
    assert mwis_result["children_obj"] >= max(len(solution_1), len(solution_2))


# Tests for direct use of maximum_weighted_independent_set
@pytest.mark.parametrize(
    "param_type",
    [
        "starting_solution",
        "fix_selection",
        "fix_unselection",
        "local_branching",
        "desired_cost",
    ],
)
def test_direct_mwis_parameters(param_type: str) -> None:
    """Test maximum_weighted_independent_set directly with different parameters."""
    instance, _ = read_mis_instance()

    # Create two random solutions
    ind = torch.rand(2, instance.n_nodes)
    sols = instance.get_feasible_from_individual_batch(ind)
    solution_1, solution_2 = sols[0].numpy().nonzero()[0], sols[1].numpy().nonzero()[0]

    # Prepare common parameters
    adj_matrix = get_lil_csr_matrix(instance.adj_matrix_np)
    weights = np.ones(instance.n_nodes)

    # Configure parameters based on test type
    params = {}

    if param_type == "starting_solution":
        # Test with a starting solution
        starting_solution = np.zeros(instance.n_nodes)
        starting_solution[solution_1] = 1  # Use solution_1 as starting point
        params["starting_solution"] = starting_solution

    elif param_type == "fix_selection":
        # Test with fixed selected nodes (intersection of parents)
        fix_selection = np.intersect1d(solution_1, solution_2)
        if len(fix_selection) == 0:
            # If no intersection, just pick a few nodes from solution_1
            fix_selection = solution_1[: min(5, len(solution_1))]
        params["fix_selection"] = fix_selection

    elif param_type == "fix_unselection":
        # Test with fixed unselected nodes (nodes not in either parent)
        all_nodes = np.arange(instance.n_nodes)
        union = np.union1d(solution_1, solution_2)
        fix_unselection = np.setdiff1d(all_nodes, union)
        if len(fix_unselection) == 0:
            # If all nodes are in the union, just pick some nodes not in solution_1
            fix_unselection = np.setdiff1d(all_nodes, solution_1)[
                : min(5, instance.n_nodes - len(solution_1))
            ]
        params["fix_unselection"] = fix_unselection

    elif param_type == "local_branching":
        # Test with local branching
        k_factor = 1.5
        n_diff = len(np.setdiff1d(solution_1, solution_2)) + len(
            np.setdiff1d(solution_2, solution_1)
        )
        k = int(n_diff * k_factor)
        local_branching = LocalBranching(k=k, sol_1=solution_1, sol_2=solution_2)
        params["local_branching"] = local_branching

    elif param_type == "desired_cost":
        # Test with desired cost
        # Set desired cost to length of solution_1 as a reasonable target
        desired_cost = len(solution_1)
        params["desired_cost"] = desired_cost

    # Patch time_limit and solver_params
    params["time_limit"] = 1
    params["solver_params"] = {"OutputFlag": 0}

    # # Call the maximum_weighted_independent_set function with patched gurobi_optimods
    # with patch("gurobi_optimods.utils.optimod", lambda: lambda f: f):
    result = maximum_weighted_independent_set(adj_matrix, weights, **params)

    # Check that result is not None and has expected properties
    assert result is not None, f"MWIS solver failed with {param_type} parameter"
    assert hasattr(result, "x"), (
        "Result should have an 'x' attribute with the solution nodes"
    )
    assert hasattr(result, "f"), (
        "Result should have an 'f' attribute with the objective value"
    )

    # Additional assertions based on parameter type
    if param_type == "fix_selection" and "fix_selection" in params:
        # All fixed nodes should be in the solution
        for node in params["fix_selection"]:
            assert node in result.x, f"Fixed node {node} not in solution"

    if param_type == "fix_unselection" and "fix_unselection" in params:
        # No fixed unselected nodes should be in the solution
        for node in params["fix_unselection"]:
            assert node not in result.x, f"Fixed unselected node {node} in solution"

    if param_type == "desired_cost" and "desired_cost" in params:
        # The solution size should match the desired cost
        assert abs(result.f - params["desired_cost"]) < 1e-6, (
            f"Solution cost {result.f} doesn't match desired cost {params['desired_cost']}"
        )

    if param_type == "local_branching" and "local_branching" in params:
        # hamming(x, solution_1) + hamming(x, solution_2) <= k
        hamming_dist_1 = hamming_distance(result.x, solution_1, instance.n_nodes)
        hamming_dist_2 = hamming_distance(result.x, solution_2, instance.n_nodes)
        assert hamming_dist_1 + hamming_dist_2 <= params["local_branching"].k

    # Result should be an independent set
    assert is_independent_set(instance, result.x)


def is_independent_set(instance: MISInstance, solution: np.ndarray) -> bool:
    """Check if a solution forms an independent set in the graph."""
    # Create a binary array where 1 indicates the node is in the solution
    binary_solution = np.zeros(instance.n_nodes)
    binary_solution[solution] = 1

    # Get the adjacency matrix
    adj_matrix = instance.adj_matrix_np

    # For each edge (i,j), check that not both i and j are in the solution
    for i in range(instance.n_nodes):
        if binary_solution[i] == 1:
            # Get all neighbors of i
            neighbors = adj_matrix[i].nonzero()[1]
            # Check that the sum of neighbors in the solution is greater than 1
            if np.sum(binary_solution[neighbors]) > 1:
                return False
    return True


def hamming_distance(
    solution_1: np.ndarray, solution_2: np.ndarray, n_nodes: int
) -> int:
    """Calculate the Hamming distance between two solutions."""
    # Convert to binary arrays
    binary_1 = np.zeros(n_nodes)
    binary_1[solution_1] = 1

    binary_2 = np.zeros(n_nodes)
    binary_2[solution_2] = 1

    # Hamming distance is the number of positions where the arrays differ
    return np.sum(binary_1 != binary_2)
