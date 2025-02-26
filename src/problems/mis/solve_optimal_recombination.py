# ruff: noqa: ANN201, ANN001
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gurobipy as gp
import numpy as np
import scipy.sparse as sp

# from gurobi_optimods.mwis import maximum_weighted_independent_set
from gurobi_optimods.utils import optimod
from gurobipy import GRB
from scipy.sparse import lil_matrix

if TYPE_CHECKING:
    from problems.mis.mis_instance import MISInstance


def solve_problem(
    instance: MISInstance,
    solution_1: np.array,
    solution_2: np.array,
    time_limit: int = 60,
) -> dict:
    assert np.all(solution_1 < instance.n_nodes), "solution_1 contains invalid indices"
    assert np.all(solution_2 < instance.n_nodes), "solution_2 contains invalid indices"

    lambda_penalty = 0.05
    weights = np.full(instance.n_nodes, 1 - lambda_penalty)
    only_1 = np.setdiff1d(solution_1, solution_2)
    weights[only_1] = 1
    only_2 = np.setdiff1d(solution_2, solution_1)
    weights[only_2] = 1
    both = np.intersect1d(solution_1, solution_2)
    weights[both] = 1 + lambda_penalty

    # Convert to lil_matrix for efficient modification
    adj_matrix_lil = lil_matrix(instance.adj_matrix_np)

    # Make instance.adj_matrix an upper triangular matrix
    adj_matrix_lil[np.tril_indices(instance.n_nodes)] = 0

    # Convert back to csr_matrix
    adj_matrix = adj_matrix_lil.tocsr()

    # get starting solution (best among the two solutions)
    starting_indices = solution_1 if len(solution_1) >= len(solution_2) else solution_2
    starting_solution = np.zeros(instance.n_nodes)
    starting_solution[starting_indices] = 1

    start_time = time.time()
    mwis = maximum_weighted_independent_set(
        adj_matrix,
        weights,
        starting_solution=starting_solution,
        time_limit=time_limit,
        solver_params={
            "DisplayInterval": 50,
        },
    )
    print(mwis.x.tolist())
    print(f"parent obj: {len(solution_1)}, {len(solution_2)}")
    print(f"children obj: {len(mwis.x)}")
    np_labels = np.zeros(instance.n_nodes)
    np_labels[mwis.x] = 1

    return {
        "runtime": round(time.time() - start_time, 4),
        "parent_1_obj": len(solution_1),
        "parent_2_obj": len(solution_2),
        "children_obj": len(mwis.x),
        "children_np_labels": np_labels,
    }


@dataclass
class MWISResult:
    """
    Data class representing the maximum weighted independent set
    (clique) and its weight

    Attributes
    ----------
    x : ndarray
        The maximum weighted independent set (clique) array
    f : float
        The total weight of the maximum weighted independent set (clique)
    """

    x: np.ndarray
    f: float


@optimod()
def maximum_weighted_independent_set(
    adjacency_matrix,
    weights,
    create_env,
    starting_solution: np.ndarray | None = None,
):
    """This implementation uses the gurobipy matrix friendly APIs which are well
    suited for the input data in scipy data structures."""
    with create_env() as env, gp.Model("mwis", env=env) as model:
        rows, cols = adjacency_matrix.tocoo().row, adjacency_matrix.tocoo().col
        num_vertices, num_edges = len(weights), len(rows)
        # x_i: 1 if vertex i is in the independent set and 0 otherwise
        x = model.addMVar(num_vertices, vtype=GRB.BINARY, name="x")
        if starting_solution is not None:
            x.Start = starting_solution
        # Maximize the sum of the vertex weights in the independent set
        model.setObjective(weights @ x, sense=GRB.MAXIMIZE)
        # Get the incident matrix from the adjacency matrix where
        # there is a column for each edge
        indices = []
        for i, j in zip(rows, cols):
            indices.extend([i, j])
        indptr = range(0, len(indices) + 2, 2)
        data = np.ones(2 * num_edges)
        A = sp.csc_array((data, indices, indptr), shape=(num_vertices, num_edges))
        # The independent set contains non-adjacent vertices
        model.addMConstr(
            A.T,
            x,
            GRB.LESS_EQUAL,
            np.ones(A.shape[1]),
            name="no_adjacent_vertices",
        )
        model.optimize()
        (mwis,) = np.where(x.X >= 0.5)
        return MWISResult(mwis, sum(weights[mwis]))
