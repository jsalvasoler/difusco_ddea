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


def get_lil_csr_matrix(adj_matrix: sp.csr_matrix) -> sp.csr_matrix:
    adj_matrix_lil = lil_matrix(adj_matrix)
    adj_matrix_lil[np.tril_indices(adj_matrix.shape[0])] = 0
    return adj_matrix_lil.tocsr()


def get_starting_solution(instance: MISInstance, solution_1: np.array, solution_2: np.array) -> np.array:
    starting_indices = solution_1 if len(solution_1) >= len(solution_2) else solution_2
    starting_solution = np.zeros(instance.n_nodes)
    starting_solution[starting_indices] = 1
    return starting_solution


@dataclass
class OptimalRecombResults:
    children_np_labels: np.array
    children: np.array
    runtime: float
    parent_1_obj: int
    parent_2_obj: int
    children_obj: int

    def __getitem__(self, key: str) -> np.array | float | int:
        return getattr(self, key)


def process_mwis_result(
    mwis: MWISResult, instance: MISInstance, solution_1: np.array, solution_2: np.array, start_time: float
) -> OptimalRecombResults:
    print(f"parent obj: {len(solution_1)}, {len(solution_2)}")
    print(f"children obj: {len(mwis.x)}")
    np_labels = np.zeros(instance.n_nodes)
    np_labels[mwis.x] = 1

    return OptimalRecombResults(
        children_np_labels=np_labels,
        children=np.array(mwis.x),
        runtime=round(time.time() - start_time, 4),
        parent_1_obj=len(solution_1),
        parent_2_obj=len(solution_2),
        children_obj=len(mwis.x),
    )


def solve_wmis(
    instance: MISInstance,
    solution_1: np.array,
    solution_2: np.array,
    time_limit: int = 60,
    **kwargs,
) -> OptimalRecombResults:
    """
    Solve the following problem:
    max weighted MIS rewarding nodes in solution_1 and solution_2 according to lambda_penalty
    st. is an IS
    """
    assert np.all(solution_1 < instance.n_nodes), "solution_1 contains invalid indices"
    assert np.all(solution_2 < instance.n_nodes), "solution_2 contains invalid indices"

    lambda_penalty = 0.05 if kwargs.get("lambda_penalty") is None else kwargs.get("lambda_penalty")
    weights = np.full(instance.n_nodes, 1 - lambda_penalty)
    only_1 = np.setdiff1d(solution_1, solution_2)
    weights[only_1] = 1
    only_2 = np.setdiff1d(solution_2, solution_1)
    weights[only_2] = 1
    both = np.intersect1d(solution_1, solution_2)
    weights[both] = 1 + lambda_penalty

    adj_matrix = get_lil_csr_matrix(instance.adj_matrix_np)
    starting_solution = get_starting_solution(instance, solution_1, solution_2)

    start_time = time.time()
    mwis = maximum_weighted_independent_set(
        adj_matrix,
        weights,
        starting_solution=starting_solution,
        time_limit=time_limit,
        solver_params={
            "OutputFlag": kwargs.get("output_flag", 0),
            "DisplayInterval": kwargs.get("display_interval", 50),
        },
    )

    return process_mwis_result(mwis, instance, solution_1, solution_2, start_time)


def solve_constrained_mis(
    instance: MISInstance,
    solution_1: np.array,
    solution_2: np.array,
    time_limit: int = 60,
    **kwargs,
) -> OptimalRecombResults:
    """
    Solve the following problem:
    max MIS
    st. is an IS
    st. if fix_selection is provided, then fix_selection nodes are selected (hard constraint)
    st. if fix_unselection is provided, then fix_unselection nodes are not selected (hard constraint)
    """
    assert np.all(solution_1 < instance.n_nodes), "solution_1 contains invalid indices"
    assert np.all(solution_2 < instance.n_nodes), "solution_2 contains invalid indices"

    adj_matrix = get_lil_csr_matrix(instance.adj_matrix_np)
    starting_solution = get_starting_solution(instance, solution_1, solution_2)
    weights = np.ones(instance.n_nodes)

    fix_selection = np.intersect1d(solution_1, solution_2) if kwargs.get("fix_selection") else None
    fix_unselection = (
        np.setdiff1d(np.arange(instance.n_nodes), np.union1d(solution_1, solution_2))
        if kwargs.get("fix_unselection")
        else None
    )
    print(f"fix_selection: {len(fix_selection) if fix_selection is not None else 'None'}")
    print(f"fix_unselection: {len(fix_unselection) if fix_unselection is not None else 'None'}")

    start_time = time.time()
    mwis = maximum_weighted_independent_set(
        adj_matrix,
        weights,
        starting_solution=starting_solution,
        fix_selection=fix_selection,
        fix_unselection=fix_unselection,
        time_limit=time_limit,
        solver_params={
            "OutputFlag": kwargs.get("output_flag", 0),
            "DisplayInterval": kwargs.get("display_interval", 50),
        },
    )

    return process_mwis_result(mwis, instance, solution_1, solution_2, start_time)


def solve_local_branching_mis(
    instance: MISInstance,
    solution_1: np.array,
    solution_2: np.array,
    time_limit: int = 60,
    **kwargs,
) -> OptimalRecombResults:
    """
    Solve the following problem:
    max MIS
    st. is an IS
    st. if h is hamming dist, h(solution_1, x) + h(solution_2, x) <= k

    We want solution_1 and solution_2 to be feasible, therefore, we need h(solution_1, solution_2) <= k,
    and k probably h(solution_1, solution_2) * 1.5.
    """
    assert np.all(solution_1 < instance.n_nodes), "solution_1 contains invalid indices"
    assert np.all(solution_2 < instance.n_nodes), "solution_2 contains invalid indices"

    adj_matrix = get_lil_csr_matrix(instance.adj_matrix_np)
    starting_solution = get_starting_solution(instance, solution_1, solution_2)
    weights = np.ones(instance.n_nodes)

    k_factor = kwargs.get("k_factor", 1.5)
    assert k_factor > 1, "k_factor must be greater than 1, otherwise problem is infeasible"
    # use local branching k equal to k_factor * hamming_distance(solution_1, solution_2)
    n_diff = len(np.setdiff1d(solution_1, solution_2)) + len(np.setdiff1d(solution_2, solution_1))
    k = n_diff * k_factor

    local_branching = LocalBranching(k=k, sol_1=solution_1, sol_2=solution_2)

    start_time = time.time()
    mwis = maximum_weighted_independent_set(
        adj_matrix,
        weights,
        starting_solution=starting_solution,
        local_branching=local_branching,
        time_limit=time_limit,
        solver_params={
            "OutputFlag": kwargs.get("output_flag", 0),
            "DisplayInterval": kwargs.get("display_interval", 50),
        },
    )

    return process_mwis_result(mwis, instance, solution_1, solution_2, start_time)


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


@dataclass
class LocalBranching:
    k: int
    sol_1: np.ndarray
    sol_2: np.ndarray


@optimod()
def maximum_weighted_independent_set(
    adjacency_matrix,
    weights,
    create_env,
    starting_solution: np.ndarray | None = None,
    fix_selection: np.ndarray | None = None,
    fix_unselection: np.ndarray | None = None,
    local_branching: LocalBranching | None = None,
):
    # validate that we do not use many things at once
    error_msg = "Cannot use fix_selection and local_branching at the same time"
    if local_branching:
        assert fix_selection is None, error_msg
        assert fix_unselection is None, error_msg
    if fix_selection is not None or fix_unselection is not None:
        assert local_branching is None, error_msg

    with create_env() as env, gp.Model("mwis", env=env) as model:
        rows, cols = adjacency_matrix.tocoo().row, adjacency_matrix.tocoo().col
        num_vertices, num_edges = len(weights), len(rows)
        # x_i: 1 if vertex i is in the independent set and 0 otherwise
        x = model.addMVar(num_vertices, vtype=GRB.BINARY, name="x")
        if starting_solution is not None:
            x.Start = starting_solution
        if fix_selection is not None:
            model.addConstr(x[fix_selection] == 1)
        if fix_unselection is not None:
            model.addConstr(x[fix_unselection] == 0)
        if local_branching is not None:
            # Get indices not in solutions (complement sets)
            sol_1_unselected = np.setdiff1d(np.arange(num_vertices), local_branching.sol_1)
            sol_2_unselected = np.setdiff1d(np.arange(num_vertices), local_branching.sol_2)
            hamming_expr = (
                gp.quicksum(1 - x[i] for i in local_branching.sol_1)
                + gp.quicksum(x[i] for i in sol_1_unselected)
                + gp.quicksum(1 - x[i] for i in local_branching.sol_2)
                + gp.quicksum(x[i] for i in sol_2_unselected)
            )

            model.addConstr(hamming_expr <= local_branching.k, name="local_branching")

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
