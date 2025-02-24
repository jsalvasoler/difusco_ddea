import time

import numpy as np
from gurobi_optimods.mwis import maximum_weighted_independent_set
from problems.mis.mis_instance import MISInstance
from scipy.sparse import lil_matrix


def solve_problem(
    instance: MISInstance,
    solution_1: np.array,
    solution_2: np.array,
) -> dict:
    assert np.all(solution_1 < instance.n_nodes), "solution_1 contains invalid indices"
    assert np.all(solution_2 < instance.n_nodes), "solution_2 contains invalid indices"

    weights = np.full(instance.n_nodes, 0.5)
    only_1 = np.setdiff1d(solution_1, solution_2)
    weights[only_1] = 1
    only_2 = np.setdiff1d(solution_2, solution_1)
    weights[only_2] = 1
    both = np.intersect1d(solution_1, solution_2)
    weights[both] = 1.5

    # Convert to lil_matrix for efficient modification
    adj_matrix_lil = lil_matrix(instance.adj_matrix_np)

    # Make instance.adj_matrix an upper triangular matrix
    adj_matrix_lil[np.tril_indices(instance.n_nodes)] = 0

    # Convert back to csr_matrix
    adj_matrix = adj_matrix_lil.tocsr()

    start_time = time.time()
    mwis = maximum_weighted_independent_set(
        adj_matrix,
        weights,
        time_limit=60,
        solver_params={
            "ThreadLimit": 8,
            "DisplayInterval": 20,
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
