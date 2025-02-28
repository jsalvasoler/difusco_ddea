import torch
from problems.mis.solve_optimal_recombination import solve_wmis

from tests.mis.test_mis_ea import read_mis_instance


def test_gurobi_optimal_solve() -> None:
    instance, _ = read_mis_instance()

    ind = torch.rand(2, instance.n_nodes)
    sols = instance.get_feasible_from_individual_batch(ind)
    solution_1, solution_2 = sols[0].numpy().nonzero()[0], sols[1].numpy().nonzero()[0]

    mwis_result = solve_wmis(instance, solution_1, solution_2, time_limit=5)
    assert mwis_result["children_np_labels"].shape == (instance.n_nodes,)

    # since we pass the best solution as starting point, the objective should be at least as good as the best solution
    assert mwis_result["children_obj"] >= max(len(solution_1), len(solution_2))
