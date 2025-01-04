import timeit

import torch
from config.myconfig import Config
from problems.mis.mis_instance import MISInstance


def metrics_on_mis_heatmaps(heatmaps: torch.Tensor, instance: MISInstance, config: Config) -> dict:
    """Calculate metrics on MIS heatmaps including selection frequencies.

    Args:
        heatmaps: Tensor of shape (n_solutions, n_vertices) containing sampled solutions
        instance: MIS problem instance
        config: Configuration object

    Returns:
        Dictionary containing metrics including costs, gaps and selection frequencies
    """
    assert heatmaps.shape[0] == config.pop_size
    assert heatmaps.shape[1] == instance.n_nodes

    solutions = None
    start_time = timeit.default_timer()
    for i in range(heatmaps.shape[0]):
        solution = instance.get_feasible_from_individual(heatmaps[i]).unsqueeze(0)
        solutions = solution if solutions is None else torch.vstack((solutions, solution))
    end_time = timeit.default_timer()
    feasibility_heuristics_time = end_time - start_time

    assert solutions.shape[0] == config.pop_size
    assert solutions.shape[1] == instance.n_nodes

    # Calculate costs and gaps
    costs = solutions.float().sum(dim=1)

    instance_results = {
        "best_cost": costs.max(),
        "avg_cost": costs.mean(),
        "best_gap": (instance.get_gt_cost() - costs.max()) / instance.get_gt_cost(),
        "avg_gap": (instance.get_gt_cost() - costs.mean()) / instance.get_gt_cost(),
        "feasibility_heuristics_time": feasibility_heuristics_time,
    }

    # Calculate selection frequencies for each node
    frequencies = solutions.float().mean(dim=0)  # Shape: (n_vertices,)
    # Special case handling: entropy is 0 when frequency is 0 or 1
    entropies = torch.zeros_like(frequencies)
    valid = (frequencies > 0) & (frequencies < 1)  # Mask for valid frequencies
    f_valid = frequencies[valid]
    entropies[valid] = -f_valid * torch.log(f_valid) - (1 - f_valid) * torch.log(1 - f_valid)
    instance_results["total_entropy_solutions"] = entropies.mean()

    # Caluclate entropy for directly on the heatmaps
    prob_nodes = heatmaps.float().mean(dim=0)
    entropies_heatmaps = torch.zeros_like(prob_nodes)
    valid = (prob_nodes > 0) & (prob_nodes < 1)  # Mask for valid frequencies
    f_valid = prob_nodes[valid]
    entropies_heatmaps[valid] = -f_valid * torch.log(f_valid) - (1 - f_valid) * torch.log(1 - f_valid)
    instance_results["total_entropy_heatmaps"] = entropies_heatmaps.mean()

    # Calculate the number of unique solutions
    unique_solutions = torch.unique(solutions, dim=0)
    instance_results["unique_solutions"] = unique_solutions.shape[0]

    # Calculate the number of solutions that are not the best solution
    non_best_solutions = solutions[costs != costs.max()]
    instance_results["non_best_solutions"] = non_best_solutions.shape[0]

    # - some measure of how integer they are → heatmaps. avg distance to nearest int per value
    heatmaps_diff = heatmaps.round() - heatmaps
    instance_results["avg_diff_to_nearest_int"] = heatmaps_diff.abs().mean(dim=0).mean()

    # - some measure of how infeasible they are -> heatmaps. avg hamming dist from heatmap to solution
    hamming_dist = (heatmaps - solutions.float()).abs().sum(dim=1)
    instance_results["avg_diff_to_solution"] = hamming_dist.mean()

    # - some measure of how infeasible they are -> heatmaps. avg hamming dist from rounded heatmap to solution
    hamming_dist_rounded = (heatmaps.round() != solutions).float().sum(dim=1)
    instance_results["avg_diff_rounded_to_solution"] = hamming_dist_rounded.mean()

    # compute average hamming distance between pairs of solutions
    hamming_dist_pairs = (solutions[:, None] != solutions).float().sum(dim=2)
    instance_results["avg_hamming_dist_pairs"] = hamming_dist_pairs.mean()

    # make sure that values are numerical and not tensors
    return {k: v.item() if torch.is_tensor(v) else v for k, v in instance_results.items()}
