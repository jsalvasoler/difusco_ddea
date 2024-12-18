import torch


from config.config import Config
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
    assert heatmaps.shape[1] == instance.n
    
    solutions = None
    for i in range(heatmaps.shape[0]):
        solution = instance.get_feasible_from_individual(heatmaps[i])
        if solutions is None:
            solutions = solution
        else:
            solutions = torch.cat((solutions, solution), dim=0)

    assert solutions.shape[0] == config.pop_size
    assert solutions.shape[1] == instance.n

    # Calculate costs and gaps
    costs = solutions.float().sum(dim=1)
    instance_results = {
        "best_cost": costs.max(),
        "avg_cost": costs.mean(),
        "best_gap": (instance.get_gt_cost() - costs.max()) / instance.get_gt_cost(),
        "avg_gap": (instance.get_gt_cost() - costs.mean()) / instance.get_gt_cost(),
    }

    # Calculate selection frequencies for each node
    frequencies = solutions.float().mean(dim=0)  # Shape: (n_vertices,)
    node_entropy = -frequencies * torch.log(frequencies) - (1 - frequencies) * torch.log(1 - frequencies)  # Shape: (n_vertices,)
    instance_results["total_entropy"] = node_entropy.mean()

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
    hamming_dist = (heatmaps != solutions).float().sum(dim=1)
    instance_results["avg_diff_to_solution"] = hamming_dist.mean()

    # - some measure of how infeasible they are -> heatmaps. avg hamming dist from rounded heatmap to solution
    hamming_dist_rounded = (heatmaps.round() != solutions).float().sum(dim=1)
    instance_results["avg_diff_rounded_to_solution"] = hamming_dist_rounded.mean()

    return instance_results



