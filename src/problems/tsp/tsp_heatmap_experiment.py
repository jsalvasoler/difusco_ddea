from __future__ import annotations

import timeit
from typing import TYPE_CHECKING

import scipy.sparse
import torch

if TYPE_CHECKING:
    from config.myconfig import Config
    from problems.tsp.tsp_instance import TSPInstance


def get_feasible_solutions(
    heatmaps: torch.Tensor, instance: TSPInstance
) -> torch.Tensor:
    solutions = None
    for i in range(heatmaps.shape[0]):
        solution = instance.get_tour_from_adjacency_np_heatmap(
            heatmaps[i].cpu().numpy()
        ).unsqueeze(0)
        solutions = (
            solution if solutions is None else torch.vstack((solutions, solution))
        )
    return solutions


def metrics_on_tsp_heatmaps(
    heatmaps: torch.Tensor, instance: TSPInstance, config: Config
) -> dict:
    """Calculate metrics on TSP heatmaps including edge selection frequencies.

    Args:
        heatmaps: Tensor of shape (n_solutions, n_vertices, n_vertices) containing sampled adjacency matrices
        instance: TSP problem instance
        config: Configuration object

    Returns:
        Dictionary containing metrics including costs, gaps and edge selection frequencies
    """
    assert heatmaps.shape[0] == config.pop_size, (
        f"Heatmaps shape: {heatmaps.shape}, config.pop_size: {config.pop_size}"
    )
    if config.sparse_factor <= 0:
        assert heatmaps.shape[1] == instance.n, (
            f"Heatmaps shape: {heatmaps.shape}x{instance.n}x{instance.n}"
        )
        assert heatmaps.shape[2] == instance.n, (
            f"Heatmaps shape: {heatmaps.shape}x{instance.n}x{instance.n}"
        )
    else:
        n_edges = instance.n * config.sparse_factor
        assert heatmaps.shape[1] == n_edges, (
            f"Heatmaps shape: {heatmaps.shape}x{n_edges}"
        )

    start_time = timeit.default_timer()
    solutions = get_feasible_solutions(heatmaps, instance)
    end_time = timeit.default_timer()
    feasibility_heuristics_time = end_time - start_time

    assert solutions.shape[0] == config.pop_size
    assert solutions.shape[1] == instance.n + 1  # +1 for closed tour

    # Calculate costs and gaps
    costs = torch.tensor([instance.evaluate_tsp_route(tour) for tour in solutions])
    instance_results = {
        "best_cost": costs.min(),  # TSP is a minimization problem
        "avg_cost": costs.mean(),
        "best_gap": (costs.min() - instance.get_gt_cost()) / instance.get_gt_cost(),
        "avg_gap": (costs.mean() - instance.get_gt_cost()) / instance.get_gt_cost(),
        "feasibility_heuristics_time": feasibility_heuristics_time,
    }

    if config.sparse_factor > 0:
        # reconstruct heatmap in adj. matrix form
        new_heatmaps = None
        for i in range(config.pop_size):
            sparse_heatmap = heatmaps[i].cpu().numpy()
            heatmap = (
                scipy.sparse.coo_matrix(
                    (
                        sparse_heatmap,
                        (instance.np_edge_index[0], instance.np_edge_index[1]),
                    ),
                ).toarray()
                + scipy.sparse.coo_matrix(
                    (
                        sparse_heatmap,
                        (instance.np_edge_index[1], instance.np_edge_index[0]),
                    ),
                ).toarray()
            )
            heatmap = torch.tensor(heatmap, device=heatmaps.device)
            assert heatmap.shape == (instance.n, instance.n)
            new_heatmaps = (
                heatmap.unsqueeze(0)
                if new_heatmaps is None
                else torch.vstack((new_heatmaps, heatmap.unsqueeze(0)))
            )

        heatmaps = new_heatmaps
        del new_heatmaps
        assert heatmaps.shape[0] == config.pop_size
        assert heatmaps.shape[1] == instance.n
        assert heatmaps.shape[2] == instance.n

    # Get indices for consecutive nodes in tours (including wrap-around)
    adj_matrices = torch.zeros_like(heatmaps)

    # Get indices for consecutive nodes in tours (including wrap-around)
    idx_from = solutions[:, :-1]  # All nodes except last
    idx_to = solutions[:, 1:]  # All nodes except first

    # Create batch indices for scatter operation
    batch_idx = (
        torch.arange(solutions.shape[0]).unsqueeze(1).expand(-1, solutions.shape[1] - 1)
    )

    # Set 1s for edges in the tours
    adj_matrices[batch_idx, idx_from, idx_to] = 1
    # Set 1s for reverse edges (since TSP graph is undirected)
    adj_matrices[batch_idx, idx_to, idx_from] = 1

    # Calculate edge selection frequencies on heatmaps
    frequencies = heatmaps.float().mean(dim=0)  # Shape: (n_vertices, n_vertices)
    # Calculate entropy for edge selections
    entropies = torch.zeros_like(frequencies)
    valid = (frequencies > 0) & (frequencies < 1)
    f_valid = frequencies[valid]
    entropies[valid] = -f_valid * torch.log(f_valid) - (1 - f_valid) * torch.log(
        1 - f_valid
    )
    instance_results["total_entropy_heatmaps"] = entropies.mean()

    # Calculate edge selection frequencies on heatmaps
    frequencies = adj_matrices.float().mean(dim=0)  # Shape: (n_vertices, n_vertices)
    # Calculate entropy for edge selections
    entropies = torch.zeros_like(frequencies)
    valid = (frequencies > 0) & (frequencies < 1)
    f_valid = frequencies[valid]
    entropies[valid] = -f_valid * torch.log(f_valid) - (1 - f_valid) * torch.log(
        1 - f_valid
    )
    instance_results["total_entropy_solutions"] = entropies.mean()

    # Calculate avg diff to solution
    adj_matrices_flat = adj_matrices.view(config.pop_size, -1)
    heatmaps_flat = heatmaps.view(config.pop_size, -1)
    diffs = (heatmaps_flat - adj_matrices_flat).abs().mean(-1)
    instance_results["avg_diff_to_solution"] = diffs.mean()

    # Calculate avg diff rounded to solution
    diffs = (heatmaps_flat.round() - adj_matrices_flat).abs().mean(-1)
    instance_results["avg_diff_rounded_to_solution"] = diffs.mean()

    # Calculate the number of unique solutions
    unique_solutions = torch.unique(solutions, dim=0)
    instance_results["unique_solutions"] = unique_solutions.shape[0]

    # Calculate the number of solutions that are not the best solution
    non_best_solutions = solutions[costs != costs.min()]
    instance_results["non_best_solutions"] = non_best_solutions.shape[0]

    # Average distance to nearest integer for heatmap values
    heatmaps_diff = heatmaps.round() - heatmaps
    instance_results["avg_diff_to_nearest_int"] = heatmaps_diff.abs().mean()

    # Average hamming distance between pairs of solutions (adj_matrices)
    diffs = (
        adj_matrices_flat.unsqueeze(0) != adj_matrices_flat.unsqueeze(1)
    ).float()  # Shape: (n_solutions, n_solutions, n_nodes * n_nodes)
    # Mean over all entries in the adjacency matrix (Hamming distance)
    pairwise_distances = diffs.mean(dim=2)

    # Extract the upper triangle of the distance matrix (excluding the diagonal)
    upper_triangle = pairwise_distances.triu(diagonal=1)  # Exclude diagonal
    n_pairs = config.pop_size * (config.pop_size - 1) / 2  # Total number of pairs

    # Compute the mean Hamming distance
    instance_results["avg_hamming_dist_pairs"] = upper_triangle.sum() / n_pairs

    # Make sure that values are numerical and not tensors
    return {
        k: v.item() if torch.is_tensor(v) else v for k, v in instance_results.items()
    }
