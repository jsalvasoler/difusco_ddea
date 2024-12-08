from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def batched_two_opt_torch(
    points: np.ndarray | torch.Tensor,
    tour: np.ndarray | torch.Tensor,
    max_iterations: int = 1000,
    device: Literal["cpu", "gpu"] = "cpu",
) -> tuple[np.ndarray | torch.Tensor, int]:
    """
    Apply the 2-opt algorithm to a batch of tours.
    Tours have N + 1 elements, i.e., the first city is repeated at the end.

    Works for both numpy and torch.

    Args:
        points: Points as numpy array or torch tensor of shape (N, 2)
        tour: Tour as numpy array or torch tensor of shape (batch_size, N+1)
        max_iterations: Maximum number of iterations
        device: Device to run computations on ("cpu" or "gpu")

    Returns:
        tuple of (optimized tour array, number of iterations performed)
    """
    iterator = 0
    return_numpy = isinstance(tour, np.ndarray)

    with torch.inference_mode():
        # Convert to torch tensors if needed
        cuda_points = points if isinstance(points, torch.Tensor) else torch.from_numpy(points).to(device)
        cuda_tour = tour if isinstance(tour, torch.Tensor) else torch.from_numpy(tour.copy()).to(device)

        # Rest of the function remains the same
        batch_size = cuda_tour.shape[0]

        min_change = -1.0
        while min_change < 0.0:
            points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
            points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
            A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
            A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
            A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            min_change = torch.min(valid_change)
            flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
            min_i = torch.div(flatten_argmin_index, len(points), rounding_mode="floor")
            min_j = torch.remainder(flatten_argmin_index, len(points))

            if min_change < -1e-6:
                for i in range(batch_size):
                    cuda_tour[i, min_i[i] + 1 : min_j[i] + 1] = torch.flip(
                        cuda_tour[i, min_i[i] + 1 : min_j[i] + 1], dims=(0,)
                    )
                iterator += 1
            else:
                break

            if iterator >= max_iterations:
                break

        # Convert back to numpy if input was numpy
        tour = cuda_tour.cpu().numpy() if return_numpy else cuda_tour

    return tour, iterator


def build_edge_lists(parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
    """
    Build edge lists for a batch of parents. Edge lists are defined as the set of nodes
    that are connected to a given node in the tours of either parent. The max number
    of edges is 4 per node, which is why we need to return a tensor of shape
    (batch_size, n, 4). Note that there might be duplicates in the edge lists.

    Args:
        parent1: Tensor of size (batch_size, n + 1), first parent tours.
        parent2: Tensor of size (batch_size, n + 1), second parent tours.

    Returns:
        edge_lists: Tensor of size (batch_size, n, 4), int
    """
    batch_size, n_plus_1 = parent1.size()
    n = n_plus_1 - 1

    # Remove last element in the tour for both parents
    parent1 = parent1[:, :-1]  # shape: (batch_size, n)
    parent2 = parent2[:, :-1]  # shape: (batch_size, n)

    # For each node, we need its neighbors in both parent tours
    # Roll the tours to get previous and next nodes
    prev_1 = torch.roll(parent1, shifts=1, dims=1)  # shape: (batch_size, n)
    next_1 = torch.roll(parent1, shifts=-1, dims=1)  # shape: (batch_size, n)
    prev_2 = torch.roll(parent2, shifts=1, dims=1)  # shape: (batch_size, n)
    next_2 = torch.roll(parent2, shifts=-1, dims=1)  # shape: (batch_size, n)

    # Create masks for all nodes at once (batch_size, n, n)
    node_indices = torch.arange(n, device=parent1.device)
    mask1 = parent1.unsqueeze(-1) == node_indices
    mask2 = parent2.unsqueeze(-1) == node_indices

    # Create edge lists (batch_size, n, 4)
    edge_lists = torch.zeros((batch_size, n, 4), dtype=torch.long, device=parent1.device)

    # Gather neighbors for all nodes at once
    edge_lists[:, :, 0] = (mask1 * prev_1.unsqueeze(-1)).sum(dim=1)
    edge_lists[:, :, 1] = (mask1 * next_1.unsqueeze(-1)).sum(dim=1)
    edge_lists[:, :, 2] = (mask2 * prev_2.unsqueeze(-1)).sum(dim=1)
    edge_lists[:, :, 3] = (mask2 * next_2.unsqueeze(-1)).sum(dim=1)

    # Sort in the dimension of the last axis
    return edge_lists.sort(dim=-1).values


def select_from_edge_lists(
    edge_lists: torch.Tensor,
    visited: torch.Tensor,
    current_node: torch.Tensor,
) -> torch.Tensor:
    """
    Select a node from the edge lists based on the number of unique elements.

    Args:
        edge_lists: Tensor of size (batch_size, n, 4), int
        visited: Tensor of size (batch_size, n), boolean

    Returns:
        selection: Tensor of size (batch_size,), int
    """
    batch_size = edge_lists.size(0)

    edge_lists_copy = edge_lists.clone()

    # candidates is a tensor of size (batch_size, 4)
    candidates = edge_lists_copy[torch.arange(batch_size), current_node, :]

    # edge_lists_candidates is a tensor of size (batch_size, 4, 4)
    edge_lists_candidates = edge_lists_copy[torch.arange(batch_size).unsqueeze(-1).expand(-1, 4), candidates, :]

    # Mask visited nodes with -1
    edge_lists_candidates[:, :, 0] = torch.where(
        visited.gather(1, edge_lists_candidates[:, :, 0]), -1, edge_lists_candidates[:, :, 0]
    )
    edge_lists_candidates[:, :, 1] = torch.where(
        visited.gather(1, edge_lists_candidates[:, :, 1]), -1, edge_lists_candidates[:, :, 1]
    )
    edge_lists_candidates[:, :, 2] = torch.where(
        visited.gather(1, edge_lists_candidates[:, :, 2]), -1, edge_lists_candidates[:, :, 2]
    )
    edge_lists_candidates[:, :, 3] = torch.where(
        visited.gather(1, edge_lists_candidates[:, :, 3]), -1, edge_lists_candidates[:, :, 3]
    )

    # Count unique elements by summing differences and adding 1 (for the first unique element)
    edge_lists_candidates = edge_lists_candidates.sort(dim=-1).values
    diffs = edge_lists_candidates[:, :, 1:] != edge_lists_candidates[:, :, :-1]
    unique_counts = diffs.sum(dim=-1) + 1

    # Discount one if -1 is present in the row
    unique_counts = unique_counts - (torch.sum(edge_lists_candidates == -1, dim=-1) > 0).int()

    # binary mask of visited candidates
    visited_candidates = visited.gather(1, candidates)  # shape: (batch_size, 4)

    # we set inf = 10 for the counts of the visited candidates, feasible max is 4
    min_unique_count = (
        torch.where(visited_candidates, 10, unique_counts).min(dim=1, keepdim=True).values
    )  # shape: (batch_size, 1)
    real_candidates_mask = (unique_counts == min_unique_count) & (~visited_candidates)

    sums = real_candidates_mask.sum(dim=-1, keepdim=True)
    to_draw_randomly = (sums == 0).bool()  # shape: (batch_size, 1)
    sums = torch.where(to_draw_randomly, 1.0, sums)
    real_candidates_mask = torch.where(to_draw_randomly, torch.ones_like(real_candidates_mask), real_candidates_mask)
    real_candidates_mask = real_candidates_mask.float().div(sums)

    idx = torch.multinomial(real_candidates_mask.float(), num_samples=1)  # shape: (batch_size, 1)
    selected_nodes = candidates.gather(1, idx)

    # select the first unvisited node if all candidates are visited
    first_unvisited = (~visited).int().argmax(dim=1)
    selected_nodes = torch.where(to_draw_randomly, first_unvisited.unsqueeze(-1), selected_nodes)

    return selected_nodes.squeeze(1)


def edge_recombination_crossover(parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
    """
    Perform edge recombination crossover (ERC) for a batch of parents in a vectorized manner.

    Note: we assume that all edges are valid, i.e., a dense graph is provided

    Args:
        parent1: Tensor of size (batch_size, n + 1), first parent tours.
        parent2: Tensor of size (batch_size, n + 1), second parent tours.

    Returns:
        offspring: Tensor of size (batch_size, n + 1), containing offspring tours.
    """
    batch_size = parent1.size(0)
    n = parent1.size(1) - 1  # Number of cities excluding the return to start
    device = parent1.device

    # Initialize tensors
    offspring = torch.zeros((batch_size, n + 1), dtype=torch.long, device=device)
    visited = torch.zeros((batch_size, n), dtype=torch.bool, device=device)

    # Build edge lists
    edge_lists = build_edge_lists(parent1, parent2)
    assert edge_lists.shape == (batch_size, n, 4)

    current_nodes = torch.zeros((batch_size,), dtype=torch.int, device=device)
    visited[torch.arange(batch_size), current_nodes] = True
    offspring[:, 0] = current_nodes

    # Generate tours
    for step in range(1, n):
        current_nodes = select_from_edge_lists(edge_lists, visited, current_nodes)

        # Update visitation and current nodes
        visited[torch.arange(batch_size), current_nodes] = True
        offspring[torch.arange(batch_size), step] = current_nodes

    # Complete tours by returning to the start node
    offspring[:, -1] = offspring[:, 0]
    return offspring
