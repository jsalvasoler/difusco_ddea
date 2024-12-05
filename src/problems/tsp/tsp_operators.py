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
    (batch_size // 2, n, 4). Note that there might be duplicates in the edge lists.

    Args:
        parent1: Tensor of size (batch_size // 2, n + 1), first parent tours.
        parent2: Tensor of size (batch_size // 2, n + 1), second parent tours.

    Returns:
        edge_lists: Tensor of size (batch_size // 2, n, 4), int
    """
    n = parent1.size(1) - 1

    # Remove last element in the tour for both parents
    parent1 = parent1[:, :-1]
    parent2 = parent2[:, :-1]

    # Create indices tensors for all positions at once
    indices = torch.arange(n)

    # Handle prev indices with wrap-around
    prev_indices = torch.where(indices == 0, n - 1, indices - 1)
    next_indices = torch.where(indices == n - 1, 0, indices + 1)

    # Get all prev and next nodes at once
    prev_1 = parent1[:, prev_indices]  # shape: (batch_size, n)
    next_1 = parent1[:, next_indices]  # shape: (batch_size, n)
    prev_2 = parent2[:, prev_indices]  # shape: (batch_size, n)
    next_2 = parent2[:, next_indices]  # shape: (batch_size, n)

    # Stack all edges together
    edge_lists = torch.stack([prev_1, next_1, prev_2, next_2], dim=-1)  # shape: (batch_size // 2, n, 4)

    # Sort in the dimension of the last axis
    return edge_lists.sort(dim=-1).values


def select_from_edge_lists(edge_lists: torch.Tensor, visited: torch.Tensor) -> torch.Tensor:
    """
    edge lists are of shape (batch_size, n, 4), int
    visited is of shape (batch_size, n), boolean
    """
    batch_size = edge_lists.size(0) * 2

    edge_lists_copy = edge_lists.clone()

    # Mask visited nodes with -1
    for col_idx in range(4):
        edge_lists_copy[:, :, col_idx] = torch.where(
            visited.gather(1, edge_lists[:, :, col_idx]), -1, edge_lists[:, :, col_idx]
        )

    # Sort edge lists to bring -1s to the front
    edge_lists_copy = edge_lists_copy.sort(dim=-1).values

    # Count unique elements by summing differences and adding 1 (for the first unique element)
    diffs = edge_lists_copy[:, :, 1:] != edge_lists_copy[:, :, :-1]
    unique_counts = diffs.sum(dim=-1) + 1

    # Discount one if -1 is present in the row
    unique_counts = unique_counts - (torch.sum(edge_lists_copy == -1, dim=-1) > 0).int()

    # Select argument with minimum unique count, randomly break ties
    min_unique_count = unique_counts[~visited].reshape(batch_size // 2, -1).min(dim=-1, keepdim=True).values
    candidates_mask = (unique_counts == min_unique_count) & (~visited)
    candidates_mask = candidates_mask.float().div(candidates_mask.sum(dim=-1, keepdim=True))

    # Generate random indices to break ties
    return torch.multinomial(candidates_mask.float(), num_samples=1).squeeze()


def edge_recombination_crossover(parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
    """
    Perform edge recombination crossover (ERC) for a batch of parents in a vectorized manner.

    Note: we assume that all edges are valid, i.e., a dense graph is provided

    Args:
        parent1: Tensor of size (batch_size // 2, n + 1), first parent tours.
        parent2: Tensor of size (batch_size // 2, n + 1), second parent tours.

    Returns:
        offspring: Tensor of size (batch_size, n + 1), containing offspring tours.
    """
    batch_size = parent1.size(0) * 2
    n = parent1.size(1) - 1  # Number of cities excluding the return to start

    # Initialize tensors
    offspring = torch.zeros((batch_size // 2, n + 1), dtype=torch.long)
    visited = torch.zeros((batch_size // 2, n), dtype=torch.bool)

    # Build edge lists
    edge_lists = build_edge_lists(parent1, parent2)
    assert edge_lists.shape == (batch_size // 2, n, 4)

    current_nodes = torch.randint(0, n, (batch_size // 2,), dtype=torch.int)
    visited[torch.arange(batch_size // 2), current_nodes] = True
    offspring[:, 0] = current_nodes

    # Generate tours
    for step in range(1, n):
        current_nodes = select_from_edge_lists(edge_lists, visited)

        # Update visitation and current nodes
        visited[torch.arange(batch_size // 2), current_nodes] = True
        offspring[:, step] = current_nodes

    # Complete tours by returning to the start node
    offspring[:, -1] = offspring[:, 0]
    return offspring
