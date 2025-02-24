import numpy as np
import torch
from scipy import sparse as sp


def mis_decode_np(predictions: np.ndarray, adj_matrix: sp.csr_matrix) -> np.ndarray:
    """Decode the labels to the MIS."""

    solution = np.zeros_like(predictions.astype(int))
    sorted_predict_labels = np.argsort(-predictions)

    for node in sorted_predict_labels:
        if solution[node] == -1:
            continue

        solution[adj_matrix[node].nonzero()[1]] = -1
        solution[node] = 1

    return (solution == 1).astype(int)


def mis_decode_torch(predictions: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Decode the labels to the MIS using PyTorch tensors.

    Args:
        predictions: The predicted labels in a torch.Tensor.
        adj_matrix: The adjacency matrix of the graph as a torch.sparse_coo_tensor.

    Returns:
        torch.Tensor: A binary tensor indicating the nodes included in the MIS.
    """

    # Initialize solution tensor on the same device as predictions.
    solution = torch.zeros_like(predictions, dtype=torch.int, device=predictions.device).clone()

    # Get sorted indices of predictions in descending order.
    sorted_predict_labels = torch.argsort(-predictions)

    # Extract adjacency matrix indices
    row_indices, col_indices = adj_matrix.indices()

    # Process each node according to the sorted prediction order
    for node in sorted_predict_labels:
        if solution[node] == -1:
            continue

        # Get neighbors of the current node
        neighbors_mask = row_indices == node  # Boolean mask for rows matching the node
        neighbors = col_indices[neighbors_mask]  # Column indices where the row matches

        # Mark neighbors as invalid (-1)
        solution[neighbors] = -1
        # Add the current node to the solution
        solution[node] = 1

    # Return a binary tensor indicating the nodes in the MIS
    return (solution == 1).bool()


def precompute_neighbors_padded(adj_csr: torch.Tensor) -> torch.Tensor:
    crow = adj_csr.crow_indices()
    col = adj_csr.col_indices()
    degrees = crow[1:] - crow[:-1]
    max_degree = degrees.max().item()
    num_nodes = adj_csr.shape[0]

    # Initialize padded tensor with -1
    neighbors_padded = torch.full((num_nodes, max_degree), -1, dtype=torch.long)

    for node in range(num_nodes):
        start, end = crow[node], crow[node + 1]
        neighbors = col[start:end]
        neighbors_padded[node, : len(neighbors)] = neighbors

    return neighbors_padded, degrees


@torch.jit.script
def mis_decode_torch_batched(
    predictions: torch.Tensor,
    neighbors_padded: torch.Tensor,
    degrees: torch.Tensor,
) -> torch.Tensor:
    """
    Decode the labels to the MIS using PyTorch tensors (batched version).

    Args:
        predictions: The predicted labels as a torch.Tensor of shape (B, n).
        neighbors_padded: The precomputed neighbors as a torch. Tensor of shape (n, max_degree).
        Padding is done with -1.
        degrees: The degrees of the nodes as a torch.Tensor of shape (n,).

    Returns:
        torch.Tensor: A binary tensor of shape (B, n) indicating the nodes included in the MIS.
    """

    B, n = predictions.shape

    # Initialize solution tensor. Shape: (B, n)
    solution = torch.zeros_like(predictions, dtype=torch.int)

    # Get sorted indices of predictions in descending order. Shape: (B, n)
    sorted_predict_labels = torch.argsort(-predictions, dim=1)

    # Process each node according to the sorted prediction order
    for node_idx in range(n):
        # Get the current nodes for each batch. Shape: (B,)
        current_nodes = sorted_predict_labels[:, node_idx]

        # Check which batch items still have valid nodes
        valid = solution[torch.arange(B), current_nodes] == 0

        if not valid.any():
            continue

        # If valid nodes, add them to the solution
        solution[valid, current_nodes[valid]] = 1

        # Vectorized neighbor invalidation ----------------------------
        # 1. Get valid batch indices and their corresponding nodes
        valid_batch = torch.where(valid)[0]  # Shape: [num_valid]
        current_valid_nodes = current_nodes[valid]  # Shape: [num_valid]

        # 2. Get neighbors for these nodes (with padding)
        neighbors = neighbors_padded[current_valid_nodes]  # Shape: [num_valid, max_degree]

        # 3. Create mask for valid neighbors (ignore -1 padding)
        mask = neighbors != -1

        # 4. Compute indices to invalidate
        # - Repeat batch indices for each valid neighbor
        batch_indices = torch.repeat_interleave(valid_batch, degrees[current_valid_nodes])
        # - Flatten and filter neighbor indices
        neighbor_indices = neighbors[mask]

        # 5. Bulk update (if there are neighbors to invalidate)
        if batch_indices.numel() > 0:
            solution[batch_indices, neighbor_indices] = -1

    # Return a binary tensor indicating the nodes in the MIS
    return (solution == 1).to(torch.bool)
