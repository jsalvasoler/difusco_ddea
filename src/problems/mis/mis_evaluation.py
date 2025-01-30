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
