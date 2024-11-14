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
        adj_matrix: The adjacency matrix of the graph in a torch.sparse_csr_tensor
    """

    def get_neighbors(adj_csr: torch.sparse.FloatTensor, node: int) -> torch.Tensor:
        row_start = adj_csr.crow_indices()[node].item()
        row_end = adj_csr.crow_indices()[node + 1].item()
        return adj_csr.col_indices()[row_start:row_end]

    # Initialize solution tensor on the same device as predictions.
    solution = torch.zeros_like(predictions, dtype=torch.int, device=predictions.device).clone()

    # Get sorted indices of predictions in descending order.
    sorted_predict_labels = torch.argsort(-predictions)

    # Process each node according to the sorted prediction order.
    for node in sorted_predict_labels:
        if solution[node] == -1:
            continue

        neighbors = get_neighbors(adj_csr=adj_matrix, node=node)
        solution[neighbors] = -1
        solution[node] = 1

    return (solution == 1).int()
