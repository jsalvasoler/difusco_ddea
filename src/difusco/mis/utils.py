import numpy as np
import torch
from scipy import sparse as sp


def mis_decode_np(predictions: np.ndarray, adj_matrix: sp.csr_matrix) -> np.ndarray:
    """Decode the labels to the MIS."""

    solution = np.zeros_like(predictions.astype(int))
    sorted_predict_labels = np.argsort(-predictions)

    for i in sorted_predict_labels:
        next_node = i

        if solution[next_node] == -1:
            continue

        solution[adj_matrix[next_node].nonzero()[1]] = -1
        solution[next_node] = 1

    return (solution == 1).astype(int)


def mis_decode_torch(predictions: torch.Tensor, adj_matrix: sp.csr_matrix) -> torch.Tensor:
    """Decode the labels to the MIS using PyTorch tensors."""

    # Initialize solution tensor on the same device as predictions.
    solution = torch.zeros_like(predictions, dtype=torch.int)

    # Get sorted indices of predictions in descending order.
    sorted_predict_labels = torch.argsort(-predictions)

    # Convert the adjacency matrix to a CSR format if it's not already.
    csr_adj_matrix = adj_matrix.tocsr()

    # Process each node according to the sorted prediction order.
    for i in sorted_predict_labels:
        next_node = i.item()  # Get scalar index

        # Skip if the node has been set to -1 (i.e., already processed).
        if solution[next_node] == -1:
            continue

        # Set neighbors of the current node to -1.
        neighbors = csr_adj_matrix[next_node].nonzero()[1]
        for neighbor in neighbors:
            solution[neighbor] = -1

        # Set the current node as part of the solution.
        solution[next_node] = 1

    return (solution == 1).int()
