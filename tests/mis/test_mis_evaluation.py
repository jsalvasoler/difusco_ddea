import numpy as np
import pytest
import scipy.sparse as sp
import torch
from problems.mis.mis_evaluation import (
    mis_decode_np,
    mis_decode_torch,
    mis_decode_torch_batched,
    precompute_neighbors_padded,
)


@pytest.fixture
def adj_matrix() -> sp.csr_matrix:
    return sp.csr_matrix(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    ).tocsr()


@pytest.fixture
def adj_matrix_torch() -> torch.sparse.FloatTensor:
    """
    Corresponds to the graph:
    0 - 1 - 2 - 3
    Therefore, both {0, 2} or {1, 3} are MISs.
    """
    dense_adj_mat = torch.tensor(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    )
    return dense_adj_mat.to_sparse()


def test_mis_decoding_numpy(adj_matrix: sp.csr_matrix) -> None:
    predictions = np.array([0.9, 0.1, 0.8, 0.3])

    result = mis_decode_np(predictions, adj_matrix)

    expected_result = np.array([1, 0, 1, 0])

    assert np.array_equal(result, expected_result), (
        f"Expected {expected_result}, but got {result}"
    )


def test_mis_decoding_torch(adj_matrix_torch: torch.FloatTensor) -> None:
    predictions = torch.tensor([0.9, 0.1, 0.8, 0.3], dtype=torch.float32)

    result = mis_decode_torch(predictions, adj_matrix_torch)

    expected_result = torch.tensor([1, 0, 1, 0])

    assert torch.equal(result, expected_result), (
        f"Expected {expected_result}, but got {result}"
    )


def test_mis_decoding_torch_batched(adj_matrix_torch: torch.FloatTensor) -> None:
    predictions = torch.tensor(
        [[0.9, 0.1, 0.8, 0.3], [0.1, 0.9, 0.2, 0.7]], dtype=torch.float32
    ).cpu()

    adj_csr = adj_matrix_torch.to_sparse_csr()
    neighbors_padded, degrees = precompute_neighbors_padded(adj_csr)

    result = mis_decode_torch_batched(predictions, neighbors_padded, degrees)

    expected_result = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool).cpu()
    assert torch.equal(result, expected_result), (
        f"Expected {expected_result}, but got {result}"
    )


def test_mis_decoding_torch_batched_on_single_tensor(
    adj_matrix_torch: torch.FloatTensor,
) -> None:
    predictions = torch.tensor([0.9, 0.1, 0.8, 0.3], dtype=torch.float32)
    adj_csr = adj_matrix_torch.to_sparse_csr()
    neighbors_padded, degrees = precompute_neighbors_padded(adj_csr)
    result = mis_decode_torch_batched(predictions, neighbors_padded, degrees)
    assert result.shape == (4,)
    assert (result == torch.tensor([1, 0, 1, 0])).all()
