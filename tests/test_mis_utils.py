import time

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm

from difusco.mis.mis_dataset import MISDataset
from difusco.mis.utils import mis_decode_np, mis_decode_torch


@pytest.fixture
def adj_matrix() -> sp.csr_matrix:
    return sp.csr_matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]).tocsr()


def test_mis_decoding_numpy(adj_matrix: sp.csr_matrix) -> None:
    predictions = np.array([0.9, 0.1, 0.8, 0.3])

    result = mis_decode_np(predictions, adj_matrix)

    expected_result = np.array([1, 0, 1, 0])

    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_mis_decoding_torch(adj_matrix: sp.csr_matrix) -> None:
    predictions = torch.tensor([0.9, 0.1, 0.8, 0.3])

    result = mis_decode_torch(predictions, adj_matrix)

    expected_result = torch.tensor([1, 0, 1, 0])

    assert torch.equal(result, expected_result), f"Expected {expected_result}, but got {result}"


def benchmark_experiment() -> None:
    device = "cuda:0"  # "cpu" or "cuda"

    dataset = MISDataset(data_dir="/home/e12223411/repos/difusco/data/mis/er_test")
    times_np = []
    times_torch = []
    for sample in tqdm(dataset):
        _, graph_data, _ = sample
        node_labels = graph_data.x
        edge_index = graph_data.edge_index

        edge_index = edge_index.to(node_labels.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        ).tocsr()

        predictions = torch.randn(node_labels.shape[0], device=device)

        start_time_np = time.time()
        pred = mis_decode_np(predictions.cpu().numpy(), adj_mat)
        times_np.append(time.time() - start_time_np)

        start_time_torch = time.time()
        pred_torch = mis_decode_torch(predictions, adj_mat)
        times_torch.append(time.time() - start_time_torch)

        assert np.array_equal(pred_torch.cpu().numpy(), pred)

    print(f"Samples: {len(times_np)}")
    print(f"Average time for numpy: {np.mean(times_np)}")
    print(f"Average time for torch: {np.mean(times_torch)}")


if __name__ == "__main__":
    # test_mis_decoding_numpy()
    benchmark_experiment()
