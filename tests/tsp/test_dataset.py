import os

import pytest
from problems.mis.mis_dataset import MISDataset
from problems.tsp.tsp_graph_dataset import TSPGraphDataset

datasets = {
    "tsp50_test_concorde.txt": {
        "sparse_factor": -1,
        "n": 50,
        "split": "test",
        "len": 1280,
    },
    "tsp50_train_concorde.txt": {
        "sparse_factor": -1,
        "n": 50,
        "split": "train",
        "len": 1502000,
    },
    "tsp100_test_concorde.txt": {
        "sparse_factor": -1,
        "n": 100,
        "split": "test",
        "len": 1280,
    },
    "tsp100_train_concorde.txt": {
        "sparse_factor": -1,
        "n": 100,
        "split": "train",
        "len": 1502000,
    },
    "tsp500_test_concorde.txt": {
        "sparse_factor": 50,
        "n": 500,
        "split": "test",
        "len": 128,
    },
    "tsp1000_test_concorde.txt": {
        "sparse_factor": 100,
        "n": 1000,
        "split": "test",
        "len": 128,
    },
    "tsp10000_test_concorde.txt": {
        "sparse_factor": 100,
        "n": 10000,
        "split": "test",
        "len": 16,
    },
}


SKIP_REASON = "data/tsp directory does not exist"


@pytest.mark.skipif(not os.path.exists("data/tsp"), reason=SKIP_REASON)
@pytest.mark.parametrize(("dataset_name", "dataset_params"), datasets.items())
def test_tsp_datasets(dataset_name: str, dataset_params: dict) -> None:
    data_file = f"data/tsp/{dataset_name}"

    dataset = TSPGraphDataset(
        data_file=data_file,
        sparse_factor=dataset_params["sparse_factor"],
    )
    assert len(dataset) == dataset_params["len"]

    it = 0
    n = dataset_params["n"]
    sparse_factor = dataset_params["sparse_factor"]
    while it < min(30, dataset_params["len"]):
        sample = dataset.__getitem__(it)
        if sparse_factor < 0:
            assert sample[0].item() == it
            assert sample[1].shape == (n, 2)
            assert sample[2].shape == (n, n)
            assert sample[3].shape == (n + 1,)

            adj_matrix = sample[2]
            tour = sample[3]
            for i in range(tour.shape[0] - 1):
                assert tour[i] != tour[i + 1]
                assert adj_matrix[tour[i], tour[i + 1]] == 1
        else:
            m = n * sparse_factor
            assert sample[0].item() == it
            assert sample[1].x.shape == (n, 2)
            assert sample[1].edge_index.shape == (2, m)
            assert sample[1].edge_attr.shape == (m, 1)
            assert sample[2].shape == (1,)
            assert sample[3].shape == (1,)

            assert sample[2].item() == n
            assert sample[3].item() == m
        it += 1


def test_tsp_dataset_sample() -> None:
    dataset = TSPGraphDataset(
        data_file="tests/resources/tsp50_example_dataset.txt",
        sparse_factor=0,
    )

    sample = dataset.__getitem__(0)

    assert len(sample) == 4
    assert sample[1].shape == (50, 2)

    adj_matrix = sample[2]
    assert adj_matrix.shape == (50, 50)

    tour = sample[3]
    assert tour.shape == (50 + 1,)

    for i in range(tour.shape[0] - 1):
        assert tour[i] != tour[i + 1]
        assert adj_matrix[tour[i], tour[i + 1]] == 1


def test_mis_dataset_is_loaded() -> None:
    dataset = MISDataset(
        data_dir="tests/resources/er_example_dataset",
    )
    assert len(dataset) == 2
