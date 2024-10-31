import os

import numpy as np

from difusco.mis.mis_dataset import MISDataset
from difusco.tsp.tsp_graph_dataset import TSPGraphDataset


def test_tsp_dataset_is_loaded() -> None:
    dataset = TSPGraphDataset(
        data_file="tests/resources/tsp50_example_dataset.txt",
        sparse_factor=0.5,
    )
    assert len(dataset) == 30


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


def test_mis_dataset_sample() -> None:
    dataset = MISDataset(
        data_dir="tests/resources/er_example_dataset",
    )

    sample = dataset.__getitem__(0)

    assert len(sample) == 3

    # First element is the idx
    assert sample[0].shape == (1,)
    assert sample[0].item() == 0

    # Second element is the graph data
    graph_data = sample[1]
    n = graph_data.num_nodes
    m = graph_data.num_edges
    assert graph_data.x.shape == (n,)
    assert graph_data.edge_index.shape == (2, m)
    # True labels
    assert 0 <= np.sum(graph_data.x.numpy()) <= n

    # Third element is the size indicator
    assert sample[2].shape == (1,)
    assert sample[2].item() == n


def test_mis_real_test_sample() -> None:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset = MISDataset(
        data_dir=f"{root_dir}/data/mis/er_test",
    )
    assert len(dataset) == 128

    sample = dataset.__getitem__(0)

    assert len(sample) == 3

    # First element is the idx
    assert sample[0].shape == (1,)
    assert sample[0].item() == 0

    # Second element is the graph data
    graph_data = sample[1]
    n = graph_data.num_nodes
    m = graph_data.num_edges
    assert graph_data.x.shape == (n,)
    assert graph_data.edge_index.shape == (2, m)
    # True labels
    true_labels = np.sum(graph_data.x.numpy())
    print(true_labels)
    assert 0 <= true_labels <= n

    # Third element is the size indicator
    assert sample[2].shape == (1,)
    assert sample[2].item() == n


def test_mis_real_train_sample() -> None:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset = MISDataset(
        data_dir=f"{root_dir}/data/mis/er_train",
        data_label_dir=f"{root_dir}/data/mis/er_train__annotations",
    )
    assert len(dataset) == 163840

    # Now go to the example dataset. Here we are sure that we generated the ground truth
    dataset = MISDataset(
        data_dir="tests/resources/er_example_dataset",
        data_label_dir="tests/resources/er_example_dataset_annotations",
    )
    assert len(dataset) == 2

    sample = dataset.__getitem__(0)

    assert len(sample) == 3

    # First element is the idx
    assert sample[0].shape == (1,)
    assert sample[0].item() == 0

    # Second element is the graph data
    graph_data = sample[1]
    n = graph_data.num_nodes
    m = graph_data.num_edges
    assert graph_data.x.shape == (n,)
    assert graph_data.edge_index.shape == (2, m)
    # True labels
    true_labels = np.sum(graph_data.x.numpy())
    print(true_labels)
    assert 0 < true_labels <= n, f"true_labels: {true_labels}"

    # Third element is the size indicator
    assert sample[2].shape == (1,)
    assert sample[2].item() == n


if __name__ == "__main__":
    # test_tsp_dataset_is_loaded()
    # test_tsp_dataset_sample()
    test_mis_real_test_sample()
    test_mis_real_train_sample()
