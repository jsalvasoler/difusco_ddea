import os

import numpy as np
from problems.mis.mis_dataset import MISDataset
from problems.tsp.tsp_graph_dataset import TSPGraphDataset


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
    assert len(dataset) == 163838

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


# def test_complete_mis_train_dataset_er() -> None:
#     data_path = "data/mis"
#     dataset = MISDataset(
#         data_dir=os.path.join(data_path, "er_train"),
#         data_label_dir=os.path.join(data_path, "er_train_annotations_mis"),
#     )
#     assert len(dataset) == 163838

#     def fake_get_example(self, idx: int) -> tuple:
#         with open(self.sample_files[idx], "rb") as f:
#             graph = pickle.load(f)

#         num_nodes = graph.number_of_nodes()

#         base_label_file = os.path.basename(self.sample_files[idx]).replace(".gpickle", "_unweighted.result")
#         node_label_file = os.path.join(self.data_label_dir, base_label_file)
#         with open(node_label_file) as f:
#             node_labels = [int(_) for _ in f.read().splitlines()]
#         node_labels = np.array(node_labels, dtype=np.int64)
#         if node_labels.shape[0] != num_nodes:
#             print(f"Mismatch in node labels for {idx}")

#     with unittest.mock.patch("problems.mis.mis_dataset.MISDataset.__getitem__", fake_get_example):
#         for i in tqdm(range(len(dataset))):
#             _ = dataset.get_example(i)


def test_er_train_annotations_match() -> None:
    base = "/home/e12223411/repos/difusco/data/mis"
    train = os.path.join(base, "er_train")
    print(len(os.listdir(train)))
    train_ann = os.path.join(base, "er_train_annotations_mis")
    print(len(os.listdir(train_ann)))

    # extract id:
    def extract_id(s: str) -> int:
        return int(s.split(".")[1].split("_")[1])

    ids_train_ann = sorted(extract_id(f) for f in os.listdir(train_ann) if f.endswith(".result"))

    ids_train = sorted(extract_id(f) for f in os.listdir(train) if f.endswith(".gpickle"))

    # print the two substractions
    print(set(ids_train_ann) - set(ids_train))
    print(set(ids_train) - set(ids_train_ann))

    assert set(ids_train_ann) == set(ids_train)


if __name__ == "__main__":
    # test_tsp_dataset_is_loaded()
    # test_tsp_dataset_sample()
    # test_mis_real_test_sample()
    # test_mis_real_train_sample()
    test_er_train_annotations_match()
