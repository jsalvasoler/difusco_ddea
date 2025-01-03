from problems.mis.mis_dataset import MISDataset
from problems.tsp.tsp_graph_dataset import TSPGraphDataset


def test_tsp_dataset_is_loaded() -> None:
    dataset = TSPGraphDataset(
        data_file="tests/resources/tsp50_example_dataset.txt",
        sparse_factor=0.5,
    )
    assert len(dataset) == 30


def test_mis_dataset_is_loaded() -> None:
    dataset = MISDataset(
        data_dir="tests/resources/er_example_dataset",
    )
    assert len(dataset) == 2
