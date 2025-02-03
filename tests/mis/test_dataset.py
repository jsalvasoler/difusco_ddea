import os
from pathlib import Path

import numpy as np
import pytest
from problems.mis.mis_dataset import MISDataset

graph_sizes = {
    "er_50_100": (50, 100),
    "er_300_400": (300, 400),
    "er_700_800": (700, 800),
    "er_1300_1500": (1300, 1500),
    "satlib": (1200, 1400),
}

expected_length_test = {
    "er_50_100": 128,
    "er_300_400": 128,
    "er_700_800": 128,
    "er_1300_1500": 500,
    "satlib": 500,
}

expected_length_train = {
    "er_50_100": 40000 - 128,
    "er_300_400": 40000 - 128,
    "er_700_800": 163837,
    "er_1300_1500": None,
    "satlib": 40000 - 500,
}

MIS_DATA_SKIP_REASON = "MIS dataset directory not found"


def extract_id(s: str, kind: str = "data") -> int:
    if kind == "data":
        return int(s.split(".")[-2].split("_")[-1])
    if kind == "label":
        return int(s.split(".")[-2].split("_")[-2])
    raise ValueError(f"Invalid kind: {kind}")


def check_dataset_samples(dataset: MISDataset, dataset_name: str, start_idx: int = 0, num_samples: int = 30) -> None:
    """Check the format and constraints of samples from a MIS dataset.

    Args:
        dataset: The dataset to check
        dataset_name: Name of the dataset for size validation
        start_idx: Starting index for sampling
        num_samples: Number of samples to check
    """
    for it in range(start_idx, start_idx + num_samples):
        sample = dataset.__getitem__(it)
        assert len(sample) == 3

        # First element is the idx
        assert sample[0].shape == (1,)
        assert sample[0].item() == it

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

        assert graph_sizes[dataset_name][0] <= n <= graph_sizes[dataset_name][1]
        assert 0 < graph_data.x.sum().item() <= n


@pytest.mark.skipif(not Path("data/mis").exists(), reason=MIS_DATA_SKIP_REASON)
@pytest.mark.parametrize("dataset_name", ["er_50_100", "er_300_400", "er_700_800", "er_1300_1500", "satlib"])
def test_er_datasets(dataset_name: str) -> None:
    """
    We have 5 MIS datasets:
    1. er_50_100 (train, test)
    2. er_300_400 (train, test)
    3. er_700_800 (train, test)
    4. er_1300_1500 (test)
    5. satlib (train, test)
    """
    test_dataset = MISDataset(
        data_dir=f"data/mis/{dataset_name}/test", data_label_dir=f"data/mis/{dataset_name}/test_labels"
    )
    assert len(test_dataset) == expected_length_test[dataset_name]
    check_dataset_samples(test_dataset, dataset_name)

    if expected_length_train[dataset_name] is None:
        return

    train_dataset = MISDataset(
        data_dir=f"data/mis/{dataset_name}/train", data_label_dir=f"data/mis/{dataset_name}/train_labels"
    )
    assert len(train_dataset) == expected_length_train[dataset_name]
    check_dataset_samples(train_dataset, dataset_name)


@pytest.mark.skipif(not Path("data/mis").exists(), reason=MIS_DATA_SKIP_REASON)
@pytest.mark.parametrize(
    ("dataset_name", "split"),
    [
        ("er_50_100", "train"),
        ("er_50_100", "test"),
        ("er_300_400", "train"),
        ("er_300_400", "test"),
        ("er_700_800", "train"),
        ("er_700_800", "test"),
        ("er_1300_1500", "test"),
    ],
)
def test_er_train_annotations_match(dataset_name: str, split: str) -> None:
    base = "data/mis"
    train = os.path.join(base, f"{dataset_name}/{split}")
    print(len(os.listdir(train)))
    train_ann = os.path.join(base, f"{dataset_name}/{split}_labels")
    print(len(os.listdir(train_ann)))

    def extract_id(s: str) -> int:
        return int(s.split(".")[1].split("_")[1])

    ids_train_ann = sorted(extract_id(f) for f in os.listdir(train_ann) if f.endswith(".result"))
    ids_train = sorted(extract_id(f) for f in os.listdir(train) if f.endswith(".gpickle"))

    # print the two substractions
    print(set(ids_train_ann) - set(ids_train))
    print(set(ids_train) - set(ids_train_ann))

    assert set(ids_train_ann) == set(ids_train)


def test_get_file_name_from_sample_idx() -> None:
    dataset_name = "er_50_100"
    dataset = MISDataset(
        data_dir=f"data/mis/{dataset_name}/test", data_label_dir=f"data/mis/{dataset_name}/test_labels"
    )
    assert dataset.get_file_name_from_sample_idx(0) == "ER_50_100_0.15_0.gpickle"
    assert dataset.get_file_name_from_sample_idx(120) == "ER_50_100_0.15_120.gpickle"


def dataloader_for_er_700_800_train() -> None:
    dataset_name = "er_700_800"
    dataset = MISDataset(
        data_dir=f"data/mis/{dataset_name}/train", data_label_dir=f"data/mis/{dataset_name}/train_labels"
    )

    for i in range(len(dataset)):
        print(f"getting sample {i} - {dataset.sample_files[i]}")
        # get label file
        label_file = os.path.join(
            dataset.data_label_dir, os.path.basename(dataset.sample_files[i]).replace(".gpickle", "_unweighted.result")
        )
        print(f"label file: {label_file}")
        dataset.__getitem__(i)


if __name__ == "__main__":
    dataloader_for_er_700_800_train()
