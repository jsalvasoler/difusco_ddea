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
    "er_1300_1500": 128,
    "satlib": 500,
}

MIS_DATA_SKIP_REASON = "MIS dataset directory not found"


def extract_id(s: str, kind: str = "data") -> int:
    if kind == "data":
        return int(s.split(".")[-2].split("_")[-1])
    if kind == "label":
        return int(s.split(".")[-2].split("_")[-2])
    raise ValueError(f"Invalid kind: {kind}")


def check_dataset_samples(
    dataset: MISDataset, dataset_name: str, start_idx: int = 0, num_samples: int = 30
) -> None:
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
@pytest.mark.parametrize(
    "dataset_name", ["er_50_100", "er_300_400", "er_700_800", "er_1300_1500", "satlib"]
)
def test_er_datasets(dataset_name: str) -> None:
    """
    We have 5 MIS datasets:
    1. er_50_100
    2. er_300_400
    3. er_700_800
    4. er_1300_1500
    5. satlib
    """
    test_dataset = MISDataset(
        data_dir=f"data/mis/{dataset_name}/test",
        data_label_dir=f"data/mis/{dataset_name}/test_labels",
    )
    assert len(test_dataset) == expected_length_test[dataset_name]
    check_dataset_samples(test_dataset, dataset_name)


@pytest.mark.skipif(not Path("data/mis").exists(), reason=MIS_DATA_SKIP_REASON)
@pytest.mark.parametrize(
    ("dataset_name", "split"),
    [
        ("er_50_100", "test"),
        ("er_300_400", "test"),
        ("er_700_800", "test"),
        ("er_1300_1500", "test"),
    ],
)
def test_er_train_annotations_match(dataset_name: str, split: str) -> None:
    base = "data/mis"
    dataset = os.path.join(base, f"{dataset_name}/{split}")
    print(len(os.listdir(dataset)))
    dataset_ann = os.path.join(base, f"{dataset_name}/{split}_labels")
    print(len(os.listdir(dataset_ann)))

    def extract_id(s: str) -> int:
        return int(s.split(".")[1].split("_")[1])

    ids_dataset_ann = sorted(
        extract_id(f) for f in os.listdir(dataset_ann) if f.endswith(".result")
    )
    ids_dataset = sorted(
        extract_id(f) for f in os.listdir(dataset) if f.endswith(".gpickle")
    )

    # print the two substractions
    print(set(ids_dataset_ann) - set(ids_dataset))
    print(set(ids_dataset) - set(ids_dataset_ann))

    assert set(ids_dataset_ann) == set(ids_dataset)


@pytest.mark.skipif(not Path("data/mis").exists(), reason=MIS_DATA_SKIP_REASON)
@pytest.mark.parametrize(
    "dataset_name", ["er_50_100", "er_300_400", "er_700_800", "er_1300_1500"]
)
def test_get_file_name_from_sample_idx(dataset_name: str) -> None:
    """Test the get_file_name_from_sample_idx method for all indices in the dataset."""
    dataset_path = Path(f"data/mis/{dataset_name}/test")
    if not dataset_path.exists():
        pytest.skip(f"Dataset directory {dataset_path} does not exist")

    dataset = MISDataset(
        data_dir=str(dataset_path),
        data_label_dir=f"data/mis/{dataset_name}/test_labels",
    )

    # Test that all file names follow the expected pattern
    for idx in range(len(dataset)):
        file_name = dataset.get_file_name_from_sample_idx(idx)
        # Verify file has the correct format with the correct dataset name
        assert file_name.startswith(f"{dataset_name.upper()}_0.15_"), (
            f"Unexpected file name format for idx {idx}: {file_name}"
        )
        # Verify the index in the filename matches the actual index
        file_idx = int(file_name.split("_")[-1].split(".")[0])
        assert file_idx == idx, (
            f"File index {file_idx} does not match sample index {idx}"
        )
