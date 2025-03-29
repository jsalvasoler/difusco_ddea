from __future__ import annotations

import shutil

import pytest
from difuscombination.dataset import MISDatasetComb


def test_mis_dataset_comb_initialization_and_len() -> None:
    dataset = MISDatasetComb(
        samples_file="data/difuscombination/mis/er_50_100/test/difuscombination_samples_2025-01-30_14-36-45.csv",
        graphs_dir="data/mis/er_50_100/test",
        labels_dir="data/difuscombination/mis/er_50_100/test_labels",
    )

    assert len(dataset) == 128


def test_mis_dataset_comb_assertion(tmp_path) -> None:  # noqa: ANN001
    # copy the data to a temporary directory
    shutil.copytree("data/mis/er_50_100/test", tmp_path / "graphs")
    shutil.copytree("data/difuscombination/mis/er_50_100/test_labels", tmp_path / "labels")
    shutil.copy(
        "data/difuscombination/mis/er_50_100/test/difuscombination_samples_2025-01-30_14-36-45.csv",
        tmp_path / "samples.csv",
    )

    # add a line in the samples.csv that is not in the graphs dir
    with open(tmp_path / "samples.csv", "a") as f:
        f.write("graph_100.gpickle,data/mis/er_50_100/test/graph_100.gpickle\n")

    with pytest.raises(AssertionError, match="Some instance files are not in"):
        _ = MISDatasetComb(
            samples_file=tmp_path / "samples.csv",
            graphs_dir=tmp_path / "graphs",
            labels_dir=tmp_path / "labels",
        )


def test_mis_dataset_comb_getitem() -> None:
    dataset = MISDatasetComb(
        samples_file="data/difuscombination/mis/er_50_100/test/difuscombination_samples_2025-01-30_14-36-45.csv",
        graphs_dir="data/mis/er_50_100/test",
        labels_dir="data/difuscombination/mis/er_50_100/test_labels",
    )

    item = dataset.__getitem__(0)

    # first item is the index
    assert item[0].item() == 0

    # third item is the number of points
    num_points = item[2].item()
    assert 50 <= num_points <= 100

    # second item is the graph with features
    graph = item[1]
    assert graph.x.shape[0] == num_points
    assert graph.x.shape[1] == 3
    assert graph.num_nodes == num_points
