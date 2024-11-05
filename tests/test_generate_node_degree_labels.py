import os
from argparse import Namespace
from typing import Any, Generator

import pytest

from difusco.node_selection.generate_node_degree_labels import generate_node_degree_labels
from difusco.node_selection.mis_dataset import MISDataset


@pytest.fixture
def setup_and_teardown() -> Generator[Any, Any, Any]:
    opts = Namespace()
    opts.train_graphs_dir = "tests/resources/er_example_dataset"
    opts.output_dir = "tests/resources/tmp_annotations"
    os.system(f"rm -r {opts.output_dir}")
    os.makedirs(opts.output_dir, exist_ok=True)
    yield opts
    os.system(f"rm -r {opts.output_dir}")


@pytest.fixture
def setup_and_teardown_2() -> Generator[Any, Any, Any]:
    opts = Namespace()
    opts.train_graphs_dir = "tests/resources/er_example_node_deg"
    opts.output_dir = "tests/resources/tmp_annotations"
    os.system(f"rm -r {opts.output_dir}")
    os.makedirs(opts.output_dir, exist_ok=True)
    yield opts
    os.system(f"rm -r {opts.output_dir}")


def test_generate_labels_on_examples(setup_and_teardown: Generator) -> None:
    opts = setup_and_teardown

    generate_node_degree_labels(opts)

    output_files = os.listdir(opts.output_dir)
    input_files = os.listdir(opts.train_graphs_dir)
    assert len(output_files) == len(input_files)
    for file in output_files:
        assert os.path.isfile(os.path.join(opts.output_dir, file))

    for file in os.listdir(opts.output_dir):
        suff = "_unweighted.result"
        pref = "ER_700_800_0.15_"
        assert file.endswith(suff)
        assert file.startswith(pref)
        # remove prefix and suffix
        idx = file[len(pref) : -len(suff)]
        assert 0 <= int(idx) <= 170, 000
        with open(os.path.join(opts.output_dir, file)) as f:
            lines = f.readlines()
            assert len(lines) in [732, 770]
            assert all(line.strip().isdigit() for line in lines)
            assert {line.strip() for line in lines}.issubset({"0", "1"})


def test_generate_labels_on_ten_and_test_mis_dataset(setup_and_teardown_2: Generator) -> None:
    opts = setup_and_teardown_2

    generate_node_degree_labels(opts)

    output_files = os.listdir(opts.output_dir)
    input_files = os.listdir(opts.train_graphs_dir)

    assert len(output_files) == len(input_files)

    dataset = MISDataset(
        data_dir=opts.train_graphs_dir,
        data_label_dir=opts.output_dir,
    )

    assert len(dataset) == len(input_files)

    for idx in range(len(dataset)):
        sample = dataset.__getitem__(idx)
        assert len(sample) == 3
        assert sample[0].shape == (1,)
        assert sample[0].item() == idx
        graph_data = sample[1]
        n = graph_data.num_nodes
        m = graph_data.num_edges
        assert graph_data.x.shape == (n,)
        assert graph_data.edge_index.shape == (2, m)
        true_labels = sum(graph_data.x.numpy())
        assert 0 < true_labels <= n
        assert sample[2].shape == (1,)
        assert sample[2].item() == n
