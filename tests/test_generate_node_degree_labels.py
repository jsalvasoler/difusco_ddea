import os
from argparse import Namespace
from typing import Any, Generator

import pytest

from difusco.node_selection.generate_node_degree_labels import generate_node_degree_labels


@pytest.fixture
def setup_and_teardown() -> Generator[Any, Any, Any]:
    opts = Namespace()
    opts.train_graphs_dir = "tests/resources/er_example_dataset"
    opts.output_dir = "tests/resources/tmp_annotations"
    os.makedirs(opts.output_dir, exist_ok=True)
    yield opts
    os.system(f"rm -r {opts.output_dir}")


def test_generate_labels_content(setup_and_teardown: Generator) -> None:
    opts = setup_and_teardown

    generate_node_degree_labels(opts)

    output_files = os.listdir(opts.output_dir)
    input_files = os.listdir(opts.train_graphs_dir)
    assert len(output_files) == len(input_files)
    for file in output_files:
        assert os.path.isfile(os.path.join(opts.output_dir, file))

    for file in os.listdir(opts.output_dir):
        with open(os.path.join(opts.output_dir, file)) as f:
            lines = f.readlines()
            assert len(lines) in [732, 770]
            assert all(line.strip().isdigit() for line in lines)
            assert {line.strip() for line in lines}.issubset({"0", "1"})
