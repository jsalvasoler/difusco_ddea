import argparse
import os

import numpy as np
import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import degree
from tqdm import tqdm

from difusco.node_selection.mis_dataset import MISDataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_graphs_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    opts, _ = parser.parse_known_args()

    assert opts.train_graphs_dir is not None, "Must provide train_graphs_dir"
    assert opts.output_dir is not None, "Must provide output_dir"

    # Make sure they exist
    assert os.path.exists(opts.train_graphs_dir), f"Path {opts.train_graphs_dir} does not exist."
    assert os.path.exists(opts.output_dir), f"Path {opts.output_dir} does not exist."

    return opts


def generate_node_degree_labels(opts: argparse.Namespace) -> None:
    dataset = MISDataset(
        data_dir=opts.train_graphs_dir,
    )

    example_filename = os.listdir(opts.train_graphs_dir)[0]
    # prefix is what comes to the left of last underscore
    prefix = "_".join(example_filename.split("_")[:-1])
    # suffix is what coes to the right of the only .
    suffix = example_filename.split(".")[-1]

    for i in tqdm(range(len(dataset))):
        num_nodes, node_labels, edge_index = dataset.get_example(i)
        graph_data = GraphData(x=torch.from_numpy(node_labels), edge_index=torch.from_numpy(edge_index))

        # Calculate the degree of each node
        node_degree = degree(graph_data.edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        mean_degree = torch.mean(node_degree)
        # Label the nodes with 1 if the degree is above the mean, 0 otherwise
        node_labels = torch.where(node_degree > mean_degree, torch.tensor(1), torch.tensor(0))
        # Save the node labels with the same name as the graph
        filename = f"{prefix}_{i}.{suffix}"
        # File contains one line per node, with a 0 or 1 indicating the label
        np.savetxt(os.path.join(opts.output_dir, filename), node_labels.numpy(), fmt="%d")


def generate_node_degree_labels_main() -> None:
    opts = parse_arguments()
    generate_node_degree_labels(opts)



if __name__ == "__main__":
    generate_node_degree_labels_main()