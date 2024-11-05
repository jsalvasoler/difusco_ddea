import argparse
import os
import pickle

import numpy as np
import torch
from torch_geometric.utils import degree
from tqdm import tqdm
from torch_geometric.data import Data as GraphData


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_graphs_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    opts, _ = parser.parse_known_args()

    assert opts.train_graphs_dir is not None, "Must provide train_graphs_dir"
    assert opts.output_dir is not None, "Must provide output_dir"

    # Make sure they exist
    assert os.path.exists(opts.train_graphs_dir), f"Path {opts.train_graphs_dir} does not exist."
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    return opts


def generate_node_degree_labels(opts: argparse.Namespace) -> None:
    files = os.listdir(opts.train_graphs_dir)

    for file in tqdm(files):
        if not file.endswith(".gpickle"):
            continue
        with open(os.path.join(opts.train_graphs_dir, file), "rb") as f:
            graph = pickle.load(f)  # noqa: S301

            num_nodes = graph.number_of_nodes()
            edges = np.array(graph.edges, dtype=np.int64)
            # add inverse edges
            edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
            # add self loop
            self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
            edges = np.concatenate([edges, self_loop], axis=0)
            edges = edges.T

            node_labels = np.zeros(num_nodes, dtype=np.int64)
            graph_data = GraphData(x=torch.from_numpy(node_labels), edge_index=torch.from_numpy(edges))

        # Calculate the degree of each node
        node_degree = degree(graph_data.edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        mean_degree = torch.mean(node_degree)
        # Label the nodes with 1 if the degree is above the mean, 0 otherwise
        node_labels = torch.where(node_degree > mean_degree, torch.tensor(1), torch.tensor(0))
        # Save the node labels with the same name as the graph
        filename = file.replace(".gpickle", "_unweighted.result")
        # File contains one line per node, with a 0 or 1 indicating the label
        np.savetxt(os.path.join(opts.output_dir, filename), node_labels.numpy(), fmt="%d")


def generate_node_degree_labels_main() -> None:
    opts = parse_arguments()
    generate_node_degree_labels(opts)


if __name__ == "__main__":
    generate_node_degree_labels_main()
