import os
from argparse import Namespace

import numpy as np
import torch
from problems.mis.mis_dataset import MISDataset
from torch_geometric.data import Batch

from difusco.arg_parser import get_arg_parser
from difusco.mis.pl_high_degree_model import HighDegreeSelection, high_degree_decode_np


def test_the_test_step() -> None:
    arg_parser = get_arg_parser()
    args = Namespace(**{action.dest: action.default for action in arg_parser._actions})  # noqa: SLF001
    args.data_path = "tests/resources"
    args.models_path = "models"
    args.training_split = "er_example_dataset"
    args.validation_split = "er_example_dataset"
    args.test_split = "er_example_dataset"
    args.test_split_label_dir = "er_example_dataset_annotations"
    args.diffusion_type = "categorical"
    args.learning_rate = 0.0002
    args.weight_decay = 0.0001
    args.lr_scheduler = "cosine-decay"
    args.batch_size = 4
    args.sequential_sampling = 2
    args.inference_schedule = "cosine"
    args.inference_diffusion_steps = 2

    dataset = MISDataset(
        data_dir=os.path.join(args.data_path, args.test_split),
        data_label_dir=os.path.join(args.data_path, args.test_split_label_dir),
    )

    model = HighDegreeSelection(param_args=args)
    model.log = lambda name, value, *args, **kwargs: None  # noqa: ARG005

    sample = dataset.__getitem__(0)

    # Create a batch of two samples
    idx = torch.tensor([sample[0]])

    # Create a data batch using torch_geometric's Batch class
    graph_batch = Batch.from_data_list([sample[1]])

    # Stack the point indicators
    n_pts = torch.stack([sample[2]])

    batch = (idx, graph_batch, n_pts)

    model.test_step(batch, 0, split="test")

    assert model.test_outputs is not None
    assert len(model.test_outputs) == 1


def test_high_degree_decode_np() -> None:
    prediction = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    decoded = high_degree_decode_np(prediction)

    assert decoded is not None
    assert decoded.shape == prediction.shape
    assert decoded.sum() == 5


def test_compute_accuracy() -> None:
    predictions = [
        np.array([1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0]),
    ]
    gt = np.array([1, 1, 1, 1, 1])

    acc = HighDegreeSelection.compute_accuracy(predictions, gt)
    assert acc == 0.5
