import os
from argparse import Namespace

import numpy as np
import pytest
import torch
from problems.tsp.tsp_evaluation import TSPEvaluator

from difusco.arg_parser import get_arg_parser, validate_args
from difusco.difusco_main import difusco
from difusco.tsp.pl_tsp_model import TSPGraphDataset, TSPModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping test that requires GPU")
def test_categorical_tsp_training_step() -> None:
    example_dataset = "tsp50_example_dataset.txt"

    arg_parser = get_arg_parser()
    args = Namespace(**{action.dest: action.default for action in arg_parser._actions})  # noqa: SLF001
    args.data_path = "tests/resources"
    args.models_path = "models"
    args.training_split = example_dataset
    args.validation_split = example_dataset
    args.test_split = example_dataset
    args.diffusion_type = "categorical"
    args.learning_rate = 0.0002
    args.weight_decay = 0.0001
    args.lr_scheduler = "cosine-decay"
    args.batch_size = 32
    args.num_epochs = 25
    args.inference_schedule = "cosine"
    args.inference_diffusion_steps = 50
    args.resume_weight_only = True

    model = TSPModel(param_args=args)
    model.log = lambda name, value, *args, **kwargs: None  # noqa: ARG005

    dataset = TSPGraphDataset(
        data_file="tests/resources/tsp50_example_dataset.txt",
        sparse_factor=0,
    )

    idx, points, adj_matrix, gt_tour = dataset.__getitem__(0)
    idx2, points2, adj_matrix2, gt_tour2 = dataset.__getitem__(1)

    idx = torch.tensor([idx, idx2])
    points = torch.stack([points, points2])
    adj_matrix = torch.stack([adj_matrix, adj_matrix2])
    gt_tour = torch.stack([gt_tour, gt_tour2])

    batch = (idx, points, adj_matrix, gt_tour)
    loss = model.categorical_training_step(batch, 0)

    assert loss is not None
    assert not loss.shape
    assert isinstance(loss.item(), float)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping test that requires GPU")
def test_tsp_test_step() -> None:
    example_dataset = "tsp50_example_dataset.txt"

    arg_parser = get_arg_parser()
    args = Namespace(**{action.dest: action.default for action in arg_parser._actions})  # noqa: SLF001
    args.data_path = "tests/resources"
    args.models_path = "models"
    args.training_split = example_dataset
    args.validation_split = example_dataset
    args.test_split = example_dataset
    args.diffusion_type = "categorical"
    args.learning_rate = 0.0002
    args.weight_decay = 0.0001
    args.lr_scheduler = "cosine-decay"
    args.batch_size = 32
    args.num_epochs = 25
    args.inference_schedule = "cosine"
    args.inference_diffusion_steps = 50
    args.resume_weight_only = True

    model = TSPModel(param_args=args)
    model.log = lambda name, value, *args, **kwargs: None  # noqa: ARG005

    dataset = TSPGraphDataset(
        data_file="tests/resources/tsp50_example_dataset.txt",
        sparse_factor=0,
    )

    idx, points, adj_matrix, gt_tour = dataset.__getitem__(0)

    # Create a batch of size 1 using unsqueeze
    idx = torch.tensor([idx]).unsqueeze(0)
    points = points.unsqueeze(0)
    adj_matrix = adj_matrix.unsqueeze(0)
    gt_tour = gt_tour.unsqueeze(0)

    batch = (idx, points, adj_matrix, gt_tour)

    model.test_step(batch, 0, split="test")

    assert model.test_outputs is not None
    assert len(model.test_outputs) == 1
    metrics = model.test_outputs[0]
    assert metrics is not None
    assert len(metrics) == 3

    # check the ground truth tour length
    evaluator = TSPEvaluator(points[0].numpy())
    gt_tour_length = evaluator.evaluate(gt_tour[0].numpy())

    assert metrics["test/gt_cost"] == gt_tour_length


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping test that requires GPU")
def test_tsp_test_step_saving_heatmaps() -> None:
    example_dataset = "tsp50_example_dataset_two_samples.txt"

    arg_parser = get_arg_parser()
    args = Namespace(**{action.dest: action.default for action in arg_parser._actions})  # noqa: SLF001
    args.task = "tsp"
    args.data_path = "tests/resources"
    args.models_path = "models"
    args.logs_path = "logs"
    args.training_split = example_dataset
    args.validation_split = example_dataset
    args.test_split = example_dataset
    args.diffusion_type = "categorical"
    args.two_opt_iterations = 0
    args.learning_rate = 0.0002
    args.weight_decay = 0.0001
    args.lr_scheduler = "cosine-decay"
    args.batch_size = 1
    args.inference_schedule = "cosine"
    args.inference_diffusion_steps = 2
    args.validation_examples = 1
    args.resume_weight_only = True
    args.save_numpy_heatmap = True
    args.wandb_logger_name = f"{args.task}_diffusion_graph_{args.diffusion_type}_justatest"
    args.do_test = True

    validate_args(args)
    difusco(args)

    wandb_path = os.path.join(args.logs_path, "wandb", "latest-run")
    files = os.listdir(wandb_path)
    file_wandb = next(file for file in files if file.endswith(".wandb"))
    run_id = file_wandb.split(".")[0].split("-")[-1]

    heatmap_path = os.path.join(args.logs_path, args.wandb_logger_name, run_id, "numpy_heatmap")
    # check that there is only one val heatmap
    heatmaps = os.listdir(heatmap_path)
    assert len(heatmaps) == 2 * 2 + 2
    test_heatmaps = [heatmap for heatmap in heatmaps if "test" in heatmap]
    assert len(test_heatmaps) == 2 * 2

    # load the heatmap and check the shape
    heatmap = np.load(os.path.join(heatmap_path, "test-heatmap-0.npy"))
    points = np.load(os.path.join(heatmap_path, "test-points-0.npy"))
    assert heatmap.shape == (1, 50, 50)
    assert points.shape == (50, 2)
