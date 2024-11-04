from argparse import Namespace

import torch
from torch_geometric.data import Batch

from difusco.arg_parser import get_arg_parser
from difusco.node_selection.mis_dataset import MISDataset
from difusco.node_selection.pl_mis_model import MISModel


def test_mis_test_step() -> None:
    example_dataset = "er_example_dataset/"

    arg_parser = get_arg_parser()
    args = Namespace(**{action.dest: action.default for action in arg_parser._actions})  # noqa: SLF001
    args.task = "mis"
    args.project_name = "difusco"
    args.data_path = "tests/resources"
    args.models_path = "models"
    args.logs_path = "logs"
    args.training_split = example_dataset
    args.validation_split = example_dataset
    args.test_split = example_dataset
    args.training_split_label_dir = "tests/resources/er_example_dataset_annotations"
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
    args.wandb_logger_name = f"{args.task}_diffusion_graph_{args.diffusion_type}_justatest"
    args.do_test = True
    args.profiler = True

    dataset = MISDataset(
        data_dir=f"tests/resources/{example_dataset}",
    )

    sample_1 = dataset.__getitem__(0)

    # Create a batch of one sample
    idx = torch.tensor([sample_1[0]])
    graph_batch = Batch.from_data_list([sample_1[1]])
    n_pts = torch.stack([sample_1[2]])

    batch = (idx, graph_batch, n_pts)

    model = MISModel(args)
    model.log = lambda name, value, *args, **kwargs: None  # noqa: ARG005

    model.test_step(batch, 0, split="test")

    assert model.test_outputs is not None
    assert len(model.test_outputs) == 1
    metrics = model.test_outputs[0]
    assert metrics is not None
    assert len(metrics) == 1


def test_categorical_training_step() -> None:
    example_dataset = "er_example_dataset"

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

    dataset = MISDataset(
        data_dir=f"tests/resources/{example_dataset}",
    )

    sample_1 = dataset.__getitem__(0)
    sample_2 = dataset.__getitem__(1)

    # Create a batch of two samples
    idx = torch.tensor([sample_1[0], sample_2[0]])

    # Create a data batch using torch_geometric's Batch class
    graph_batch = Batch.from_data_list([sample_1[1], sample_2[1]])

    # Stack the point indicators
    n_pts = torch.stack([sample_1[2], sample_2[2]])

    batch = (idx, graph_batch, n_pts)

    model = MISModel(args)
    model.log = lambda name, value, *args, **kwargs: None  # noqa: ARG005

    loss = model.categorical_training_step(batch, 0)

    assert loss is not None
    assert not loss.shape
    assert isinstance(loss.item(), float)
