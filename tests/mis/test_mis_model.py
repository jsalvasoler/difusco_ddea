import os
from argparse import Namespace

import pytest
import torch
from difuscombination.dataset import MISDatasetComb
from difuscombination.pl_difuscombination_mis_model import DifusCombinationMISModel
from problems.mis.mis_dataset import MISDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from difusco.arg_parser import get_arg_parser
from difusco.mis.pl_mis_model import MISModel


@pytest.mark.parametrize("diffusion_type", ["categorical", "gaussian"])
def test_mis_test_step(diffusion_type: str) -> None:
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
    args.diffusion_type = diffusion_type
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


@pytest.mark.parametrize("diffusion_type", ["categorical", "gaussian"])
def test_categorical_training_step(diffusion_type: str) -> None:
    example_dataset = "er_example_dataset"

    arg_parser = get_arg_parser()
    args = Namespace(**{action.dest: action.default for action in arg_parser._actions})  # noqa: SLF001
    args.data_path = "tests/resources"
    args.models_path = "models"
    args.training_split = example_dataset
    args.validation_split = example_dataset
    args.test_split = example_dataset
    args.diffusion_type = diffusion_type
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
    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))

    model = MISModel(args)
    model.log = lambda name, value, *args, **kwargs: None  # noqa: ARG005

    loss = model.training_step(batch, 0)

    assert loss is not None
    assert not loss.shape
    assert isinstance(loss.item(), float)


@pytest.mark.parametrize("diffusion_type", ["categorical", "gaussian"])
def test_difuscombination_training_step(diffusion_type: str) -> None:
    from config.configs.mis_inference import config as mis_inf_config

    samples_file = "difuscombination/mis/er_50_100/test"
    graphs_dir = "mis/er_50_100/test"
    labels_dir = "difuscombination/mis/er_50_100/test_labels"

    config = mis_inf_config.update(
        task="mis",
        device="cuda",
        data_path="data",
        training_graphs_dir=graphs_dir,
        training_samples_file=samples_file,
        training_labels_dir=labels_dir,
        test_graphs_dir=graphs_dir,
        test_samples_file=samples_file,
        test_labels_dir=labels_dir,
        validation_graphs_dir=graphs_dir,
        validation_samples_file=samples_file,
        validation_labels_dir=labels_dir,
        diffusion_type=diffusion_type,
        node_feature_only=True,
    )
    model = DifusCombinationMISModel(config)
    model.log = lambda name, value, *args, **kwargs: None  # noqa: ARG005

    dataset = MISDatasetComb(
        samples_file=os.path.join(config.data_path, samples_file),
        graphs_dir=os.path.join(config.data_path, graphs_dir),
        labels_dir=os.path.join(config.data_path, labels_dir),
    )
    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))

    loss = model.training_step(batch, 0)

    assert loss is not None
    assert not loss.shape
    assert isinstance(loss.item(), float)


@pytest.mark.parametrize("diffusion_type", ["categorical", "gaussian"])
def test_difuscombination_test_step(diffusion_type: str) -> None:
    from config.configs.mis_inference import config as mis_inf_config

    samples_file = "difuscombination/mis/er_50_100/test"
    graphs_dir = "mis/er_50_100/test"
    labels_dir = "difuscombination/mis/er_50_100/test_labels"

    config = mis_inf_config.update(
        task="mis",
        device="cuda",
        data_path="data",
        training_graphs_dir=graphs_dir,
        training_samples_file=samples_file,
        training_labels_dir=labels_dir,
        test_graphs_dir=graphs_dir,
        test_samples_file=samples_file,
        test_labels_dir=labels_dir,
        validation_graphs_dir=graphs_dir,
        validation_samples_file=samples_file,
        validation_labels_dir=labels_dir,
        diffusion_type=diffusion_type,
        node_feature_only=True,
        sequential_sampling=1,
        parallel_sampling=1,
    )
    model = DifusCombinationMISModel(config)
    model.log = lambda name, value, *args, **kwargs: None  # noqa: ARG005

    dataset = MISDatasetComb(
        samples_file=os.path.join(config.data_path, samples_file),
        graphs_dir=os.path.join(config.data_path, graphs_dir),
        labels_dir=os.path.join(config.data_path, labels_dir),
    )
    dataloader = DataLoader(dataset, batch_size=1)  # testing only allows for batch size of 1
    batch = next(iter(dataloader))

    model.test_step(batch, 0, split="test")

    assert model.test_outputs is not None
    assert len(model.test_outputs) == 1
    metrics = model.test_outputs[0]
    assert metrics is not None
    assert len(metrics) == 1
