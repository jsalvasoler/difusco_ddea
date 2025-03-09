import os
from argparse import ArgumentParser

from config.myconfig import Config


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train a Pytorch-Lightning diffusion-base recombination model for a COP dataset."
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--models_path", type=str, default=None)
    parser.add_argument("--logs_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)

    parser.add_argument("--training_samples_file", type=str, required=True)
    parser.add_argument("--training_graphs_dir", type=str, required=True)
    parser.add_argument("--training_labels_dir", type=str, required=True)
    parser.add_argument("--test_samples_file", type=str, required=True)
    parser.add_argument("--test_graphs_dir", type=str, required=True)
    parser.add_argument("--test_labels_dir", type=str, required=True)
    parser.add_argument("--validation_samples_file", type=str, required=True)
    parser.add_argument("--validation_graphs_dir", type=str, required=True)
    parser.add_argument("--validation_labels_dir", type=str, required=True)

    parser.add_argument("--validation_examples", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")

    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_activation_checkpoint", action="store_true")

    parser.add_argument("--diffusion_type", type=str, default="categorical")
    parser.add_argument("--diffusion_schedule", type=str, default="linear")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_diffusion_steps", type=int, default=50)
    parser.add_argument("--inference_schedule", type=str, default="linear")
    parser.add_argument("--inference_trick", type=str, default="ddim")
    parser.add_argument("--sequential_sampling", type=int, default=1)
    parser.add_argument("--parallel_sampling", type=int, default=1)

    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sparse_factor", type=int, default=-1)
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--two_opt_iterations", type=int, default=1000)
    parser.add_argument("--save_numpy_heatmap", action="store_true")

    parser.add_argument("--project_name", type=str, default="difusco")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_logger_name", type=str, default=None)
    parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume_weight_only", action="store_true")

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_valid_only", action="store_true")
    parser.add_argument("--profiler", action="store_true")

    return parser


def validate_config(config: Config) -> None:
    assert config.task in ["tsp", "mis"]
    assert config.diffusion_type in ["gaussian", "categorical"]
    assert config.diffusion_schedule in ["linear", "cosine"]

    for dir_path in [config.data_path, config.models_path, config.logs_path]:
        if dir_path:
            assert os.path.exists(dir_path), f"Path {dir_path} does not exist."

    if config.do_test or config.do_train or config.do_valid_only:
        assert all(x for x in [config.data_path, config.models_path, config.logs_path])

    assert isinstance(config.parallel_sampling, int), "parallel_sampling must be an integer"
    assert config.parallel_sampling >= 0, "parallel_sampling must be greater than or equal to 0"

    assert config.project_name == "difusco", "Project name must be of the form difusco."

    # Validate wandb logger name. Format example: tsp_diffusion_graph_categorical_tsp50_test
    if config.wandb_logger_name:
        assert config.wandb_logger_name.startswith(f"{config.task}_")

    if config.ckpt_path:
        assert os.path.exists(
            os.path.join(config.models_path, config.ckpt_path)
        ), f"Path {config.ckpt_path} does not exist."

        if "categorical" in config.ckpt_path:
            assert config.diffusion_type == "categorical", "diffusion_type must be categorical"
        elif "gaussian" in config.ckpt_path:
            assert config.diffusion_type == "gaussian", "diffusion_type must be gaussian"
