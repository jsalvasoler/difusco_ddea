import os
from argparse import ArgumentParser, Namespace


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train a Pytorch-Lightning diffusion model on a TSP dataset.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--models_path", type=str, default=None)
    parser.add_argument("--logs_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument(
        "--training_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for training split (used for MIS).",
    )
    parser.add_argument("--training_split", type=str, default=None)
    parser.add_argument("--validation_split", type=str, default=None)
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument("--validation_examples", type=int, default=64)

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
    parser.add_argument("--inference_diffusion_steps", type=int, default=1000)
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

    # Arguments for heuristic evaluation
    parser.add_argument("--heuristic_eval", action="store_true")
    parser.add_argument("--strategy", type=str, default=None)

    return parser


def validate_args(args: Namespace) -> None:
    assert args.task in ["tsp", "mis"]
    assert args.diffusion_type in ["gaussian", "categorical"]
    assert args.diffusion_schedule in ["linear", "cosine"]

    for dir_path in [args.data_path, args.models_path, args.logs_path]:
        if dir_path:
            assert os.path.exists(dir_path), f"Path {dir_path} does not exist."

    if args.do_test or args.do_train or args.do_valid_only:
        assert all(x for x in [args.data_path, args.models_path, args.logs_path])

    for split in ["training_split", "validation_split", "test_split"]:
        if not getattr(args, split):
            continue
        full_path = os.path.join(args.data_path, getattr(args, split))
        assert os.path.exists(full_path), f"Path {getattr(args, split)} does not exist."

    assert args.project_name == "difusco", "Project name must be of the form <task>_diffusion."

    # Validate wandb logger name. Format example: tsp_diffusion_graph_categorical_tsp50_test
    if args.wandb_logger_name:
        assert args.wandb_logger_name.startswith(f"{args.task}_diffusion_graph_{args.diffusion_type}_")

    if args.ckpt_path:
        assert os.path.exists(os.path.join(args.models_path, args.ckpt_path)), f"Path {args.ckpt_path} does not exist."

    # Heuristic evaluation
    if args.heuristic_eval:
        assert not any(
            x for x in [args.do_train, args.do_test, args.do_valid_only]
        ), "To run heuristic evaluation, set do_train, do_test, and do_valid_only to False."

        assert args.test_split, "Must provide test split for heuristic evaluation."
        assert not any(
            x for x in [args.train_split, args.validation_split]
        ), "Do not provide training or validation split for heuristic evaluation."

        assert args.results_path, "Must provide results path for heuristic evaluation."
        valid_strategies = ["construction", "construction+2opt"]
        assert args.strategy in valid_strategies, f"Heuristic evaluation strategy must be in {valid_strategies}."
        if args.strategy == "construction+2opt":
            assert args.two_opt_iterations > 0, "Number of 2-opt iterations must be greater than 0."


def parse_args() -> Namespace:
    parser = get_arg_parser()
    args, _ = parser.parse_known_args()
    validate_args(args)
    return args
