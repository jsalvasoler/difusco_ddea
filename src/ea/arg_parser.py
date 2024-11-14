import os
from argparse import ArgumentParser, Namespace


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run an evolutionary algorithm")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--logs_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--test_split", type=str, required=True)
    parser.add_argument("--test_split_label_dir", type=str, default=None)

    parser.add_argument("--project_name", type=str, default="difusco")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_logger_name", type=str, default=None)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_parallel_evals", type=int, default=0)

    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--n_generations", type=int, default=100)

    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--validate_samples", type=int, default=None)

    return parser


def validate_args(args: Namespace) -> None:
    assert args.task in ["tsp", "mis", "high_degree_selection"]

    for dir_path in [args.data_path, args.logs_path]:
        if dir_path:
            assert os.path.exists(dir_path), f"Path {dir_path} does not exist."

    for split in ["test_split"]:
        if not getattr(args, split):
            continue
        full_path = os.path.join(args.data_path, getattr(args, split))
        assert os.path.exists(full_path), f"Path {getattr(args, split)} does not exist."

    assert args.project_name == "difusco", "Project name must be of the form difusco."

    # Validate wandb logger name. Format example: tsp_diffusion_graph_categorical_tsp50_test
    if args.wandb_logger_name:
        assert args.wandb_logger_name.startswith(f"{args.task}_ea_"), "Wandb logger name must start with task_ea_"


def parse_args() -> Namespace:
    parser = get_arg_parser()
    args, _ = parser.parse_known_args()
    validate_args(args)
    return args
