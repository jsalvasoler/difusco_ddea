from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run an evolutionary algorithm")

    general = parser.add_argument_group("general")
    general.add_argument("--task", type=str, required=True)
    general.add_argument("--data_path", type=str, required=True)
    general.add_argument("--logs_path", type=str, default=None)
    general.add_argument("--results_path", type=str, default=None)
    general.add_argument("--test_split", type=str, required=True)
    general.add_argument("--test_split_label_dir", type=str, default=None)

    wandb = parser.add_argument_group("wandb")
    wandb.add_argument("--project_name", type=str, default="difusco")
    wandb.add_argument("--wandb_entity", type=str, default=None)
    wandb.add_argument("--wandb_logger_name", type=str, default=None)

    ea_settings = parser.add_argument_group("ea_settings")
    ea_settings.add_argument("--device", type=str, default="cpu")
    ea_settings.add_argument("--pop_size", type=int, default=100)
    ea_settings.add_argument("--n_generations", type=int, default=100)
    ea_settings.add_argument("--max_two_opt_it", type=int, default=-1)
    ea_settings.add_argument("--initialization", type=str, default="random_feasible")
    ea_settings.add_argument("--recombination", type=str, default="classic")
    ea_settings.add_argument("--config_name", type=str, default=None)
    ea_settings.add_argument("--save_results", type=lambda x: x.lower() in ["true", "1", "yes", "y"], default=False)

    difusco_settings = parser.add_argument_group("difusco_settings")
    difusco_settings.add_argument("--models_path", type=str, default=".")
    difusco_settings.add_argument("--ckpt_path", type=str, default=None)
    difusco_settings.add_argument("--ckpt_path_difuscombination", type=str, default=None)
    difusco_settings.add_argument("--diffusion_type", type=str, default="categorical")
    difusco_settings.add_argument("--diffusion_schedule", type=str, default="linear")
    difusco_settings.add_argument("--inference_schedule", type=str, default="cosine")
    difusco_settings.add_argument("--diffusion_steps", type=int, default=1000)
    difusco_settings.add_argument("--inference_diffusion_steps", type=int, default=1000)
    difusco_settings.add_argument("--parallel_sampling", type=int, default=1)
    difusco_settings.add_argument("--sequential_sampling", type=int, default=1)
    difusco_settings.add_argument("--hidden_dim", type=int, default=256)
    difusco_settings.add_argument("--aggregation", type=str, default="sum")
    difusco_settings.add_argument("--n_layers", type=int, default=12)
    difusco_settings.add_argument("--use_activation_checkpoint", action="store_true")
    difusco_settings.add_argument("--fp16", action="store_true")
    difusco_settings.add_argument("--training_split", type=str, default=None)
    difusco_settings.add_argument("--validation_split", type=str, default=None)

    tsp_settings = parser.add_argument_group("tsp_settings")
    tsp_settings.add_argument("--sparse_factor", type=int, default=-1)

    mis_settings = parser.add_argument_group("mis_settings")
    mis_settings.add_argument("--tournament_size", type=int, default=2)
    mis_settings.add_argument("--deselect_prob", type=float, default=0.05)
    mis_settings.add_argument("--mutation_prob", type=float, default=0.25)
    mis_settings.add_argument("--opt_recomb_time_limit", type=int, default=15)
    mis_settings.add_argument("--save_results", type=lambda x: x.lower() in ["true", "1", "yes", "y"], default=False)
    mis_settings.add_argument(
        "--preserve_optimal_recombination", type=lambda x: x.lower() in ["true", "1", "yes", "y"], default=False
    )

    dev = parser.add_argument_group("dev")
    dev.add_argument("--profiler", action="store_true")
    dev.add_argument("--validate_samples", type=int, default=None)

    return parser


def validate_args(args: Namespace) -> None:
    assert args.task in ["tsp", "mis"]

    assert args.pop_size > 2, "Population size must be greater than 2."
    assert args.initialization in ["random_feasible", "difusco_sampling"]
    assert args.recombination in ["classic", "difuscombination", "optimal"]

    if args.task == "mis":
        assert args.recombination in [
            "classic",
            "optimal",
            "difuscombination",
        ], "Choose a valid recombination method for mis."
        assert args.initialization in [
            "random_feasible",
            "difusco_sampling",
        ], "Choose a valid initialization method for mis."
        assert args.tournament_size > 0, "Tournament size must be greater than 0 for mis."
        assert args.deselect_prob > 0, "Deselect probability must be greater than 0 for mis."

    if args.task == "tsp":
        assert args.max_two_opt_it > 0, "max_two_opt_it must be greater than 0 for tsp."

    for dir_path in [args.data_path, args.logs_path]:
        if dir_path:
            assert os.path.exists(dir_path), f"Path {dir_path} does not exist."

    for split in ["test_split"]:
        if not getattr(args, split):
            continue
        full_path = os.path.join(args.data_path, getattr(args, split))
        assert os.path.exists(full_path), f"Path {getattr(args, split)} does not exist."

    assert args.project_name == "difusco", "Project name must be of the form difusco."
    assert args.config_name is not None, "Config name must be provided."

    # Validate wandb logger name. Format example: tsp_diffusion_graph_categorical_tsp50_test
    if args.wandb_logger_name:
        assert args.wandb_logger_name.startswith(f"{args.task}_ea_"), "Wandb logger name must start with task_ea_"


def parse_args() -> tuple[Namespace, list[str]]:
    parser = get_arg_parser()
    args, extra = parser.parse_known_args()
    validate_args(args)
    return args, extra
