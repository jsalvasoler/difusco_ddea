from __future__ import annotations

import sys

from config.myconfig import Config


def run_difusco() -> None:
    """Run the Difusco main command."""
    from difusco.arg_parser import parse_args
    from difusco.difusco_main import main_difusco

    args = parse_args()
    main_difusco(args)


def generate_tsp_data() -> None:
    """Generate TSP data."""
    from difusco.tsp.generate_tsp_data import generate_tsp_data, parse_arguments

    opts = parse_arguments()
    generate_tsp_data(opts)


def generate_node_degree_labels() -> None:
    """Generate MIS degree labels."""
    from difusco.mis.generate_node_degree_labels import generate_node_degree_labels, parse_arguments

    opts = parse_arguments()
    generate_node_degree_labels(opts)


def run_tsp_heuristics() -> None:
    """Run TSP heuristics."""
    from difusco.tsp.run_tsp_heuristics import parse_args, run_tsp_heuristics_main

    args = parse_args()
    run_tsp_heuristics_main(args)


def run_ea() -> None:
    """Run the Evolutionary Algorithm."""
    from ea.ea_arg_parser import parse_args
    from ea.evolutionary_algorithm import main_ea

    args, extra = parse_args()
    config = Config.load_from_args(args, extra)
    main_ea(config)


def run_difusco_initialization_experiments() -> None:
    """Run Difusco initialization experiments."""

    from difusco.difusco_initialization_experiments import main_init_experiments, parse_arguments

    args, extra = parse_arguments()
    config = Config.load_from_args(args, extra)
    main_init_experiments(config)


def generate_difuscombination_samples() -> None:
    """Generate training samples for difuscombination."""
    from ea.generate_difuscombination_samples import parse_arguments, run_training_data_generation

    args, extra = parse_arguments()
    config = Config.load_from_args(args, extra)
    run_training_data_generation(config)


def solve_difuscombination() -> None:
    from data.solve_difuscombination import get_arg_parser, solve_difuscombination

    args, _ = get_arg_parser().parse_known_args()
    config = Config.load_from_namespace(args)
    solve_difuscombination(config)


def main() -> None:
    # Top-level commands and usage help
    commands = {
        "difusco": {
            "run-difusco": run_difusco,
            "generate-tsp-data": generate_tsp_data,
            "generate-node-degree-labels": generate_node_degree_labels,
            "run-tsp-heuristics": run_tsp_heuristics,
            "run-difusco-initialization-experiments": run_difusco_initialization_experiments,
            "generate-difuscombination-samples": generate_difuscombination_samples,
        },
        "ea": {
            "run-ea": run_ea,
        },
        "data": {
            "solve-difuscombination": solve_difuscombination,
        },
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage:")
        print(" - hatch run cli difusco <subcommand>")
        print(" - hatch run cli ea <subcommand>")
        print(" - hatch run cli data <subcommand>")
        print("\nCommands:")
        for group, subcommands in commands.items():
            print(f" - {group}:")
            for subcommand in subcommands:
                print(f"     {subcommand}")
        print("\nFor command-specific info, run 'hatch run cli <group> <subcommand>' --help")
        sys.exit(1)

    group = sys.argv[1]
    subcommands = commands[group]

    if len(sys.argv) < 3 or sys.argv[2] not in subcommands:
        print(f"Usage for '{group}':")
        for subcommand in subcommands:
            print(f"  {group} {subcommand}")
        sys.exit(1)

    subcommand = sys.argv[2]

    # Call the appropriate function
    subcommands[subcommand]()


if __name__ == "__main__":
    main()
