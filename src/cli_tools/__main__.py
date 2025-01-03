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

    args = parse_args()
    main_ea(args)


def run_difusco_initialization_experiments() -> None:
    """Run Difusco initialization experiments."""

    from difusco.difusco_initialization_experiments import parse_arguments
    from difusco.difusco_initialization_experiments import run_difusco_initialization_experiments as run_dif_init_main

    args, extra = parse_arguments()
    config = Config.load_from_args(args, extra)
    run_dif_init_main(config)


def main() -> None:
    # Top-level commands and usage help
    commands = {
        "difusco": {
            "run-difusco": run_difusco,
            "generate-tsp-data": generate_tsp_data,
            "generate-node-degree-labels": generate_node_degree_labels,
            "run-tsp-heuristics": run_tsp_heuristics,
            "run-difusco-initialization-experiments": run_difusco_initialization_experiments,
        },
        "ea": {
            "run-ea": run_ea,
        },
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage:")
        print(" - hatch run cli difusco <subcommand>")
        print(" - hatch run cli ea <subcommand>")
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
