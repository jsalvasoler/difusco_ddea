import sys

from ea.evolutionary_algorithm import main_ea

AVAILABLE_COMMANDS = ["difusco", "generate_tsp_data", "generate_mis_degree_labels", "run_tsp_heuristics"]


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Please specify a command. Available commands: {', '.join(AVAILABLE_COMMANDS)}")
        return

    command = sys.argv[1]

    if command == "ea":
        main_ea()
    else:
        print(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
