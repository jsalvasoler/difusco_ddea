import sys

from difusco.difusco import main_difusco
from difusco.tsp.generate_tsp_data import main_tsp_data_generation
from difusco.tsp.run_tsp_heuristics import run_tsp_heuristics
from difusco.node_selection.generate_node_degree_labels import generate_node_degree_labels_main

AVAILABLE_COMMANDS = ["difusco", "generate_tsp_data", "generate_mis_degree_labels"]


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Please specify a command. Available commands: {', '.join(AVAILABLE_COMMANDS)}")
        return

    command = sys.argv[1]

    if command == "difusco":
        main_difusco()
    elif command == "generate_tsp_data":
        main_tsp_data_generation()
    elif command == "run_tsp_heuristics":
        run_tsp_heuristics()
    elif command == "generate_mis_degree_labels":
        generate_node_degree_labels_main()
    else:
        print(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
