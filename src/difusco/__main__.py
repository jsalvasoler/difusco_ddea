import sys

from difusco.difusco import main_difusco
from difusco.tsp.generate_tsp_data import main_tsp_data_generation
from difusco.tsp.run_tsp_heuristics import run_tsp_heuristics

AVAILABLE_COMMANDS = ["difusco", "generate_tsp_data"]


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
    else:
        print(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
