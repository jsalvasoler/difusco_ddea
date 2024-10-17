import sys

from difusco.train import main_difusco
from difusco.tsp.generate_tsp_data import main_tsp_data_generation

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
    else:
        print(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
