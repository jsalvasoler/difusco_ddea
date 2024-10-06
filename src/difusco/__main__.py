import sys

from difusco_edward_sun.difusco.train import arg_parser
from difusco_edward_sun.difusco.train import main as difusco_main

from difusco.tsp.generate_tsp_data import main_tsp_data_generation

AVAILABLE_COMMANDS = ["difusco", "generate_tsp_data"]


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Please specify a command. Available commands: {', '.join(AVAILABLE_COMMANDS)}")
        return

    command = sys.argv[1]

    match command:
        case "difusco":
            args = arg_parser()
            difusco_main(args)
        case "generate_tsp_data":
            main_tsp_data_generation()
        case _:
            print(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
