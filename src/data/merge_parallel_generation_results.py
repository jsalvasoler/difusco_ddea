import argparse
import os
import pandas as pd
from datetime import datetime

def get_id_from_filename(filename: str) -> int | None:
    try:
        return int(filename.split("_")[-1].split(".")[0])
    except ValueError:
        return None


def merge_parallel_exec_files() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num_processes", type=int, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    args = arg_parser.parse_args()

    
    # assert path exists
    assert os.path.exists(args.output_path), f"Output path {args.output_path} does not exist"

    files = os.listdir(args.output_path)
    # filter out non-csv files
    files = [f for f in files if f.endswith(".csv")]
    # filter out if does not start with difuscombination_samples
    files = [f for f in files if f.startswith("difuscombination_samples")]

    # filter out those that do not end with _{some_valid_index}.csv
    files = [f for f in files if get_id_from_filename(f) in set(range(args.num_processes))]

    assert len(files) == args.num_processes, f"Expected {args.num_processes} files, got {len(files)}"


    # we will sort the files by the index
    files = sorted(files, key=lambda x: get_id_from_filename(x))

    df_list = []
    for i in range(args.num_processes):
        df_list.append(pd.read_csv(os.path.join(args.output_path, files[i])))
        assert i == get_id_from_filename(files[i]), f"Expected index {i}, got {get_id_from_filename(files[i])}"

    df = pd.concat(df_list)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(os.path.join(args.output_path, f"difuscombination_samples_{timestamp}.csv"), index=False)


if __name__ == "__main__":
    merge_parallel_exec_files()
