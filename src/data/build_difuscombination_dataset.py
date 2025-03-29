from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from config.mytable import TableSaver
from problems.mis.mis_dataset import MISDataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build the difuscombination dataset")

    parser.add_argument("--which", type=str, default="er_700_800", help="Dataset identifier (e.g., 'er_700_800')")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "val"],
        help="Dataset split (train, test, or val)",
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"),
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--n_select", type=int, default=20, help="Number of random examples to select for each instance"
    )
    parser.add_argument(
        "--timestamp", type=str, default=None, help="Optional timestamp for output file (default: current time)"
    )

    return parser.parse_args()


def merge_table_files(source_dir: str) -> pd.DataFrame:
    """Merge all table_*.csv files in the source directory."""
    # Get all files starting with table_ and ending with .csv
    files = [f for f in os.listdir(source_dir) if f.startswith("table_") and f.endswith(".csv")]

    df_list = []
    for f in tqdm(files, desc="Merging tables"):
        print(f)
        df_x = pd.read_csv(os.path.join(source_dir, f))
        print(f"before filtering: {len(df_x)}")

        # You can uncomment these lines if needed
        # # filter second sequence
        # df_x = filter_second_sequence(df_x.copy())
        # # offset instance_id by 10_000
        # df_x["instance_id"] = df_x["instance_id"] + 10_000

        df_list.append(df_x.copy())
        print(f"after filtering: {len(df_x)}")

    df = pd.concat(df_list)

    print(f"total examples: {len(df)}")

    # Get unique instance_id
    unique_instance_ids = df["instance_id"].unique()
    # Remove nan
    unique_instance_ids = unique_instance_ids[~np.isnan(unique_instance_ids)]
    print(f"unique instance ids: {len(unique_instance_ids)}")

    return df


def process_merged_table(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Process the merged table: sort parents, remove duplicates, and filter rows."""
    # Write merged table to file
    df.to_csv(os.path.join(output_dir, "all_tables.csv"), index=False)

    # Read it back (this step might not be strictly necessary but matches the notebook)
    df = pd.read_csv(os.path.join(output_dir, "all_tables.csv"), nrows=None)
    print(f"Loaded all_tables.csv: {len(df)} rows")

    # Take unique parent_1, parent_2, children, instance_id
    df = df[["parent_1", "parent_2", "children", "instance_id"]].drop_duplicates()

    # Sort parent_1 and parent_2 by length and alphabetically
    df["parent_1_len"] = df["parent_1"].str.count(",")
    df["parent_2_len"] = df["parent_2"].str.count(",")

    df["parent_1_sorted"] = np.where(df["parent_1_len"] > df["parent_2_len"], df["parent_2"], df["parent_1"])
    df["parent_2_sorted"] = np.where(df["parent_1_len"] > df["parent_2_len"], df["parent_1"], df["parent_2"])

    # Remove duplicates again
    df = df[["parent_1_sorted", "parent_2_sorted", "children", "instance_id"]].drop_duplicates()

    # Remove cases in which parent_1_sorted = children, or parent_2_sorted = children
    df = df[df["parent_1_sorted"] != df["children"]]
    df = df[df["parent_2_sorted"] != df["children"]]
    print(f"After deduplication and filtering: {len(df)} rows")

    # Save to all_tables_unique.csv
    df.to_csv(os.path.join(output_dir, "all_tables_unique.csv"), index=False)

    return df


def select_examples(df: pd.DataFrame, output_dir: str, n_select: int) -> pd.DataFrame:
    """Select a subset of examples for each instance ID."""
    # Check length of unique dataframe
    print(f"Processing unique table with {len(df)} rows")
    print(f"Unique instance_ids: {len(df['instance_id'].unique())}")
    print(f"Min instance_id: {df['instance_id'].min()}")
    print(f"Max instance_id: {df['instance_id'].max()}")

    # Add fitness column
    df["children_fitness"] = df["children"].str.count(",")

    # Select n_select random examples for each instance
    df_selected = (
        df.groupby("instance_id")
        .apply(lambda x: x.sample(n=min(n_select, len(x)), replace=False))
        .reset_index(drop=True)
    )

    print(f"Selected {len(df_selected)} examples")
    df_selected.to_csv(os.path.join(output_dir, "to_become_dataset.csv"), index=False)

    return df_selected


def build_dataset(
    df: pd.DataFrame, output_dir: str, source_dir: str, which: str, split: str, timestamp: str | None = None
) -> None:
    """Build the final dataset CSV and corresponding label files."""
    # Initialize MISDataset
    mis_dataset = MISDataset(
        data_dir=f"{source_dir}/mis/{which}/{split}", data_label_dir=f"{source_dir}/mis/{which}/{split}_labels"
    )

    # Set up output paths
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = f"{output_dir}/difuscombination/mis/{which}/{split}"
    table_name = f"{output_dir}/difuscombination_samples_{timestamp}.csv"
    os.makedirs(output_dir, exist_ok=True)

    table_saver = TableSaver(table_name)

    # Create labels directory
    labels_dir = f"{output_dir}/difuscombination/mis/{which}/{split}_labels"
    os.makedirs(labels_dir, exist_ok=True)

    # Process each sample
    for sample in tqdm(mis_dataset, desc="Building dataset"):
        file_name = mis_dataset.get_file_name_from_sample_idx(sample[0])
        idx = sample[0].item()

        # Find solutions for this instance
        instance_sols = df[df["instance_id"] == idx + 10_000]
        if len(instance_sols) == 0:
            continue

        solution_str = ""
        k = 0
        for _, row in instance_sols.iterrows():
            parent_1 = " ".join(row["parent_1_sorted"].split(","))
            parent_2 = " ".join(row["parent_2_sorted"].split(","))
            solution_str += f" | {parent_1} | {parent_2}"

            if split == "test" and k >= 1:
                continue

            label_file_name = f"{file_name}___{2 * k}_{2 * k + 1}.txt"
            children_np = np.array(list(map(int, row["children"].split(","))))
            children_np_labels = np.zeros(sample[1].x.shape[0], dtype=np.int64)
            children_np_labels[children_np] = 1
            assert (
                0 <= children_np.min() <= children_np.max() <= sample[1].x.shape[0] - 1
            ), f"children_np: {children_np}"

            with open(os.path.join(labels_dir, label_file_name), "w") as f:
                np.savetxt(f, children_np_labels, fmt="%d")

            k += 1

        # Using lstrip to remove leading characters that are part of a multi-character string can be misleading.
        # Instead, we can use lstrip(" |") to remove the leading space and pipe character.
        solution_str = solution_str.lstrip(" |").lstrip()

        row = {
            "instance_file_name": file_name,
            "solution_str": solution_str,
        }
        table_saver.put(row)


def main() -> None:
    """Main function to build the difuscombination dataset."""
    args = parse_args()

    # Set up paths
    source_dir = os.path.join(args.raw_data_dir, f"difuscombination/{args.which}/{args.split}")
    output_dir = os.path.join(args.data_dir, f"difuscombination/mis/{args.which}/{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Merge table files
    print(f"Merging table files from {source_dir}")
    df = merge_table_files(source_dir)

    # Step 2: Process merged table
    print("Processing merged table")
    df = process_merged_table(df, output_dir)

    # Step 3: Select examples
    print(f"Selecting up to {args.n_select} examples per instance")
    df = select_examples(df, output_dir, args.n_select)

    # Step 4: Build dataset
    print("Building final dataset")
    build_dataset(
        df=df,
        output_dir=output_dir,
        source_dir=source_dir,
        which=args.which,
        split=args.split,
        timestamp=args.timestamp,
    )

    print(f"Dataset successfully built and saved to {output_dir}")


if __name__ == "__main__":
    main()
