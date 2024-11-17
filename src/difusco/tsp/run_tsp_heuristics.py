import os
import time
from argparse import Namespace
from datetime import datetime

import numpy as np
import pandas as pd
from problems.tsp.tsp_evaluation import TSPEvaluator, batched_two_opt_torch, merge_tours
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from tqdm import tqdm

from difusco.arg_parser import parse_args


def run_tsp_heuristics_main() -> None:
    args = parse_args()
    assert args.strategy in ["construction", "construction+2opt"]

    print(f"Running heuristic evaluation with strategy: {args.strategy}")

    # load the dataset
    dataset = TSPGraphDataset(
        data_file=os.path.join(args.data_path, args.test_split),
        sparse_factor=args.sparse_factor,
    )

    sample = dataset.__getitem__(0)
    _, points, _, _ = sample
    n = points.shape[0]
    heatmap = np.ones((n, n))

    # stack as many heatmaps as args.parallel_sampling
    heatmaps = np.stack([heatmap] * args.parallel_sampling, axis=0)

    # add noise to the heatmaps except first one
    for i in range(1, args.parallel_sampling):
        heatmaps[i] -= np.random.normal(0, 1, size=(n, n))

    scores = []
    gt_scores = []

    start_time = time.time()
    # Run the construction heuristic for all graphs in the dataset
    for sample in tqdm(dataset):
        idx, points, adj_matrix, gt_tour = sample
        points = points.numpy()

        # 1. Construction heuristic
        tours, _ = merge_tours(heatmaps, points, None, sparse_graph=False, parallel_sampling=args.parallel_sampling)

        evaluator = TSPEvaluator(points)

        best_score = float("inf")
        for i in range(args.parallel_sampling):
            tour = tours[i]
            best_score = min(best_score, evaluator.evaluate(tour))

        if args.strategy == "construction+2opt":
            # 2. 2-opt heuristic
            tours, _ = batched_two_opt_torch(
                points.astype("float64"),
                np.array(tours).astype("int64"),
                max_iterations=args.two_opt_iterations,
                device="cpu",
            )

            best_score = float("inf")
            for i in range(args.parallel_sampling):
                tour = tours[i]
                best_score = min(best_score, evaluator.evaluate(tour))

        scores.append(best_score)
        gt_scores.append(evaluator.evaluate(gt_tour.numpy()))

    print(f"\nAverage score: {np.mean(scores)}")
    print(f"Average ground truth score: {np.mean(gt_scores)}")
    print(f"Relative improvement: {np.mean(scores) / np.mean(gt_scores) - 1}")

    results = {
        "N": n,
        "parallel_sampling": args.parallel_sampling,
        "avg_length": np.mean(scores),
        "avg_gt_length": np.mean(gt_scores),
        "relative_improvement": np.mean(scores) / np.mean(gt_scores) - 1,
        "n_samples": len(scores),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "strategy": args.strategy,
        "runtime": time.time() - start_time,
    }
    write_results(args, results)


def write_results(args: Namespace, results: dict) -> None:
    required_keys = [
        "N",
        "parallel_sampling",
        "avg_length",
        "avg_gt_length",
        "relative_improvement",
        "n_samples",
        "timestamp",
        "strategy",
        "runtime",
    ]
    assert all(key in results for key in required_keys), f"Results dictionary must contain keys {required_keys}."

    filename = "heuristic_results_tsp.csv"
    filepath = os.path.join(args.results_path, filename)

    # check if csv with filename
    df = pd.read_csv(filepath) if os.path.exists(filepath) else pd.DataFrame(columns=required_keys)

    df.loc[len(df)] = [results[key] for key in required_keys]
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    run_tsp_heuristics_main()
