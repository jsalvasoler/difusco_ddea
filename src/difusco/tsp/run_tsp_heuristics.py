import os
import time
from argparse import Namespace
from datetime import datetime

import numpy as np
import pandas as pd
from difusco_edward_sun.difusco.co_datasets.tsp_graph_dataset import TSPGraphDataset

from difusco.arg_parser import parse_args
from difusco.tsp.utils import TSPEvaluator, merge_tours


def run_tsp_heuristics() -> None:
    args = parse_args()

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

    scores = []
    gt_scores = []

    start_time = time.time()
    # Run the construction heuristic for all graphs in the dataset
    for sample in dataset:
        idx, points, adj_matrix, gt_tour = sample
        points = points.numpy()

        tours, _ = merge_tours(heatmaps, points, None, sparse_graph=False, parallel_sampling=args.parallel_sampling)

        print(f"{len(tours)} tours for graph {idx.item()}")

        evaluator = TSPEvaluator(points)
        best_score = float("inf")
        for i in range(args.parallel_sampling):
            tour = tours[i]
            print(f"Tour {i}: {tour}")
            print(f"Tour length: {evaluator.evaluate(tour)}")
            best_score = min(best_score, evaluator.evaluate(tour))
            print(f"Ground truth tour: {evaluator.evaluate(gt_tour.numpy())}")

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
        "strategy": "construction",
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
    run_tsp_heuristics()
