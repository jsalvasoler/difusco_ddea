from __future__ import annotations

import argparse
import pprint as pp
import time
import warnings
from multiprocessing import Pool

import lkh
import numpy as np
import tqdm
from tsplib95.models import StandardProblem

# from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde

warnings.filterwarnings("ignore")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--min_nodes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=128000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--solver", type=str, default="lkh")
    parser.add_argument("--lkh_path", type=str, default=None)
    parser.add_argument("--lkh_trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    opts, _ = parser.parse_known_args()

    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"
    assert opts.solver in ["concorde", "lkh"], "Unknown solver"
    assert opts.solver != "lkh" or opts.lkh_path is not None, "LKH path must be provided"

    return opts


def solve_tsp(nodes_coord: list, opts: argparse.Namespace) -> list:
    if opts.solver == "concorde":
        msg = "Concorde solver is currently unsupported."
        raise ValueError(msg)
        # scale = 1e6
        # solver = TSPSolver.from_data(nodes_coord[:, 0] * scale, nodes_coord[:, 1] * scale, norm="EUC_2D")
        # solution = solver.solve(verbose=False)
        # tour = solution.tour
    if opts.solver == "lkh":
        scale = 1e6
        problem = StandardProblem(
            name="TSP",
            type="TSP",
            dimension=len(nodes_coord),
            edge_weight_type="EUC_2D",
            node_coords={n + 1: nodes_coord[n] * scale for n in range(len(nodes_coord))},
        )

        solution = lkh.solve(opts.lkh_path, problem=problem, max_trials=opts.lkh_trials, runs=10)
        return [n - 1 for n in solution[0]]

    error_message = f"Unknown solver: {opts.solver}"
    raise ValueError(error_message)


def generate_tsp_data(opts: argparse.Namespace) -> None:
    np.random.seed(opts.seed)

    if opts.filename is None:
        opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_concorde.txt"

    # Pretty print the run args
    pp.pprint(vars(opts))

    start_time = time.time()
    with open(opts.filename, "w") as f:
        for _ in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
            num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
            assert opts.min_nodes <= num_nodes <= opts.max_nodes

            batch_nodes_coord = np.random.random([opts.batch_size, num_nodes, 2])

            with Pool(opts.batch_size) as p:
                tours = p.map(solve_tsp_wrapper, [(batch_nodes_coord[idx], opts) for idx in range(opts.batch_size)])

            for idx, tour in enumerate(tours):
                if (np.sort(tour) == np.arange(num_nodes)).all():
                    f.write(" ".join(str(x) + " " + str(y) for x, y in batch_nodes_coord[idx]))
                    f.write(" " + "output" + " ")
                    f.write(" ".join(str(node_idx + 1) for node_idx in tour))
                    f.write(" " + str(tour[0] + 1) + " ")
                    f.write("\n")

        end_time = time.time() - start_time

    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / opts.num_samples:.1f}s")


def solve_tsp_wrapper(args: tuple[list, argparse.Namespace]) -> None:
    nodes_coord, opts = args
    return solve_tsp(nodes_coord, opts)


if __name__ == "__main__":
    opts = parse_arguments()
    generate_tsp_data(opts)
