import os
from argparse import Namespace

import numpy as np
import wandb
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from pyinstrument import Profiler
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from ea.config import Config

from difusco.mis.mis_dataset import MISDataset
from difusco.tsp.tsp_graph_dataset import TSPGraphDataset
from ea.arg_parser import parse_args, validate_args
from ea.mis import MISInstance, create_mis_ea, create_mis_instance
from ea.tsp import TSPInstance

ProblemInstance = MISInstance | TSPInstance


def ea_factory(config: Config, instance: ProblemInstance) -> GeneticAlgorithm:
    if args.task == "mis":
        return create_mis_ea(instance, args)
    error_msg = f"No evolutionary algorithm for task {args.task}."
    raise ValueError(error_msg)


def instance_factory(task: str, sample: tuple) -> ProblemInstance:
    if task == "mis":
        return create_mis_instance(sample)
    error_msg = f"No instance for task {task}."
    raise ValueError(error_msg)


def dataset_factory(config: Config) -> Dataset:
    data_path = os.path.join(args.data_path, args.test_split)
    data_label_dir = os.path.join(args.data_path, args.test_split_label_dir) if args.test_split_label_dir else None

    if args.task == "mis":
        return MISDataset(data_dir=data_path, data_label_dir=data_label_dir)

    if args.task == "tsp":
        return TSPGraphDataset(data_dir=data_path)

    error_msg = f"No dataset for task {args.task}."
    raise ValueError(error_msg)


def main_ea() -> None:
    args = parse_args()
    validate_args(args)

    if args.profiler:
        with Profiler() as profiler:
            run_ea(args)
        print(profiler.output_text(unicode=True, color=True))
    else:
        run_ea(args)


def run_ea(config: Config) -> None:
    dataset = dataset_factory(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    wandb.init(
        project=args.project_name,
        name=args.wandb_logger_name,
        entity=args.wandb_entity,
        config=args.__dict__,
        dir=args.logs_path,
    )

    results = []

    for i, sample in enumerate(dataloader):
        instance = instance_factory(args.task, sample)
        ea = ea_factory(args, instance)

        _ = StdOutLogger(
            searcher=ea,
            interval=10,
        )

        ea.run(args.n_generations)

        cost = ea.status["pop_best_eval"]
        gt_cost = instance.evaluate_mis_individual(instance.gt_labels)
        gap = (gt_cost - cost) / gt_cost

        run_results = {"cost": cost, "gt_cost": gt_cost, "gap": gap}
        wandb.log(run_results, step=i)
        results.append(run_results)

        if args.validate_samples is not None and i == args.validate_samples - 1:
            break

    agg_results = {
        "avg_cost": np.mean([r["cost"] for r in results]),
        "avg_gt_cost": np.mean([r["gt_cost"] for r in results]),
        "avg_gap": np.mean([r["gap"] for r in results]),
    }
    wandb.log(agg_results)

    wandb.finish()
