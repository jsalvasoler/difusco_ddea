import os
import timeit

import numpy as np
import wandb
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_ea import create_mis_ea, create_mis_instance
from problems.tsp.tsp_ea import create_tsp_ea, create_tsp_instance
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from pyinstrument import Profiler
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from ea.arg_parser import parse_args, validate_args
from ea.config import Config
from ea.ea_utils import save_results
from ea.problem_instance import ProblemInstance


def ea_factory(config: Config, instance: ProblemInstance) -> GeneticAlgorithm:
    if config.task == "mis":
        return create_mis_ea(instance, config)
    if config.task == "tsp":
        return create_tsp_ea(instance, config)
    error_msg = f"No evolutionary algorithm for task {config.task}."
    raise ValueError(error_msg)


def instance_factory(config: Config, sample: tuple) -> ProblemInstance:
    if config.task == "mis":
        return create_mis_instance(sample, device=config.device)
    if config.task == "tsp":
        return create_tsp_instance(sample, device=config.device, sparse_factor=config.sparse_factor)
    error_msg = f"No instance for task {config.task}."
    raise ValueError(error_msg)


def dataset_factory(config: Config) -> Dataset:
    data_path = os.path.join(config.data_path, config.test_split)
    data_label_dir = (
        os.path.join(config.data_path, config.test_split_label_dir) if config.test_split_label_dir else None
    )

    if config.task == "mis":
        return MISDataset(data_dir=data_path, data_label_dir=data_label_dir)

    if config.task == "tsp":
        return TSPGraphDataset(data_file=data_path)

    error_msg = f"No dataset for task {config.task}."
    raise ValueError(error_msg)


def main_ea() -> None:
    args = parse_args()
    validate_args(args)

    config = Config.load_from_args(args)

    if args.profiler:
        with Profiler() as profiler:
            run_ea(config)
        print(profiler.output_text(unicode=True, color=True))
    else:
        run_ea(config)


def run_ea(config: Config) -> None:
    dataset = dataset_factory(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    wandb.init(
        project=config.project_name,
        name=config.wandb_logger_name,
        entity=config.wandb_entity,
        config=config.__dict__,
        dir=config.logs_path,
    )

    results = []

    is_validation_run = config.validate_samples is not None
    for i, sample in enumerate(dataloader):
        instance = instance_factory(config, sample)
        ea = ea_factory(config, instance)

        _ = StdOutLogger(searcher=ea, interval=10)

        start_time = timeit.default_timer()
        ea.run(config.n_generations)

        cost = ea.status["pop_best_eval"]
        gt_cost = instance.gt_cost
        gap = (gt_cost - cost) / gt_cost

        run_results = {"cost": cost, "gt_cost": gt_cost, "gap": gap, "runtime": timeit.default_timer() - start_time}
        wandb.log(run_results, step=i)
        results.append(run_results)

        if is_validation_run and i == config.validate_samples - 1:
            break

    agg_results = {
        "avg_cost": np.mean([r["cost"] for r in results]),
        "avg_gt_cost": np.mean([r["gt_cost"] for r in results]),
        "avg_gap": np.mean([r["gap"] for r in results]),
        "avg_runtime": np.mean([r["runtime"] for r in results]),
    }
    wandb.log(agg_results)

    agg_results["wandb_id"] = wandb.run.id

    # save results to results_path if it is not a validation run
    if not is_validation_run:
        save_results(config, agg_results)

    wandb.finish()
