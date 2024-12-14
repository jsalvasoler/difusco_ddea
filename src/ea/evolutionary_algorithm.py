import os
import timeit
from argparse import Namespace
from gc import collect

import numpy as np
import torch
import wandb
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from problems.mis.mis_brkga import create_mis_brkga
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_ga import create_mis_ga
from problems.mis.mis_instance import create_mis_instance
from problems.tsp.tsp_brkga import create_tsp_brkga
from problems.tsp.tsp_ga import create_tsp_ga
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from problems.tsp.tsp_instance import create_tsp_instance
from pyinstrument import Profiler
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ea.config import Config
from ea.ea_utils import save_results
from ea.problem_instance import ProblemInstance


def ea_factory(config: Config, instance: ProblemInstance) -> GeneticAlgorithm:
    if config.algo == "brkga":
        if config.task == "mis":
            return create_mis_brkga(instance, config)
        if config.task == "tsp":
            return create_tsp_brkga(instance, config)
    elif config.algo == "ga":
        if config.task == "mis":
            return create_mis_ga(instance, config)
        if config.task == "tsp":
            return create_tsp_ga(instance, config)
    error_msg = f"No evolutionary algorithm for task {config.task}."
    raise ValueError(error_msg)


def instance_factory(config: Config, sample: tuple) -> ProblemInstance:
    if config.task == "mis":
        return create_mis_instance(sample, device=config.device, np_eval=config.np_eval)
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


def main_ea(args: Namespace) -> None:
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

    is_validation_run = config.validate_samples is not None
    if not is_validation_run:
        wandb.init(
            project=config.project_name,
            name=config.wandb_logger_name,
            entity=config.wandb_entity,
            config=config.__dict__,
            dir=config.logs_path,
        )

    results = []

    for i, sample in tqdm(enumerate(dataloader)):
        instance = instance_factory(config, sample)
        ea = ea_factory(config, instance)

        _ = StdOutLogger(searcher=ea, interval=10, after_first_step=True)

        start_time = timeit.default_timer()
        ea.run(config.n_generations)

        cost = ea.status["pop_best_eval"]
        gt_cost = instance.get_gt_cost()

        diff = cost - gt_cost if ea.problem.objective_sense == "min" else gt_cost - cost
        gap = diff / gt_cost

        run_results = {"cost": cost, "gt_cost": gt_cost, "gap": gap, "runtime": timeit.default_timer() - start_time}

        if not is_validation_run:
            wandb.log(run_results, step=i)

        results.append(run_results)

        # Clean up GPU memory
        del instance
        del ea
        torch.cuda.empty_cache()
        collect()

        if is_validation_run and i == config.validate_samples - 1:
            break

    agg_results = {
        "avg_cost": np.mean([r["cost"] for r in results]),
        "avg_gt_cost": np.mean([r["gt_cost"] for r in results]),
        "avg_gap": np.mean([r["gap"] for r in results]),
        "avg_runtime": np.mean([r["runtime"] for r in results]),
        "n_evals": len(results),
    }
    if not is_validation_run:
        wandb.log(agg_results)
        agg_results["wandb_id"] = wandb.run.id
        save_results(config, agg_results)
        wandb.finish()
