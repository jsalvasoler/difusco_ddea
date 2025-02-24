from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from config.myconfig import Config
from config.mytable import TableSaver
from evotorch.logging import Logger
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_ga import MISGaProblem
from problems.mis.mis_instance import create_mis_instance
from problems.tsp.tsp_ga import TSPGAProblem
from problems.tsp.tsp_graph_dataset import TSPGraphDataset
from problems.tsp.tsp_instance import create_tsp_instance

from ea.ea_arg_parser import get_arg_parser

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from config.myconfig import Config
    from evotorch import Problem
    from torch.utils.data import Dataset

    from ea.problem_instance import ProblemInstance


def filter_args_by_group(parser: ArgumentParser, group_name: str) -> dict:
    group = next((g for g in parser._action_groups if g.title == group_name), None)  # noqa: SLF001
    if group is None:
        error_msg = f"Group '{group_name}' not found"
        raise ValueError(error_msg)

    return [a.dest for a in group._group_actions]  # noqa: SLF001


def get_results_dict(config: Config, results: dict) -> None:
    # filter the name of the arguments of the grup ea_settings
    parser = get_arg_parser()
    ea_setting_args = filter_args_by_group(parser, "ea_settings")
    tsp_setting_args = filter_args_by_group(parser, "tsp_settings")
    mis_setting_args = filter_args_by_group(parser, "mis_settings")
    difusco_setting_args = filter_args_by_group(parser, "difusco_settings")
    dev_setting_args = filter_args_by_group(parser, "dev")

    row = {
        "task": config.task,
        "wandb_logger_name": config.wandb_logger_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    row.update({k: getattr(config, k) for k in ea_setting_args})
    row.update({k: getattr(config, k) for k in tsp_setting_args})
    row.update({k: getattr(config, k) for k in mis_setting_args})
    row.update({k: getattr(config, k) for k in difusco_setting_args})
    row.update({k: getattr(config, k) for k in dev_setting_args})
    row.update(results)

    return row


def dataset_factory(config: Config, split: str = "test") -> Dataset:
    if split == "train":
        split = "training"
    assert split in ["test", "training", "validation"], f"Invalid split: {split}"
    data_path = os.path.join(config.data_path, getattr(config, f"{split}_split"))
    data_label_dir = (
        os.path.join(config.data_path, getattr(config, f"{split}_split_label_dir"))
        if getattr(config, f"{split}_split_label_dir", None)
        else None
    )

    if config.task == "mis":
        return MISDataset(data_dir=data_path, data_label_dir=data_label_dir)

    if config.task == "tsp":
        return TSPGraphDataset(data_file=data_path, sparse_factor=config.sparse_factor)

    error_msg = f"No dataset for task {config.task}."
    raise ValueError(error_msg)


def instance_factory(config: Config, sample: tuple) -> ProblemInstance:
    if config.task == "mis":
        return create_mis_instance(sample, device=config.device)
    if config.task == "tsp":
        return create_tsp_instance(sample, device=config.device, sparse_factor=config.sparse_factor)
    error_msg = f"No instance for task {config.task}."
    raise ValueError(error_msg)


def problem_factory(task: str) -> Problem:
    if task == "mis":
        return MISGaProblem()
    if task == "tsp":
        return TSPGAProblem()
    error_msg = f"No problem for task {task}."
    raise ValueError(error_msg)


class CustomLogger(Logger):
    def __init__(self, table_name: str, instance_id: int, gt_cost: float, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self.keys_to_log = ["iter", "mean_eval", "median_eval", "pop_best_eval", "best_eval", "worst_eval"]
        self.table_saver = TableSaver(table_name=table_name)
        self.instance_id = instance_id
        self.gt_cost = gt_cost
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _log(self, status: dict) -> None:
        data = {k: status[k] for k in self.keys_to_log}
        data["instance_id"] = self.instance_id
        data["gt_cost"] = self.gt_cost
        data["timestamp"] = self.timestamp
        self.table_saver.put(data)
