from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from ea.arg_parser import get_arg_parser

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ea.config import Config


def filter_args_by_group(parser: ArgumentParser, group_name: str) -> dict:
    group = next((g for g in parser._action_groups if g.title == group_name), None)  # noqa: SLF001
    if group is None:
        error_msg = f"Group '{group_name}' not found"
        raise ValueError(error_msg)

    return [a.dest for a in group._group_actions]  # noqa: SLF001


def save_results(config: Config, results: dict[str, float | int | str]) -> None:
    # filter the name of the arguments of the grup ea_settings
    parser = get_arg_parser()
    ea_setting_args = filter_args_by_group(parser, "ea_settings")

    row = {
        "task": config.task,
        "wandb_logger_name": config.wandb_logger_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    row.update({k: getattr(config, k) for k in ea_setting_args})
    row.update(results)
    results_df = pd.DataFrame([row])

    results_file = os.path.join(config.results_path, "ea_results.csv")
    current_results = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
    current_results = pd.concat([current_results, results_df])

    current_results.to_csv(results_file, index=False)
