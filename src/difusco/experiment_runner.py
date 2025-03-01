from __future__ import annotations

import multiprocessing as mp
import traceback
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
import wandb

if TYPE_CHECKING:
    from config.myconfig import Config
    from torch_geometric.loader import DataLoader
from config.mytable import TableSaver
from pyinstrument import Profiler
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm


class Experiment(ABC):
    @abstractmethod
    def run_single_iteration(self, sample: tuple) -> None:
        pass

    @abstractmethod
    def get_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_final_results(self, results: list[dict]) -> dict:
        pass

    @abstractmethod
    def get_table_name(self) -> str:
        pass


class ExperimentRunner:
    """
    This class runs an experiment:
    - Runs a function for each sample in the given dataloader in a separate process
    - Logs the results to wandb if enabled
    - Saves the results to a table if enabled
    """

    def __init__(self, config: Config, experiment: Experiment) -> None:
        self.config = config
        self.experiment = experiment

    def process_iteration(self, sample: tuple[Any, ...], queue: mp.Queue) -> None:
        """Run the single iteration and store the result in the queue."""

        def run_iteration() -> None:
            try:
                result = self.experiment.run_single_iteration(sample)
                queue.put(result)
            except Exception:  # noqa: BLE001
                queue.put({"error": traceback.format_exc()})
            finally:
                # Clean up CUDA tensors
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()

        if self.config.profiler:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                run_iteration()
            print(p.key_averages().table(sort_by="cpu_time_total"))
        else:
            run_iteration()

    def main(self) -> None:
        """Run the evolutionary algorithm with optional profiling."""
        if self.config.profiler:
            with Profiler() as profiler:
                self.run()
            print(profiler.output_text(unicode=True, color=True))
        else:
            self.run()

    def run(self) -> None:
        print(f"Running experiment with config: {self.config}")
        dataloader = self.experiment.get_dataloader()

        is_validation_run = self.config.validate_samples is not None
        if not is_validation_run:
            wandb.init(
                project=self.config.project_name,
                name=self.config.wandb_logger_name,
                entity=self.config.wandb_entity,
                config=self.config.__dict__,
                dir=self.config.logs_path,
            )

        results = []
        ctx = mp.get_context("spawn")

        for i, sample in tqdm(enumerate(dataloader)):
            queue = ctx.Queue()
            process = ctx.Process(target=self.process_iteration, args=(sample, queue))

            process.start()
            process.join(timeout=120 * 60)  # 2h timeout
            if process.is_alive():
                process.terminate()
                raise TimeoutError(f"Process timed out for iteration {i}")

            if queue.empty():
                raise RuntimeError("No result returned from the process")

            run_results = queue.get()
            if "error" in run_results:
                raise RuntimeError(run_results["error"])

            results.append(run_results)
            if not is_validation_run:
                wandb.log(run_results, step=i)
            else:
                print(run_results)

            if process.is_alive():
                process.terminate()
                process.join()

            if is_validation_run and i >= self.config.validate_samples - 1:
                break

        if not is_validation_run:
            table_name = self.experiment.get_table_name()
            table_saver = TableSaver(table_name=table_name)
            final_results = self.experiment.get_final_results(results)
            final_results["wandb_id"] = wandb.run.id
            table_saver.put(final_results)

            wandb.finish()
