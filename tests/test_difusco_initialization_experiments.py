from __future__ import annotations

import pytest
from config.config import Config
from difusco.difusco_initialization_experiments import run_difusco_initialization_experiments
from config.configs.mis_inference import config as mis_inference_config
from config.configs.tsp_inference import config as tsp_inference_config

@pytest.fixture
def config_factory() -> Config:
    def _config_factory(task: str) -> Config:
        pop_size = 2
        if task == "mis":
            config = Config(
                task="mis",
                parallel_sampling=pop_size,
                sequential_sampling=1,
                diffusion_steps=2,
                inference_diffusion_steps=50,
                validate_samples=2,
                np_eval=True,
                pop_size=pop_size,
            )
            return mis_inference_config.update(config)
        if task == "tsp":
            config = Config(
                task="tsp",
                parallel_sampling=pop_size,
                sequential_sampling=1,
                diffusion_steps=2,
                inference_diffusion_steps=50,
                validate_samples=2,
                np_eval=True,
                pop_size=pop_size,
            )
            return tsp_inference_config.update(config)
    return _config_factory

@pytest.mark.parametrize("task", ["mis", "tsp"])
def test_difusco_initialization_experiments(config_factory: Config, task: str) -> None:
    config = config_factory(task)
    run_difusco_initialization_experiments(config)
