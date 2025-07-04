from __future__ import annotations

import pytest
import torch
from config.myconfig import Config
from difuscombination.recombination_experiments import main_recombination_experiments


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available, skipping test that requires GPU",
)
def test_recombination_experiments() -> None:
    """Test that recombination experiments can run with minimal configuration."""

    config = Config(
        task="mis",
        data_path="/home/e12223411/repos/difusco/data",
        logs_path="/home/e12223411/repos/difusco/logs",
        results_path="/home/e12223411/repos/difusco/results",
        models_path="/home/e12223411/repos/difusco/models",
        test_graphs_dir="mis/er_50_100/test",
        test_samples_file="difuscombination/mis/er_50_100/test",
        test_labels_dir="difuscombination/mis/er_50_100/test_labels",
        ckpt_path_difusco="mis/mis_er_50_100_gaussian.ckpt",
        ckpt_path_difuscombination="difuscombination/mis_er_50_100_gaussian.ckpt",
        parallel_sampling=2,
        sequential_sampling=1,
        diffusion_steps=2,
        inference_diffusion_steps=50,
        validate_samples=2,
        profiler=False,
        device="cuda",
        num_processes=1,
        process_idx=0,
        split="test",
    )
    from config.configs.mis_inference import config as mis_inference_config

    config = mis_inference_config.update(config)

    # Run the experiment
    main_recombination_experiments(config)
