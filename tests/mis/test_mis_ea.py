# ruff: noqa: SLF001 we access private methods for testing

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch
from config.configs.mis_inference import config as mis_inference_config
from config.myconfig import Config
from difuscombination.dataset import MISDatasetComb
from evotorch.operators import CrossOver
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_ga import MISGACrossover, MISGAMutation, MISGaProblem, create_mis_ga
from problems.mis.mis_instance import MISInstance, MISInstanceBase, create_mis_instance
from scipy.sparse import csr_matrix
from torch_geometric.loader import DataLoader

if TYPE_CHECKING:
    from pathlib import Path


def read_mis_instance(device: str = "cpu") -> tuple[MISInstance, tuple]:
    resource_dir = "tests/resources"
    dataset = MISDataset(
        data_dir=os.path.join(resource_dir, "er_example_dataset"),
        data_label_dir=os.path.join(resource_dir, "er_example_dataset_annotations"),
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    sample = next(iter(dataloader))

    return create_mis_instance(sample, device=device), sample


common_config = Config(
    device="cpu",
    tournament_size=4,
    deselect_prob=0.05,
    opt_recomb_time_limit=15,
    mutation_prob=0.25,
    recombination="classic",
    initialization="random_feasible",
    preserve_optimal_recombination=False,
)


def test_create_mis_instance() -> None:
    instance, _ = read_mis_instance()
    assert instance.n_nodes == 756
    assert instance.gt_labels.sum().item() == 45
    assert (
        756
        == instance.gt_labels.shape[0]
        == instance.n_nodes
        == instance.adj_matrix_np.shape[0]
        == instance.adj_matrix_np.shape[1]
        == instance.neighbors_padded.shape[0]
        == instance.degrees.shape[0]
    )
    assert instance.edge_index.shape[0] == 2

    assert isinstance(instance, MISInstance)
    assert isinstance(instance.adj_matrix_np, csr_matrix)
    assert isinstance(instance.neighbors_padded, torch.Tensor)
    assert isinstance(instance.degrees, torch.Tensor)


def test_mis_problem_evaluation() -> None:
    instance, _ = read_mis_instance()
    config = common_config.update(
        recombination="classic",
        initialization="random_feasible",
        pop_size=10,
    )
    ga = create_mis_ga(instance, config=config, sample=())
    problem = ga.problem

    # create ind as a random tensor with size n_nodes
    ind = torch.rand(config.pop_size, instance.n_nodes)
    feasible = instance.get_feasible_from_individual_batch(ind)
    assert torch.equal(feasible.sum(dim=-1), problem._objective_func(feasible))

    # evaluate the labels (ground truth) of the instance
    obj_gt = instance.gt_labels.sum().item()
    assert obj_gt == problem._objective_func(instance.gt_labels.unsqueeze(0)).item()


@pytest.fixture
def square_instance() -> MISInstance:
    return MISInstance(
        n_nodes=4,
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]),
    )


def test_mis_degrees(square_instance: MISInstance) -> None:
    instance = square_instance
    degrees = instance.get_degrees()
    assert degrees.shape == (4,)
    assert degrees.sum() == 8
    assert (degrees == torch.tensor([2, 2, 2, 2])).all()
    assert torch.equal(degrees, instance.degrees)


def assert_valid_initialized_population(values: torch.Tensor, instance: MISInstanceBase) -> None:
    assert values.shape[1] == instance.n_nodes

    # Check that the sum of every row is equal to instance.evaluate_solution
    for i in range(values.shape[0]):
        assert values[i].sum() == instance.evaluate_solution(values[i])


def test_mis_ga_fill_random_feasible() -> None:
    instance, sample = read_mis_instance()
    config = common_config.update(
        pop_size=10,
    )
    ga = create_mis_ga(instance, config=config, sample=sample)
    problem = ga.problem

    values = torch.zeros(config.pop_size, instance.n_nodes, dtype=torch.bool)
    problem._fill(values)

    assert_valid_initialized_population(values, instance)

    # Check that the first solution is the deterministic construction heuristic using degrees as priorities
    assert (values[0] == instance.get_feasible_from_individual(-instance.get_degrees())).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available, skipping test that requires GPU")
def test_mis_ga_fill_difusco() -> None:
    instance, sample = read_mis_instance(device="cuda")

    pop_size = 20

    config = Config(
        pop_size=pop_size,
        parallel_sampling=pop_size,
        sequential_sampling=1,
        inference_diffusion_steps=2,
        diffusion_steps=2,
        initialization="difusco_sampling",
        data_path="data",
        models_path="models",
        task="mis",
        training_split="mis/er_50_100/train",
        training_split_label_dir="mis/er_50_100/train_labels",
        test_split="mis/er_50_100/test",
        test_split_label_dir="mis/er_50_100/test_labels",
        validation_split="mis/er_50_100/test",
        validation_split_label_dir="mis/er_50_100/test_labels",
        ckpt_path="mis/mis_er_50_100_gaussian.ckpt",
        recombination="classic",
    )

    config = mis_inference_config.update(config)
    problem = MISGaProblem(instance, config=config, sample=sample)

    values = torch.zeros(pop_size, instance.n_nodes, dtype=torch.bool, device=config.device)
    problem._fill(values)

    assert_valid_initialized_population(values, instance)


def test_mis_ga_crossover_small(square_instance: MISInstanceBase) -> None:
    instance = square_instance

    config = common_config.update(
        pop_size=4,
    )
    ga = create_mis_ga(instance, config=config, sample=())

    parents_1 = torch.from_numpy(np.array([[1, 0, 1, 0], [0, 1, 0, 1]]))
    parents_2 = torch.from_numpy(np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))

    crossover = ga._operators[0]
    assert isinstance(crossover, MISGACrossover)
    children = crossover._do_cross_over(parents_1, parents_2)

    assert children.values.shape == (4, instance.n_nodes)
    assert (children.values[0] == torch.tensor([1, 0, 1, 0])).all()
    assert (children.values[1] == torch.tensor([0, 1, 0, 1])).all()
    assert children.values[2].sum() == 2
    assert children.values[3].sum() == 2


@pytest.fixture(params=["difuscombination", "classic", "optimal"])
def recombination_config(request: pytest.FixtureRequest) -> Config:
    from config.configs.mis_inference import config as mis_inference_config

    recombination = request.param

    samples_file = "difuscombination/mis/er_50_100/test"
    labels_dir = "difuscombination/mis/er_50_100/test_labels"
    graphs_dir = "mis/er_50_100/test"

    return mis_inference_config.update(
        models_path="models",
        data_path="data",
        task="mis",
        pop_size=4,
        device="cpu",
        initialization="random_feasible",
        recombination=recombination,
        test_samples_file=samples_file,
        test_labels_dir=labels_dir,
        test_graphs_dir=graphs_dir,
        ckpt_path="mis/mis_er_50_100_gaussian.ckpt",
        ckpt_path_difuscombination="difuscombination/mis_er_50_100_gaussian.ckpt",
        test_split=graphs_dir,
        test_split_label_dir=None,
        tournament_size=4,
        deselect_prob=0.05,
        opt_recomb_time_limit=15,
    )


def test_mis_ga_crossovers(recombination_config: Config) -> None:
    config = common_config.update(recombination_config)

    dataset = MISDatasetComb(
        samples_file=os.path.join("data", config.test_samples_file),
        graphs_dir=os.path.join("data", config.test_graphs_dir),
        labels_dir=os.path.join("data", config.test_labels_dir),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))

    instance = create_mis_instance(batch, device="cpu")
    parents_1 = torch.rand((config.pop_size // 2, instance.n_nodes))
    parents_2 = torch.rand((config.pop_size // 2, instance.n_nodes))
    parents_1 = instance.get_feasible_from_individual_batch(parents_1)
    parents_2 = instance.get_feasible_from_individual_batch(parents_2)
    parents_1 = parents_1.int()
    parents_2 = parents_2.int()

    ga = create_mis_ga(instance, config=config, sample=batch)
    crossover = ga._operators[0]
    assert isinstance(crossover, CrossOver)
    children = crossover._do_cross_over(parents_1, parents_2)

    assert children.values.shape == (config.pop_size, instance.n_nodes)
    for i in range(config.pop_size):
        assert children.values[i].sum() == instance.evaluate_individual(children.values[i].clone().int())


def test_mis_ga_recombination_in_ga(recombination_config: Config) -> None:
    """
    This checks that the result of the recombination is the best cost of the population
    """
    config = common_config.update(recombination_config).update(
        pop_size=4,
        deselect_prob=0,
        mutation_prob=0.25,
    )

    dataset = MISDatasetComb(
        samples_file=os.path.join("data", config.test_samples_file),
        graphs_dir=os.path.join("data", config.test_graphs_dir),
        labels_dir=os.path.join("data", config.test_labels_dir),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))

    instance = create_mis_instance(batch, device="cpu")

    ga = create_mis_ga(instance, config=config, sample=batch)
    ga.run(num_generations=1)

    # evaluate the population manually
    pop = ga.population.values
    evals = pop.sum(dim=-1)
    assert evals.max().item() == ga.status["pop_best_eval"]


def test_mis_ga_one_generation(recombination_config: Config) -> None:
    config = common_config.update(recombination_config).update(
        pop_size=16,
        deselect_prob=0,
        mutation_prob=0.25,
    )

    dataset = MISDatasetComb(
        samples_file=os.path.join("data", config.test_samples_file),
        graphs_dir=os.path.join("data", config.test_graphs_dir),
        labels_dir=os.path.join("data", config.test_labels_dir),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))

    instance = create_mis_instance(batch, device="cpu")

    ga = create_mis_ga(instance, config=config, sample=batch)
    ga.run(num_generations=1)

    # evaluate the population manually
    pop = ga.population.values
    evals = pop.sum(dim=-1)
    assert evals.max().item() == ga.status["pop_best_eval"]


def test_mis_save_optimal_recombination_results(recombination_config: Config, tmp_path: Path) -> None:
    if recombination_config.recombination != "optimal":
        return

    config = common_config.update(recombination_config).update(deselect_prob=0, mutation_prob=0.25)

    dataset = MISDatasetComb(
        samples_file=os.path.join("data", config.test_samples_file),
        graphs_dir=os.path.join("data", config.test_graphs_dir),
        labels_dir=os.path.join("data", config.test_labels_dir),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))

    instance = create_mis_instance(batch, device="cpu")

    ga = create_mis_ga(instance, config=config, sample=batch, tmp_dir=tmp_path)

    ga.run(num_generations=1)

    # if we are doing optimal recombination, check that the results are saved
    if config.recombination == "optimal":
        assert os.path.exists(os.path.join(tmp_path, "optimal_recombination.csv"))
        results = ga.get_recombination_saved_results()
        assert len(results) == config.pop_size // 2


def test_mis_ga_mutation(square_instance: MISInstanceBase) -> None:
    instance = square_instance
    config = common_config.update(
        pop_size=2,
        deselect_prob=0.05,
        opt_recomb_time_limit=15,
        mutation_prob=0.25,
        preserve_optimal_recombination=False,
    )
    ga = create_mis_ga(instance, config=config, sample=())

    # Set first individual to [1, 0, 1, 0]
    data = ga.population.access_values()
    data[0] = torch.tensor([1, 0, 1, 0])

    mutation = ga._operators[1]
    assert isinstance(mutation, MISGAMutation)

    # Mock random values to force deselection of first node for first individual
    # and no mutations for second individual
    with patch(
        "torch.rand",
        side_effect=[
            torch.tensor([0.0, 1.0]),  # first call is the probability of mutation
            torch.tensor([[0.0, 1.0, 0.0, 1.0]]),  # First ind: deselect mask
            torch.tensor([[0.0, 1.0, 0.0, 1.0]]),  # First ind: priorities for selection
        ],
    ):
        children = mutation._do(ga.population)

    assert children.values.shape == (2, instance.n_nodes)
    assert torch.equal(children.values[0], torch.tensor([0, 1, 0, 1]))  # Mutated to opposite
    assert torch.equal(children.values[1], ga.population.values[1])  # No mutation


def test_mis_ga_mutation_no_deselection(square_instance: MISInstanceBase) -> None:
    instance = square_instance
    config = common_config.update(pop_size=2, deselect_prob=0)
    ga = create_mis_ga(instance, config=config, sample=())

    mutation = ga._operators[1]
    assert isinstance(mutation, MISGAMutation)

    with patch(
        "torch.rand",
        side_effect=[
            torch.zeros(2),  # mutation probability
            torch.ones(2, 4),  # deselect mask
            torch.ones(2, 4),  # priorities
        ],
    ):
        children = mutation._do(ga.population)

    assert torch.equal(children.values[0], ga.population.values[0])
    assert torch.equal(children.values[1], ga.population.values[1])


def test_mis_ga_mutation_no_mutation_prob(square_instance: MISInstanceBase) -> None:
    instance = square_instance
    config = common_config.update(pop_size=2, mutation_prob=0)
    ga = create_mis_ga(instance, config=config, sample=())

    mutation = ga._operators[1]
    assert isinstance(mutation, MISGAMutation)

    with patch("torch.rand", return_value=torch.ones(2)):
        children = mutation._do(ga.population)

    assert torch.equal(children.values[0], ga.population.values[0])
    assert torch.equal(children.values[1], ga.population.values[1])


def test_mis_ga_mutation_preserve_optimal_recombination() -> None:
    """we need to check that the first half of the population is not mutated"""
    instance, sample = read_mis_instance()
    config = common_config.update(
        pop_size=4,
        recombination="optimal",
        preserve_optimal_recombination=True,
        mutation_prob=1,
    )
    ga = create_mis_ga(instance, config=config, sample=sample)

    mutation = ga._operators[1]
    assert isinstance(mutation, MISGAMutation)

    # set the first half of the population to all false solutions
    n_pairs = config.pop_size // 2
    data = ga.population.access_values()
    data[:n_pairs] = torch.zeros(n_pairs, instance.n_nodes, dtype=torch.bool, device=config.device)

    assert (ga.population.values[:n_pairs].sum(dim=-1) == 0).all()
    assert (ga.population.values[n_pairs:].sum(dim=-1) > 0).all()
    mutation._do(ga.population)
    # we do not mutate the first half of the population, so they should still be all false
    assert (ga.population.values[:n_pairs].sum(dim=-1) == 0).all()
    assert (ga.population.values[n_pairs:].sum(dim=-1) > 0).all()


def test_duplicate_batch() -> None:
    # Create a mock configuration
    config = Config(pop_size=10)

    # batch will be yielded by the dataloader
    dataset = MISDatasetComb(
        samples_file="data/difuscombination/mis/er_50_100/test",
        graphs_dir="data/mis/er_50_100/test",
        labels_dir="data/difuscombination/mis/er_50_100/test_labels",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))

    # Call the _get_batch method
    result_batch = MISGaProblem._duplicate_batch(config.pop_size // 2, batch)

    # Check if the result is a Batch object
    assert isinstance(result_batch, tuple)
    assert len(result_batch) == 3
    assert result_batch[0].shape == (config.pop_size // 2, batch[0].shape[1])
    assert result_batch[1].x.shape == (config.pop_size // 2 * 56, 3)
    assert result_batch[2].shape == (config.pop_size // 2, 1)


def test_temp_saver(tmp_path: Path) -> None:
    instance, sample = read_mis_instance()
    config = common_config.update(pop_size=10)
    ga = create_mis_ga(instance, config=config, sample=sample, tmp_dir=tmp_path)

    # Access the temp_saver directly from the ga instance instead of from operators
    saver = ga._temp_saver
    population = ga.population

    # check the solution string representation
    sol_str = saver._get_population_string(population)
    assert len(sol_str.split(" | ")) == config.pop_size

    # check the saving of the solution
    saver.save(population)
    assert os.path.exists(os.path.join(tmp_path, "population.txt"))
    with open(os.path.join(tmp_path, "population.txt")) as f:
        last_line = f.readlines()[-1]
        assert last_line.strip() == sol_str
