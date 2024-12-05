# ruff: noqa: SLF001 we access private methods for testing

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ea.config import Config
from evotorch import Problem
from problems.mis.mis_brkga import create_mis_brkga
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_ga import MISGACrossover, MISGAMutation, MISGaProblem, create_mis_ga
from problems.mis.mis_instance import MISInstance, MISInstanceBase, MISInstanceNumPy, create_mis_instance
from scipy.sparse import coo_matrix
from torch_geometric.loader import DataLoader


def read_mis_instance(np_eval: bool = False) -> MISInstance:
    resource_dir = "tests/resources"
    dataset = MISDataset(
        data_dir=os.path.join(resource_dir, "er_example_dataset"),
        data_label_dir=os.path.join(resource_dir, "er_example_dataset_annotations"),
    )

    sample = dataset.__getitem__(0)

    return create_mis_instance(sample, np_eval=np_eval)


@pytest.mark.parametrize("np_eval", [False, True])
def test_create_mis_instance(np_eval: bool) -> None:
    instance = read_mis_instance(np_eval=np_eval)
    assert instance.n_nodes == 788
    assert instance.gt_labels.sum().item() == 45
    assert (
        788
        == instance.gt_labels.shape[0]
        == instance.n_nodes
        == instance.adj_matrix.shape[0]
        == instance.adj_matrix.shape[1]
    )


@pytest.mark.parametrize("np_eval", [False, True])
def test_mis_brkga_runs(np_eval: bool) -> None:
    instance = read_mis_instance(np_eval=np_eval)

    brkga = create_mis_brkga(instance, config=Config(pop_size=5, n_parallel_evals=0, device="cpu"))
    brkga.run(num_generations=2)

    status = brkga.status
    assert status["iter"] == 2


@pytest.mark.parametrize("np_eval", [False, True])
def test_mis_problem_evaluation(np_eval: bool) -> None:
    instance = read_mis_instance(np_eval=np_eval)

    problem = Problem(
        objective_func=instance.evaluate_individual,
        objective_sense="max",
        solution_length=instance.n_nodes,
        device="cpu",
    )

    # create ind as a random tensor with size n_nodes
    ind = torch.rand(instance.n_nodes)
    obj = instance.evaluate_individual(ind)
    assert obj == problem._objective_func(ind)

    # evaluate the labels (ground truth) of the instance
    obj_gt = instance.evaluate_individual(instance.gt_labels)
    assert obj_gt == problem._objective_func(instance.gt_labels)

    assert obj_gt == instance.gt_labels.sum()
    assert obj_gt >= obj


@pytest.mark.parametrize("np_eval", [False, True])
def test_mis_brkga_runs_with_dataloader(np_eval: bool) -> None:
    dataset = MISDataset(
        data_dir="tests/resources/er_example_dataset",
        data_label_dir="tests/resources/er_example_dataset_annotations",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for sample in dataloader:
        instance = create_mis_instance(sample, device="cpu", np_eval=np_eval)
        ga = create_mis_brkga(instance, config=Config(pop_size=10, device="cpu", n_parallel_evals=0))
        ga.run(num_generations=2)

        status = ga.status
        assert status["iter"] == 2
        break


@pytest.fixture
def square_instance(np_eval: bool) -> MISInstance | MISInstanceNumPy:
    if np_eval:
        instance = MISInstanceNumPy(
            adj_matrix=coo_matrix(np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])).tocsr(),
            n_nodes=4,
        )
    else:
        # Create indices for non-zero elements
        indices = torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 3],  # row indices
                [1, 3, 0, 2, 1, 3, 0, 2],
            ]
        )  # column indices
        values = torch.ones(8)  # values for the non-zero elements

        instance = MISInstance(
            adj_matrix=torch.sparse_coo_tensor(indices=indices, values=values, size=(4, 4)).coalesce(),
            n_nodes=4,
        )
    return instance


@pytest.mark.parametrize("np_eval", [False, True])
def test_mis_degrees(square_instance: MISInstanceBase) -> None:
    instance = square_instance
    degrees = instance.get_degrees()
    assert degrees.shape == (4,)
    assert degrees.sum() == 8
    assert (degrees == torch.tensor([2, 2, 2, 2])).all()


@pytest.mark.parametrize("np_eval", [False, True])
def test_mis_ga_fill(np_eval: bool) -> None:
    instance = read_mis_instance(np_eval=np_eval)
    problem = MISGaProblem(instance, Config(pop_size=10, device="cpu", n_parallel_evals=0))

    values = torch.zeros(10, instance.n_nodes, dtype=torch.bool)
    problem._fill(values)  # just for testing

    # Check that the sum of every row is equal to instance.evaluate_solution
    for i in range(values.shape[0]):
        assert values[i].sum() == instance.evaluate_solution(values[i])

    # Check that the first solution is the deterministic construction heuristic
    assert (values[0] == instance.get_feasible_from_individual(-instance.get_degrees())).all()


@pytest.mark.parametrize("np_eval", [False, True])
def test_mis_ga_crossover(np_eval: bool, square_instance: MISInstanceBase) -> None:
    instance = square_instance

    ga = create_mis_ga(instance, config=Config(pop_size=4, device="cpu", n_parallel_evals=0, np_eval=np_eval))

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


@pytest.mark.parametrize("np_eval", [False, True])
def test_mis_ga_mutation(np_eval: bool, square_instance: MISInstanceBase) -> None:
    instance = square_instance
    ga = create_mis_ga(instance, config=Config(pop_size=2, device="cpu", n_parallel_evals=0, np_eval=np_eval))

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
            torch.tensor(
                [
                    [0.0, 0.9, 0.9, 0.9],  # First ind: deselect first node
                    [0.9, 0.9, 0.9, 0.9],  # Second ind: no mutations
                ]
            ),
            torch.tensor(
                [
                    [0.0, 1.0, 0.0, 1.0],  # First ind: let's have the other solution
                    [0.0, 0.0, 0.0, 0.0],  # Second ind: will be unused
                ]
            ),
        ],
    ):
        children = mutation._do(ga.population)

    assert children.values.shape == (2, instance.n_nodes)
    assert torch.equal(children.values[0], torch.tensor([0, 1, 0, 1]))  # Mutated to opposite
    assert torch.equal(children.values[1], ga.population.values[1])  # No mutation


@pytest.mark.parametrize("np_eval", [False, True])
def test_mis_ga_mutation_no_deselection(np_eval: bool, square_instance: MISInstanceBase) -> None:
    instance = square_instance
    ga = create_mis_ga(instance, config=Config(pop_size=2, device="cpu", n_parallel_evals=0, np_eval=np_eval))

    mutation = ga._operators[1]
    assert isinstance(mutation, MISGAMutation)

    with patch("torch.rand", return_value=torch.ones(2, 4)):
        children = mutation._do(ga.population)

    assert torch.equal(children.values[0], ga.population.values[0])
    assert torch.equal(children.values[1], ga.population.values[1])


@pytest.mark.parametrize("np_eval", [False, True])
def test_mis_ga_runs_with_dataloader(np_eval: bool) -> None:
    dataset = MISDataset(
        data_dir="tests/resources/er_example_dataset",
        data_label_dir="tests/resources/er_example_dataset_annotations",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for sample in dataloader:
        instance = create_mis_instance(sample, device="cpu", np_eval=np_eval)
        brkga = create_mis_brkga(instance, config=Config(pop_size=10, device="cpu", n_parallel_evals=0))
        brkga.run(num_generations=2)

        status = brkga.status
        assert status["iter"] == 2


def test_gpu_memory() -> None:
    # Helper function to check GPU memory usage
    def get_gpu_memory() -> dict[str, float]:
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "cached_mb": torch.cuda.memory_reserved() / 1024**2,
        }

    dataset = MISDataset(
        data_dir="tests/resources/er_example_dataset",
        data_label_dir="tests/resources/er_example_dataset_annotations",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for sample in dataloader:
        instance = create_mis_instance(sample, device="cuda", np_eval=False)
        brkga = create_mis_brkga(instance, config=Config(pop_size=10, device="cuda", n_parallel_evals=0))
        brkga.run(num_generations=2)

        status = brkga.status
        memory = get_gpu_memory()
        assert memory["allocated_mb"] > 0
        assert memory["cached_mb"] > 0

        del instance
        del brkga
        del status

        from gc import collect

        collect()
        torch.cuda.empty_cache()
        new_memory = get_gpu_memory()

        assert new_memory["allocated_mb"] == 0
        assert new_memory["cached_mb"] <= memory["cached_mb"]
        break


if __name__ == "__main__":
    test_gpu_memory()
