from __future__ import annotations

import os
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Literal

import torch
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.decorators import vectorized
from evotorch.operators import CopyingOperator, CrossOver
from problems.mis.solve_optimal_recombination import solve_problem
from torch import no_grad

from difusco.sampler import DifuscoSampler

if TYPE_CHECKING:
    from config.myconfig import Config
    from problems.mis.mis_instance import MISInstance

from torch_geometric.data import Batch


@vectorized
def evaluate_population(population: torch.Tensor) -> torch.Tensor:
    return population.sum(dim=-1)


class MISGaProblem(Problem):
    def __init__(self, instance: MISInstance, config: Config, sample: tuple) -> None:
        self.instance = instance
        self.config = config
        self.config.task = "mis"
        self.sample = sample

        if config.recombination == "difuscombination":
            self.sampler = self._get_difuscombination_sampler()
            self.batch = self._duplicate_batch(config.pop_size // 2, self.sample)
        else:
            self.sampler = None
            self.batch = None

        super().__init__(
            objective_func=evaluate_population,
            objective_sense="max",
            solution_length=instance.n_nodes,
            device=config.device,
            dtype=torch.bool,
        )

    @staticmethod
    def _fake_paths_for_difuscombination_models(config: Config) -> Config:
        # fake paths for difuscombination
        config.test_graphs_dir = config.test_split
        config.training_samples_file = config.test_samples_file
        config.training_labels_dir = config.test_labels_dir
        config.training_graphs_dir = config.test_split
        config.validation_samples_file = config.test_samples_file
        config.validation_labels_dir = config.test_labels_dir
        config.validation_graphs_dir = config.test_split
        return config

    @staticmethod
    def _fake_paths_for_difusco_models(config: Config) -> Config:
        config.training_split = config.test_split
        config.training_split_label_dir = config.test_split_label_dir
        config.validation_split = config.test_split
        config.validation_split_label_dir = config.test_split_label_dir
        return config

    def _get_difuscombination_sampler(self) -> DifuscoSampler:
        config = self.config.update(
            parallel_sampling=2,  # for every pairing, we generate 2 children
            sequential_sampling=1,
            device="cuda",
            mode="difuscombination",
        )
        config = self._fake_paths_for_difuscombination_models(config).update(
            ckpt_path=config.ckpt_path_difuscombination,
        )
        return DifuscoSampler(config)

    def _get_difusco_sampler(self) -> DifuscoSampler:
        config = self.config.update(
            parallel_sampling=self.config.pop_size,
            sequential_sampling=1,
            mode="difusco",
        )
        config = MISGaProblem._fake_paths_for_difusco_models(config)
        return DifuscoSampler(config)

    @staticmethod
    def _duplicate_batch(n_times: int, batch: tuple) -> tuple:
        # Duplicate the first tensor
        tensor0 = batch[0]  # e.g., shape: [1, ...]
        tensor0_dup = tensor0.repeat(n_times, 1)  # repeat along the first dimension

        # Handle the DataBatch in index 1
        # Convert it to a list of Data objects (should contain one element)
        data_list = batch[1].to_data_list()
        # Duplicate the single graph n times
        duplicated_data = [deepcopy(data_list[0]) for _ in range(n_times)]
        # Rebuild the DataBatch; this will automatically handle shifting node indices
        # and creating a new `batch` attribute
        new_data_batch = Batch.from_data_list(duplicated_data)

        # Duplicate the third tensor
        tensor2 = batch[2]  # e.g., shape: [1, ...]
        tensor2_dup = tensor2.repeat(n_times, 1)

        # Return the new batch as a list
        return (tensor0_dup, new_data_batch, tensor2_dup)

    def _fill(self, values: torch.Tensor) -> None:
        if self.config.initialization == "random_feasible":
            return self._fill_random_feasible_initialization(values)
        if self.config.initialization == "difusco_sampling":
            if self.config.device != "cuda":
                warnings.warn(
                    "We recommend using CUDA for Difusco sampling. Performance may be degraded.",
                    UserWarning,
                    stacklevel=2,
                )
            return self._fill_difusco_sampling(values)
        error_msg = f"Invalid initialization method: {self.config.initialization}"
        raise ValueError(error_msg)

    def _fill_random_feasible_initialization(self, values: torch.Tensor) -> None:
        """
        Values is a tensor of shape (B, solution_length).
        Initialization heuristic: (randomized) construction heuristic based on node degree.
        """
        degrees = self.instance.get_degrees()
        inversed_normalized_degrees = 1 - degrees / degrees.max()

        # Create scaling factors for noise (0 to 0.2)
        scales = torch.linspace(0, 0.2, values.shape[0], device=self.device)
        # Generate noise for all solutions at once (pop_size, solution_length)
        noise = torch.randn(values.shape, device=self.device) * scales.unsqueeze(1)
        # Broadcast inversed_normalized_degrees to match the shape
        priorities = inversed_normalized_degrees.unsqueeze(0) + noise

        values[:] = self.instance.get_feasible_from_individual_batch(priorities)

    def _fill_difusco_sampling(self, values: torch.Tensor) -> None:
        """
        Values is a tensor of shape (B, solution_length).
        Uses Difusco to sample initial solutions.
        """
        sampler = self._get_difusco_sampler()
        popsize = self.config.pop_size
        assert popsize == values.shape[0], "Population size must match the number of solutions"
        assert (
            self.config.parallel_sampling * self.config.sequential_sampling == popsize
        ), "Population size must match the number of solutions"

        # Sample node scores using Difusco
        node_scores = sampler.sample(self.sample)

        # Convert scores to feasible solutions
        values[:] = self.instance.get_feasible_from_individual_batch(node_scores)


class MISGAMutation(CopyingOperator):
    def __init__(
        self,
        problem: Problem,
        instance: MISInstance,
        deselect_prob: float = 0.05,
        mutation_prob: float = 0.25,
        optimal_recombination: bool = False,
    ) -> None:
        """
        Mutation operator for the Maximum Independent Set problem. With probability deselect_prob, a selected node is
        unselected and gets a probability of zero. Solution is then made feasible.
        Only applies mutation with probability mutation_prob. When optimal_recombination is True, mutation is not
        applied to the first half of the population.

        Args:
            problem: The problem object to work with.
            instance: The instance object to work with.
            deselect_prob: The probability of deselecting a selected node.
            mutation_prob: The probability of mutating a solution.
            optimal_recombination: Whether optimal recombination is used.
                In this case, the mutation is not applied for the first half of the population.
        """

        super().__init__(problem)
        self._instance = instance
        self._deselect_prob = deselect_prob
        self._mutation_prob = mutation_prob
        self._optimal_recombination = optimal_recombination

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        print(f"pre-mutation: {data.sum(dim=-1)}")

        pop_size, n_nodes = data.shape

        # Decide which individuals to mutate
        mutation_mask = torch.rand(pop_size, device=data.device, dtype=torch.float16) <= self._mutation_prob
        if self._optimal_recombination:
            # Skip mutation for the first half of the population
            mutation_mask[: pop_size // 2] = False

        # If no individuals are selected for mutation, exit early.
        if not mutation_mask.any():
            print(f"post-mutation: {data.sum(dim=-1)}")
            return result

        # Get indices of individuals to mutate.
        mutate_indices = mutation_mask.nonzero(as_tuple=True)[0]

        # Generate deselect mask and priorities only for the solutions that are going to be mutated.
        sub_data = data[mutate_indices]
        deselect_mask = torch.rand(sub_data.shape, device=data.device, dtype=torch.float32) <= self._deselect_prob
        priorities = torch.rand(sub_data.shape, device=data.device, dtype=torch.float32)

        # For nodes that are both deselected and originally selected, set priority to zero.
        priorities[deselect_mask & sub_data.bool()] = 0

        # Only update those solutions that have at least one deselected node.
        update_mask = deselect_mask.sum(dim=-1) > 0
        if update_mask.any():
            indices_to_update = mutate_indices[update_mask]
            feasible = self._instance.get_feasible_from_individual_batch(priorities[update_mask])
            data[indices_to_update] = feasible

        print(f"post-mutation: {data.sum(dim=-1)}")
        return result


class MISGACrossverOptimal(CrossOver):
    def __init__(
        self,
        problem: MISGaProblem,
        instance: MISInstance,
        tournament_size: int = 4,
        opt_recomb_time_limit: int = 15,
    ) -> None:
        super().__init__(
            problem,
            tournament_size=tournament_size,
            cross_over_rate=0.5,
        )
        self._instance = instance
        self._opt_recomb_time_limit = opt_recomb_time_limit

    @torch.no_grad()
    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        return self._do_cross_over_optimal(parents1, parents2)

    @no_grad()
    def _do_cross_over_optimal(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        """
        parents1 and parents2 are two solutions of shape (num_pairings, n_nodes).
        """
        num_pairings = parents1.shape[0]
        device = parents1.device

        children = parents1.clone()

        for i in range(num_pairings):
            solution_1 = parents1[i].cpu().numpy().nonzero()[0]
            solution_2 = parents2[i].cpu().numpy().nonzero()[0]

            result = solve_problem(self._instance, solution_1, solution_2, time_limit=self._opt_recomb_time_limit)

            children[i] = torch.tensor(result["children_np_labels"], device=device)

        return self._make_children_batch(children)


class MISGACrossover(CrossOver):
    def __init__(
        self,
        problem: MISGaProblem,
        instance: MISInstance,
        tournament_size: int = 4,
        mode: Literal["classic", "difuscombination"] = "classic",
    ) -> None:
        super().__init__(problem, tournament_size=tournament_size)
        self._instance = instance
        self._mode = mode

    @no_grad()
    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        if self._mode == "classic":
            return self._do_cross_over_classic(parents1, parents2)
        if self._mode == "difuscombination":
            return self._do_cross_over_difuscombination(parents1, parents2)

        raise ValueError(f"Invalid mode: {self._mode}")

    @no_grad()
    def _do_cross_over_difuscombination(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        """
        parents1 and parents2 are two solutions of shape (num_pairings, n_nodes).
        DifuscSampling parameters:
        - parallel_sampling: 2
        - sequential_sampling: 1
        - batch_size: num_pairings
        """
        num_pairings = parents1.shape[0]

        features = torch.stack([parents1, parents2], dim=2)
        assert features.shape == (num_pairings, self.problem.solution_length, 2), "Incorrect features shape"
        # we need to reshape the features to (num_pairings * n_nodes, 2)
        features = features.reshape(num_pairings * self.problem.solution_length, 2)
        assert features.shape == (num_pairings * self.problem.solution_length, 2), "Incorrect features shape"
        heatmaps = self._problem.sampler.sample(self._problem.batch, features=features).to(self.problem.device)
        assert heatmaps.shape == (
            num_pairings,
            2,
            self.problem.solution_length,
        ), f"Incorrect heatmaps shape: {heatmaps.shape}, expected (num_pairings, 2, solution_length)"

        # split into two children by dropping dimension 1 -> (num_pairings, solution_length)
        heatmaps_child1 = heatmaps.select(1, 0)
        heatmaps_child2 = heatmaps.select(1, 1)

        heatmaps_child = torch.cat([heatmaps_child1, heatmaps_child2], dim=0)
        children = self._instance.get_feasible_from_individual_batch(heatmaps_child)
        return self._make_children_batch(children)

    @no_grad()
    def _do_cross_over_classic(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        """
        Parents are two solutions of shape (num_pairings, n_nodes).
        """
        num_pairings = parents1.shape[0]
        device = parents1.device

        # Find common nodes between parents
        common_nodes = (parents1 & parents2).bool()  # Element-wise AND

        # Random values between 0 and 0.5, 1 if node forced to selection
        priority1 = torch.rand(num_pairings, self.problem.solution_length, device=device, dtype=torch.float32) * 0.5
        priority1[common_nodes] = 1
        # Random values between 0.5 and 1, 0 if node penalized for selection
        priority2 = (
            torch.rand(num_pairings, self.problem.solution_length, device=device, dtype=torch.float32) * 0.5 + 0.5
        )
        priority2[common_nodes] = 0

        priorities = torch.cat([priority1, priority2], dim=0)
        children = self._instance.get_feasible_from_individual_batch(priorities)
        return self._make_children_batch(children)


class TempSaver(CopyingOperator):
    """A fake operator that saves the population to a file."""

    def __init__(self, problem: Problem, tmp_file: str) -> None:
        super().__init__(problem)
        self._tmp_file = tmp_file
        os.makedirs(os.path.dirname(tmp_file), exist_ok=True)

    def _get_population_string(self, batch: SolutionBatch) -> str:
        data = batch.access_values()
        # for each solution in the batch, take the non-zero indices
        solutions_str = []
        for i in range(data.shape[0]):
            indices = data[i].nonzero().flatten().tolist()
            solutions_str.append(",".join(map(str, indices)))
        return " | ".join(solutions_str)

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        with open(self._tmp_file, "a") as f:
            f.write(self._get_population_string(batch) + "\n")  # Added newline for better readability
        return batch


def create_mis_ga(
    instance: MISInstance, config: Config, sample: tuple, tmp_dir: str | Path | None = None
) -> GeneticAlgorithm:
    if tmp_dir is None:
        tmp_dir = Path(mkdtemp())

    problem = MISGaProblem(instance, config, sample)

    if config.recombination == "optimal":
        crossover = MISGACrossverOptimal(
            problem,
            instance,
            tournament_size=config.tournament_size,
            opt_recomb_time_limit=config.opt_recomb_time_limit,
        )
    else:
        crossover = MISGACrossover(
            problem,
            instance,
            mode=config.recombination,
            tournament_size=config.tournament_size,
        )

    return GeneticAlgorithm(
        problem=problem,
        popsize=config.pop_size,
        re_evaluate=False,
        operators=[
            crossover,
            MISGAMutation(
                problem,
                instance,
                deselect_prob=config.deselect_prob,
                optimal_recombination=config.recombination == "optimal",
                mutation_prob=config.mutation_prob,
            ),
            TempSaver(problem, tmp_dir / "population.txt"),
        ],
    )
