from __future__ import annotations

import math
import os
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import torch
from config.mytable import TableSaver
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.decorators import vectorized
from evotorch.operators import CopyingOperator, CrossOver
from problems.mis.solve_optimal_recombination import (
    OptimalRecombResults,
    solve_local_branching_mis,
)
from torch import no_grad
from torch_geometric.data import Batch

from difusco.sampler import DifuscoSampler

if TYPE_CHECKING:
    import pandas as pd
    from config.myconfig import Config
    from problems.mis.mis_instance import MISInstance


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
        return config.update(
            test_graphs_dir=config.test_split,
            training_samples_file=config.test_samples_file,
            training_labels_dir=config.test_labels_dir,
            training_graphs_dir=config.test_split,
            validation_samples_file=config.test_samples_file,
            validation_labels_dir=config.test_labels_dir,
            validation_graphs_dir=config.test_split,
        )

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
        assert popsize == values.shape[0], (
            "Population size must match the number of solutions"
        )
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
        preserve_optimal_recombination: bool = False,
    ) -> None:
        """
        Mutation operator for the Maximum Independent Set problem. With probability deselect_prob, a selected node is
        unselected and gets a probability of zero. Solution is then made feasible.
        Only applies mutation with probability mutation_prob. When preserve_preserve_optimal_recombination is True,
        mutation is not applied to the first half of the population.

        Args:
            problem: The problem object to work with.
            instance: The instance object to work with.
            deselect_prob: The probability of deselecting a selected node.
            mutation_prob: The probability of mutating a solution.
            preserve_optimal_recombination: Whether optimal recombination is used.
                In this case, the mutation is not applied for the first half of the population.
        """

        super().__init__(problem)
        self._instance = instance
        self._deselect_prob = deselect_prob
        self._mutation_prob = mutation_prob
        self._preserve_optimal_recombination = preserve_optimal_recombination

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()

        pop_size, n_nodes = data.shape

        # Decide which individuals to mutate
        mutation_mask = (
            torch.rand(pop_size, device=data.device, dtype=torch.float16)
            <= self._mutation_prob
        )
        if self._preserve_optimal_recombination:
            # Skip mutation for the first half of the population
            mutation_mask[: pop_size // 2] = False

        # If no individuals are selected for mutation, exit early.
        if not mutation_mask.any():
            return result

        # Get indices of individuals to mutate.
        mutate_indices = mutation_mask.nonzero(as_tuple=True)[0]

        # Generate deselect mask and priorities only for the solutions that are going to be mutated.
        sub_data = data[mutate_indices]
        deselect_mask = (
            torch.rand(sub_data.shape, device=data.device, dtype=torch.float32)
            <= self._deselect_prob
        )
        priorities = torch.rand(sub_data.shape, device=data.device, dtype=torch.float32)

        # For nodes that are both deselected and originally selected, set priority to zero.
        priorities[deselect_mask & sub_data.bool()] = 0

        # Only update those solutions that have at least one deselected node.
        update_mask = deselect_mask.sum(dim=-1) > 0
        if update_mask.any():
            indices_to_update = mutate_indices[update_mask]
            feasible = self._instance.get_feasible_from_individual_batch(
                priorities[update_mask]
            )
            data[indices_to_update] = feasible

        return result


class LocalBranchingSolver:
    """Solver for the recombination. Cache results for the same instance."""

    # Class-level cache dictionary with ClassVar annotation
    _cache: ClassVar[dict[tuple, OptimalRecombResults]] = {}

    def __init__(self, instance: MISInstance) -> None:
        self.instance = instance
        # Instance-specific cache key prefix (based on instance identity)
        self._instance_id = id(instance)

    def solve(
        self, solution_1: tuple, solution_2: tuple, time_limit: int = 60, **kwargs
    ) -> OptimalRecombResults:
        # Create a cache key that includes the instance ID and all parameters
        k_factor = kwargs.get("k_factor", None)
        cache_key = (self._instance_id, solution_1, solution_2, time_limit, k_factor)

        # Check if result is in cache
        if cache_key in self._cache:
            print("using cached result")
            return self._cache[cache_key]

        # Convert tuples back to np.array
        solution_1_array = np.array(solution_1)
        solution_2_array = np.array(solution_2)

        # Call the original function
        result = solve_local_branching_mis(
            self.instance,
            solution_1_array,
            solution_2_array,
            time_limit=time_limit,
            **kwargs,
        )

        # Store in cache
        self._cache[cache_key] = result
        return result

    @classmethod
    def clear_cache(cls: type[LocalBranchingSolver]) -> None:
        """Clear the class-level cache."""
        cls._cache.clear()


class MISGACrossverOptimal(CrossOver):
    def __init__(
        self,
        problem: MISGaProblem,
        instance: MISInstance,
        tmp_dir: Path,
        tournament_size: int = 2,
        opt_recomb_time_limit: int = 15,
    ) -> None:
        super().__init__(
            problem,
            tournament_size=tournament_size,
        )
        self._instance = instance
        self._tmp_dir = tmp_dir
        self._opt_recomb_time_limit = opt_recomb_time_limit
        self.solver = LocalBranchingSolver(instance)  # Initialize the solver

        # create a TableSaver object to save the results of the optimal recombination
        self.table_saver = TableSaver(self._tmp_dir / "optimal_recombination.csv")

    @torch.no_grad()
    def _do_cross_over(
        self, parents1: torch.Tensor, parents2: torch.Tensor
    ) -> SolutionBatch:
        return self._do_cross_over_optimal(parents1, parents2)

    @no_grad()
    def _do_cross_over_optimal(
        self, parents1: torch.Tensor, parents2: torch.Tensor
    ) -> SolutionBatch:
        """
        parents1 and parents2 are two solutions of shape (num_pairings, n_nodes).
        """
        num_pairings = parents1.shape[0]
        device = parents1.device

        children_1 = parents1.clone()
        children_2 = parents2.clone()

        for i in range(num_pairings):
            solution_1 = tuple(parents1[i].cpu().numpy().nonzero()[0])
            solution_2 = tuple(parents2[i].cpu().numpy().nonzero()[0])

            result = self.solver.solve(
                solution_1,
                solution_2,
                time_limit=self._opt_recomb_time_limit,
                k_factor=1.75,
            )

            save_in_dict = {
                "parent_1": ",".join(map(str, solution_1)),
                "parent_2": ",".join(map(str, solution_2)),
                "children": ",".join(map(str, result.children)),
                "instance_id": self._problem.sample[0].item(),
                "runtime": result.runtime,
            }
            self.table_saver.put(save_in_dict)

            children_1[i] = torch.tensor(result.children_np_labels, device=device)
            children_2[i] = (
                parents1[i].clone() if torch.rand(1) < 0.5 else parents2[i].clone()
            )

        children = torch.cat([children_1, children_2], dim=0)
        return self._make_children_batch(children)


class MISGACrossover(CrossOver):
    def __init__(
        self,
        problem: MISGaProblem,
        instance: MISInstance,
        tournament_size: int = 2,
        mode: Literal["classic", "difuscombination"] = "classic",
    ) -> None:
        super().__init__(problem, tournament_size=tournament_size)
        self._instance = instance
        self._mode = mode

    @no_grad()
    def _do_cross_over(
        self, parents1: torch.Tensor, parents2: torch.Tensor
    ) -> SolutionBatch:
        if self._mode == "classic":
            return self._do_cross_over_classic(parents1, parents2)
        if self._mode == "difuscombination":
            return self._do_cross_over_difuscombination(parents1, parents2)

        raise ValueError(f"Invalid mode: {self._mode}")

    @no_grad()
    def _do_cross_over_difuscombination(
        self, parents1: torch.Tensor, parents2: torch.Tensor
    ) -> SolutionBatch:
        """
        parents1 and parents2 are two solutions of shape (num_pairings, n_nodes).
        DifuscSampling parameters:
        - parallel_sampling: 2
        - sequential_sampling: 1
        - batch_size: num_pairings
        """
        num_pairings = parents1.shape[0]

        features = torch.stack([parents1, parents2], dim=2)
        assert features.shape == (num_pairings, self.problem.solution_length, 2), (
            "Incorrect features shape"
        )
        # we need to reshape the features to (num_pairings * n_nodes, 2)
        features = features.reshape(num_pairings * self.problem.solution_length, 2)
        assert features.shape == (num_pairings * self.problem.solution_length, 2), (
            "Incorrect features shape"
        )
        heatmaps = self._problem.sampler.sample(
            self._problem.batch, features=features
        ).to(self.problem.device)
        assert heatmaps.shape == (
            num_pairings,
            2,
            self.problem.solution_length,
        ), (
            f"Incorrect heatmaps shape: {heatmaps.shape}, expected (num_pairings, 2, solution_length)"
        )

        # split into two children by dropping dimension 1 -> (num_pairings, solution_length)
        heatmaps_child1 = heatmaps.select(1, 0)
        heatmaps_child2 = heatmaps.select(1, 1)

        heatmaps_child = torch.cat([heatmaps_child1, heatmaps_child2], dim=0)
        children = self._instance.get_feasible_from_individual_batch(heatmaps_child)
        return self._make_children_batch(children)

    @no_grad()
    def _do_cross_over_classic(
        self, parents1: torch.Tensor, parents2: torch.Tensor
    ) -> SolutionBatch:
        """
        Parents are two solutions of shape (num_pairings, n_nodes).
        """
        num_pairings = parents1.shape[0]
        device = parents1.device

        # Find common nodes between parents
        common_nodes = (parents1 & parents2).bool()  # Element-wise AND

        # Random values between 0 and 0.5, 1 if node forced to selection
        priority1 = (
            torch.rand(
                num_pairings,
                self.problem.solution_length,
                device=device,
                dtype=torch.float32,
            )
            * 0.5
        )
        priority1[common_nodes] = 1
        # Random values between 0.5 and 1, 0 if node penalized for selection
        priority2 = (
            torch.rand(
                num_pairings,
                self.problem.solution_length,
                device=device,
                dtype=torch.float32,
            )
            * 0.5
            + 0.5
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
        data = batch.values
        # for each solution in the batch, take the non-zero indices
        solutions_str = []
        for i in range(data.shape[0]):
            indices = data[i].nonzero().flatten().tolist()
            solutions_str.append(",".join(map(str, indices)))
        return " | ".join(solutions_str)

    def save(self, batch: SolutionBatch) -> None:
        with open(self._tmp_file, "a") as f:
            f.write(
                self._get_population_string(batch) + "\n"
            )  # Added newline for better readability


DEFAULT_PARAMETERS = {
    "elite_ratio": 0.05,
    "tournament_size": 2,
    "selection_method": "best_unique",  # Can be "tournament", "roulette", or "best_unique"
}


class MISGA(GeneticAlgorithm):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002
        self._tmp_dir = kwargs.pop("tmp_dir", Path(mkdtemp()))
        self._elite_ratio = kwargs.pop("elite_ratio", DEFAULT_PARAMETERS["elite_ratio"])
        self._tournament_size = kwargs.pop(
            "tournament_size", DEFAULT_PARAMETERS["tournament_size"]
        )
        self._selection_method = kwargs.pop(
            "selection_method", DEFAULT_PARAMETERS["selection_method"]
        )
        print(f"selection_method: {self._selection_method}")
        super().__init__(*args, **kwargs)
        self._temp_saver = TempSaver(self._problem, self._tmp_dir / "population.txt")

    def get_recombination_saved_results(self) -> pd.DataFrame | None:
        try:
            return self._operators[0].table_saver.get()
        except AttributeError:
            return None

    @torch.no_grad()
    def _take_tournament(self, batch: SolutionBatch, popsize: int) -> SolutionBatch:
        """
        Selects the next generation using a combination of elitism and tournament selection.

        This method first preserves a small percentage of the best-performing ("elite")
        solutions from the combined parent-child population. The remainder of the
        new population is then filled by individuals who win a series of stochastic
        "tournaments".

        This balanced approach ensures that the best solutions are not lost, while
        the tournament provides selection pressure that is less greedy than pure
        truncation, helping to maintain genetic diversity and prevent premature
        convergence.

        Args:
            batch: The combined population of parents and children.
            popsize: The desired size of the next generation.

        Returns:
            A new SolutionBatch representing the selected next generation.
        """
        device = batch.device

        # 1. ELITISM
        # Calculate the number of elite individuals to carry over to the next generation.
        num_elites = math.ceil(popsize * self._elite_ratio)

        # Get the indices of the best individuals from the extended population.
        # `argsort` sorts in ascending order, so for problems where higher values are better,
        # we would need to adjust. Assuming `argsort` correctly identifies the best solutions
        # as is conventional in evotorch utilities.
        all_sorted_indices = batch.argsort()
        elite_indices = all_sorted_indices[:num_elites]

        # 2. TOURNAMENT SELECTION
        # Calculate how many individuals we need to select via tournament.
        num_tournament_winners = popsize - num_elites

        # For single-objective, rank solutions from -0.5 (worst) to 0.5 (best).
        ranks = batch.utility(ranking_method="centered")

        # Create random tournaments. Each row is a tournament, and each column is a participant's index.
        tournament_indices = self._problem.make_randint(
            (num_tournament_winners, self._tournament_size), n=len(batch), device=device
        )

        # Get the ranks of the competitors for each tournament
        tournament_ranks = ranks[tournament_indices]

        # For each tournament (row), find the index of the competitor with the maximum rank.
        winner_indices_in_tournament = torch.argmax(tournament_ranks, dim=-1)

        # Get the indices of the winning solutions from the original batch
        tournament_rows = torch.arange(0, num_tournament_winners, device=device)
        tournament_winner_indices = tournament_indices[
            tournament_rows, winner_indices_in_tournament
        ]

        # 3. COMBINE AND CREATE NEW POPULATION
        # Concatenate the indices from elitism and tournament selection
        final_indices = torch.cat([elite_indices, tournament_winner_indices])
        assert final_indices.shape[0] == popsize, (
            f"Final indices shape: {final_indices.shape}, expected {popsize}"
        )

        # Create the new population from the final list of selected indices
        return SolutionBatch(slice_of=(batch, final_indices.tolist()))

    @torch.no_grad()
    def _roulette(
        self, extended_population: SolutionBatch, popsize: int
    ) -> SolutionBatch:
        """
        Selects the next generation using a combination of elitism and roulette wheel selection.

        Args:
            extended_population: The combined population of parents and children.
            popsize: The desired size of the next generation.

        Returns:
            A new SolutionBatch representing the selected next generation.
        """
        # 1. ELITISM
        # Calculate the number of elite individuals to carry over.
        num_elites = math.ceil(popsize * self._elite_ratio)

        # If the number of elites is greater than or equal to the population size,
        # simply select the best `popsize` individuals.
        if num_elites >= popsize:
            best_indices = extended_population.argsort()[:popsize]
            return SolutionBatch(slice_of=(extended_population, best_indices.tolist()))

        # Get the indices of the best individuals (the elites).
        all_sorted_indices = extended_population.argsort()
        elite_indices = all_sorted_indices[:num_elites]

        # 2. ROULETTE WHEEL SELECTION
        # Determine how many individuals need to be selected via the roulette wheel.
        num_roulette_winners = popsize - num_elites

        # Get the fitness values for the entire extended population.
        # These values will serve as weights for the multinomial sampling.
        # For this problem, fitness (the sum) is always non-negative.
        fitness_values = extended_population.evals.float()

        # Perform roulette wheel selection using multinomial sampling.
        roulette_indices = torch.multinomial(
            input=(fitness_values / fitness_values.sum()).flatten(),
            num_samples=num_roulette_winners,
            replacement=False,
        )

        # 3. COMBINE AND CREATE THE NEW POPULATION
        # Concatenate the indices from elitism and roulette selection.
        final_indices = torch.cat([elite_indices, roulette_indices])
        assert final_indices.shape[0] == popsize, (
            f"Final indices shape: {final_indices.shape}, expected {popsize}"
        )

        # Create the new population from the final list of selected indices.
        return SolutionBatch(slice_of=(extended_population, final_indices.tolist()))

    def _step(self) -> None:
        # Get the population size
        popsize = self._popsize

        # Produce and get an extended population in a single SolutionBatch
        extended_population = self._make_extended_population(split=False)

        # Select the next generation based on the selection method
        if self._selection_method == "best_unique":
            self._population = self._take_best_unique(extended_population, popsize)
        elif self._selection_method == "roulette":
            self._population = self._roulette(extended_population, popsize)
        elif self._selection_method == "tournament":
            self._population = self._take_tournament(extended_population, popsize)
        else:
            raise ValueError(f"Invalid selection method: {self._selection_method}")
        print(f"population: {self._population.values.sum(dim=-1)}")

        # Save population stats to file
        self._temp_saver.save(self._population)

    def _take_best_unique(
        self, extended_population: SolutionBatch, popsize: int
    ) -> SolutionBatch:
        sorted_indices = extended_population.argsort().tolist()

        # get unique indices
        unique_indices = get_unique_indices(extended_population.values)
        num_unique = unique_indices.shape[0]

        # sort unique indices by objective value, i.e., by the sorted_indices.index(idx)
        unique_indices_sorted = sorted(
            unique_indices, key=lambda idx: sorted_indices.index(idx)
        )
        unique_indices_sorted_torch = torch.tensor(
            unique_indices_sorted, device=extended_population.device
        )

        # Repeat indices to match population size
        k = popsize // num_unique
        remainder = popsize % num_unique

        final_indices = torch.cat(
            [
                unique_indices_sorted_torch.repeat(k),
                unique_indices_sorted_torch[:remainder],
            ]
        )
        return SolutionBatch(slice_of=(extended_population, final_indices.tolist()))


def get_unique_indices(t: torch.Tensor) -> torch.Tensor:
    """Returns the indices of the first occurrences of unique rows in a 2D tensor, preserving order."""
    n = t.size(0)
    unique_mask = torch.ones(n, dtype=torch.bool, device=t.device)
    for i in range(n - 1):
        if unique_mask[i]:
            unique_mask[i + 1 :] &= ~(t[i + 1 :] == t[i]).all(dim=1)
    return torch.nonzero(unique_mask).squeeze(dim=1)


def create_mis_ga(
    instance: MISInstance,
    config: Config,
    sample: tuple,
    tmp_dir: str | Path | None = None,
) -> MISGA:
    if tmp_dir is None:
        tmp_dir = Path(mkdtemp())

    problem = MISGaProblem(instance, config, sample)

    if config.recombination == "optimal":
        crossover = MISGACrossverOptimal(
            problem,
            instance,
            tournament_size=config.tournament_size,
            opt_recomb_time_limit=config.opt_recomb_time_limit,
            tmp_dir=tmp_dir,
        )
    else:
        crossover = MISGACrossover(
            problem,
            instance,
            mode=config.recombination,
            tournament_size=config.tournament_size,
        )

    new_kwargs = {
        "problem": problem,
        "popsize": config.pop_size,
        "re_evaluate": False,
        "operators": [
            crossover,
            MISGAMutation(
                problem,
                instance,
                deselect_prob=config.deselect_prob,
                preserve_optimal_recombination=config.preserve_optimal_recombination,
                mutation_prob=config.mutation_prob,
            ),
        ],
        "tmp_dir": tmp_dir,
    }

    if "selection_method" in config:
        new_kwargs["selection_method"] = config.selection_method

    return MISGA(**new_kwargs)
