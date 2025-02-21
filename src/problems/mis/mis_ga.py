from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import torch
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import CopyingOperator, CrossOver
from problems.mis.solve_optimal_recombination import solve_problem
from torch import no_grad

from difusco.sampler import DifuscoSampler

if TYPE_CHECKING:
    from config.myconfig import Config
    from problems.mis.mis_instance import MISInstance

from torch_geometric.data import Batch


class MISGaProblem(Problem):
    def __init__(self, instance: MISInstance, config: Config, sample: tuple) -> None:
        self.instance = instance
        self.config = config
        self.config.task = "mis"
        self.sample = sample

        if config.recombination == "difuscombination":
            self.sampler = self._get_difuscombination_sampler()
            self.batch = self._duplicate_batch(config, self.sample)
        else:
            self.sampler = None
            self.batch = None

        super().__init__(
            objective_func=instance.evaluate_solution,
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
            device="cuda",
            mode="difusco",
        )
        config = MISGaProblem._fake_paths_for_difusco_models(config)
        return DifuscoSampler(config)

    @staticmethod
    def _duplicate_batch(config: Config, batch: tuple) -> tuple:
        n = config.pop_size // 2

        # Duplicate the first tensor
        tensor0 = batch[0]  # e.g., shape: [1, ...]
        tensor0_dup = tensor0.repeat(n, 1)  # repeat along the first dimension

        # Handle the DataBatch in index 1
        # Convert it to a list of Data objects (should contain one element)
        data_list = batch[1].to_data_list()
        # Duplicate the single graph n times
        duplicated_data = [deepcopy(data_list[0]) for _ in range(n)]
        # Rebuild the DataBatch; this will automatically handle shifting node indices
        # and creating a new `batch` attribute
        new_data_batch = Batch.from_data_list(duplicated_data)

        # Duplicate the third tensor
        tensor2 = batch[2]  # e.g., shape: [1, ...]
        tensor2_dup = tensor2.repeat(n, 1)

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
        Values is a tensor of shape (n_solutions, solution_length).
        Initialization heuristic: (randomized) construction heuristic based on node degree.
        """
        degrees = self.instance.get_degrees()
        inversed_normalized_degrees = 1 - degrees / degrees.max()

        for i in range(values.shape[0]):
            std = 0.2 * i / values.shape[0]  # scales from 0 to 0.2
            noise = torch.randn(self.solution_length, device=self.device) * std
            values[i] = self.instance.get_feasible_from_individual(
                inversed_normalized_degrees + noise,
            )

    def _fill_difusco_sampling(self, values: torch.Tensor) -> None:
        """
        Values is a tensor of shape (n_solutions, solution_length).
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
        for i in range(popsize):
            values[i] = self.instance.get_feasible_from_individual(node_scores[i])


class MISGAMutation(CopyingOperator):
    """
    Gaussian mutation operator.

    Follows the algorithm description in:

        Sean Luke, 2013, Essentials of Metaheuristics, Lulu, second edition
        available for free at http://cs.gmu.edu/~sean/book/metaheuristics/
    """

    def __init__(self, problem: Problem, instance: MISInstance, deselect_prob: float = 0.05) -> None:
        """
        Mutation operator for the Maximum Independent Set problem. With probability deselect_prob, a selected node is
        unselected and gets a probabilty of zero. Solution is then made feasible.

        Args:
            problem: The problem object to work with.
            instance: The instance object to work with.
            deselect_prob: The probability of deselecting a selected node.
        """

        super().__init__(problem)
        self._instance = instance
        self._deselect_prob = deselect_prob

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        deselect_mask = torch.rand(data.shape, device=data.device, dtype=torch.float32) <= self._deselect_prob
        priorities = torch.rand(data.shape, device=data.device, dtype=torch.float32)
        priorities[deselect_mask.bool() & data.bool()] = 0

        for i in range(data.shape[0]):
            if deselect_mask[i].sum() > 0:
                data[i] = self._instance.get_feasible_from_individual(priorities[i])

        return result


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
        if self._mode == "optimal":
            return self._do_cross_over_optimal(parents1, parents2)

        raise ValueError(f"Invalid mode: {self._mode}")

    @no_grad()
    def _do_cross_over_optimal(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        """
        parents1 and parents2 are two solutions of shape (num_pairings, n_nodes).
        """
        num_pairings = parents1.shape[0]
        device = parents1.device

        children_1 = parents1.clone()
        children_2 = parents2.clone()

        for i in range(num_pairings):
            solution_1 = parents1[i].cpu().numpy()
            solution_2 = parents2[i].cpu().numpy()
            result = solve_problem(self._instance, solution_1, solution_2)

            children_1[i] = torch.tensor(result["children_np_labels"], device=device)
            children_2[i] = parents1[i].clone() if torch.rand(1) < 0.5 else parents2[i].clone()

        children = torch.cat([children_1, children_2], dim=0)
        return self._make_children_batch(children)

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
        heatmaps = self._problem.sampler.sample(self._problem.batch, features=features)
        assert heatmaps.shape == (
            num_pairings,
            2,
            self.problem.solution_length,
        ), f"Incorrect heatmaps shape: {heatmaps.shape}, expected (num_pairings, 2, solution_length)"

        # split into two children by dropping dimension 1 -> (num_pairings, solution_length)
        heatmaps_child1 = heatmaps.select(1, 0)
        heatmaps_child2 = heatmaps.select(1, 1)
        children1 = parents1.clone()
        children2 = parents2.clone()

        # finally, we need to make the heatmaps feasible
        for i in range(num_pairings):
            # Get feasible solutions based on priorities
            children1[i] = self._instance.get_feasible_from_individual(heatmaps_child1[i])
            children2[i] = self._instance.get_feasible_from_individual(heatmaps_child2[i])

        children = torch.cat([children1, children2], dim=0)
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

        children1 = parents1.clone()
        children2 = parents2.clone()

        for i in range(num_pairings):
            # Get feasible solutions based on priorities
            children1[i] = self._instance.get_feasible_from_individual(priority1[i])
            children2[i] = self._instance.get_feasible_from_individual(priority2[i])

        # Combine children into final result
        children = torch.cat([children1, children2], dim=0)

        return self._make_children_batch(children)


def create_mis_ga(instance: MISInstance, config: Config, sample: tuple) -> GeneticAlgorithm:
    problem = MISGaProblem(instance, config, sample)

    return GeneticAlgorithm(
        problem=problem,
        popsize=config.pop_size,
        re_evaluate=False,
        operators=[
            MISGACrossover(problem, instance, mode=config.recombination),
            MISGAMutation(problem, instance, deselect_prob=0.2),
        ],
    )
