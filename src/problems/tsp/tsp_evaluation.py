from __future__ import annotations

import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
from problems.tsp.cython_merge.cython_merge import merge_cython, merge_cython_get_tour


def numpy_merge(points: np.ndarray, adj_mat: np.ndarray) -> tuple[np.ndarray, int]:
    """Currently unused. Supposed to be a numpy implementation of the cython merge function."""
    dists = np.linalg.norm(points[:, None] - points, axis=-1)

    components = np.zeros((adj_mat.shape[0], 2)).astype(int)
    components[:] = np.arange(adj_mat.shape[0])[..., None]
    real_adj_mat = np.zeros_like(adj_mat)
    merge_iterations = 0
    for edge in (-adj_mat / dists).flatten().argsort():
        merge_iterations += 1
        a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
        if not (a in components and b in components):
            continue
        ca = np.nonzero((components == a).sum(1))[0][0]
        cb = np.nonzero((components == b).sum(1))[0][0]
        if ca == cb:
            continue
        cca = sorted(components[ca], key=lambda x: x == a)
        ccb = sorted(components[cb], key=lambda x: x == b)
        newc = np.array([[cca[0], ccb[0]]])
        m, M = min(ca, cb), max(ca, cb)
        real_adj_mat[a, b] = 1
        components = np.concatenate([components[:m], components[m + 1 : M], components[M + 1 :], newc], 0)
        if len(components) == 1:
            break
    real_adj_mat[components[0, 1], components[0, 0]] = 1
    real_adj_mat += real_adj_mat.T
    return real_adj_mat, merge_iterations


def cython_merge(points: np.ndarray, adj_mat: np.ndarray) -> tuple[np.ndarray, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adj_matrix, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
        return np.asarray(adj_matrix), merge_iterations


def cython_merge_get_tour(points: np.ndarray, adj_mat: np.ndarray) -> tuple[np.ndarray, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tour, merge_iterations = merge_cython_get_tour(points.astype("double"), adj_mat.astype("double"))
        return np.asarray(tour), merge_iterations


def merge_tours(
    adj_mat: np.ndarray,
    np_points: np.ndarray,
    edge_index_np: np.ndarray,
    sparse_graph: bool = False,
    parallel_sampling: int = 1,
) -> tuple[list, float]:
    """
    Merge tours using the cython implementation of the merge function.

    Args:
        ajd_mat: P x N x N array of adjacency matrices. P parallel samples, N number of nodes.
        np_points: N x 2 array of node coordinates.
        edge_index_np: 2 x E array of edges. Only used if sparse_graph is True.
        parallel_sampling: Number of parallel samples to run (= P).

    Returns:
        tours: List of tours. Each tour is a list of node indices.
        merge_iterations: Average number of merge iterations across all samples.
    """
    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)

    if not sparse_graph:
        splitted_adj_mat = [adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat]
    else:
        splitted_adj_mat = [
            scipy.sparse.coo_matrix(
                (adj_mat, (edge_index_np[0], edge_index_np[1])),
            ).toarray()
            + scipy.sparse.coo_matrix(
                (adj_mat, (edge_index_np[1], edge_index_np[0])),
            ).toarray()
            for adj_mat in splitted_adj_mat
        ]

    splitted_points = [np_points for _ in range(parallel_sampling)]

    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
            results = p.starmap(
                cython_merge,
                zip(splitted_points, splitted_adj_mat),
            )
    else:
        results = [
            cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
        ]

    splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

    tours = []
    for i in range(parallel_sampling):
        tour = [0]
        while len(tour) < splitted_adj_mat[i].shape[0] + 1:
            n = np.nonzero(splitted_real_adj_mat[i][tour[-1]])[0]
            if len(tour) > 1:
                n = n[n != tour[-2]]
            tour.append(n.max())
        tours.append(tour)

    merge_iterations = np.mean(splitted_merge_iterations)
    return tours, merge_iterations


class TSPEvaluator:
    def __init__(self, points: np.ndarray) -> None:
        self.dist_mat = scipy.spatial.distance_matrix(points, points)

    def evaluate(self, route: np.array) -> float:
        total_cost = 0
        for i in range(len(route) - 1):
            total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost


class TSPTorchEvaluator:
    def __init__(self, points: torch.Tensor) -> None:
        self.dist_mat = torch.cdist(points, points).to(points.device)

    def evaluate(self, route: torch.Tensor) -> float:
        """Route is a tensor of size n+1, with the last value being the first value."""

        # Get the consecutive pairs of indices from the route (e.g., [route[i], route[i+1]])
        route_pairs = torch.stack([route[:-1], route[1:]], dim=1)

        # Use gather to select the distances between consecutive points along the route
        distances = self.dist_mat[route_pairs[:, 0], route_pairs[:, 1]]

        # Sum the distances to get the total route cost
        return distances.sum().item()


def evaluate_tsp_route_np(dist_mat: np.ndarray, route: np.ndarray) -> float:
    total_cost = 0
    for i in range(len(route) - 1):
        total_cost += dist_mat[route[i], route[i + 1]]
    return total_cost


def evaluate_tsp_route_torch(dist_mat: torch.Tensor, route: torch.Tensor) -> float:
    route_pairs = torch.stack([route[:-1], route[1:]], dim=1)
    distances = dist_mat[route_pairs[:, 0], route_pairs[:, 1]]
    return distances.sum().item()


def adj_mat_to_tour(adj_mat: np.ndarray) -> list:
    N = adj_mat.shape[0]
    tour = [0]
    while len(tour) < N + 1:
        n = np.nonzero(adj_mat[tour[-1]])[0]
        if len(tour) > 1:
            n = n[n != tour[-2]]
        tour.append(n.max())
    return tour


@torch.no_grad()
def cdist_v2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>

    device = x.device
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    x_sq_norm = x_cpu.pow(2).sum(dim=-1)
    y_sq_norm = y_cpu.pow(2).sum(dim=-1)
    x_dot_y = torch.matmul(x_cpu, y_cpu.t())
    del x_cpu, y_cpu

    # Compute distance in-place to avoid extra allocation
    dist = x_sq_norm.unsqueeze(dim=1) + y_sq_norm.unsqueeze(dim=0)
    dist.sub_(2 * x_dot_y)
    dist.clamp_(min=0.0)
    dist.sqrt_()  # In-place square root

    # Clean up intermediate tensors
    del x_sq_norm, y_sq_norm, x_dot_y

    dist = dist.detach()  # Ensure the tensor is detached from any graph
    return dist.to(device)
