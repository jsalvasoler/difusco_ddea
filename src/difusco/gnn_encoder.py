from __future__ import annotations

import functools
from math import pi
from typing import TYPE_CHECKING

import torch
import torch.utils.checkpoint as activation_checkpoint
from torch import nn
from torch.nn.functional import relu
from torch_sparse import SparseTensor
from torch_sparse import max as sparse_max
from torch_sparse import mean as sparse_mean
from torch_sparse import sum as sparse_sum

from difusco.nn_utils import normalization, timestep_embedding, zero_module

if TYPE_CHECKING:
    import torch_sparse


class GNNLayer(nn.Module):
    """Configurable GNN Layer
    Implements the Gated Graph ConvNet layer:
        h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
        sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
        e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
        where Aggr. is an aggregation function: sum/mean/max.
    References:
        - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs.
        In International Conference on Learning Representations, 2018.
        - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson.
        Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
    """

    def __init__(
        self,
        hidden_dim: int,
        aggregation: int = "sum",
        norm: int | None = "batch",
        learn_norm: bool = True,
        track_norm: bool = False,
        gated: bool = True,
    ) -> None:
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
            gated: Whether to use edge gating (True/False)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        assert self.gated, "Use gating with GCN, pass the `--gated` flag"

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm),
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm),
        }.get(self.norm, None)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        graph: torch.Tensor | torch_sparse.SparseTensor,
        mode: str = "residual",
        edge_index: torch.Tensor | None = None,
        sparse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            In Dense version:
              h: Input node features (B x V x H)
              e: Input edge features (B x V x V x H)
              graph: Graph adjacency matrices (B x V x V)
              mode: str
            In Sparse version:
              h: Input node features (V x H)
              e: Input edge features (E x H)
              graph: torch_sparse.SparseTensor
              mode: str
              edge_index: Edge indices (2 x E)
            sparse: Whether to use sparse tensors (True/False)
        Returns:
            Updated node and edge features
        """
        if not sparse:
            batch_size, num_nodes, hidden_dim = h.shape
        else:
            batch_size = None
            num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H

        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1) if not sparse else self.V(h[edge_index[1]])

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H, source
        Bh = self.B(h)  # B x V x H, target
        Ce = self.C(e)  # B x V x V x H / E x H

        # Update edge features and compute edge gates
        # not sparse -> B x V x V x H, sparse -> E x H
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce if not sparse else Ah[edge_index[1]] + Bh[edge_index[0]] + Ce

        gates = torch.sigmoid(e)  # B x V x V x H / E x H

        # Update node features
        h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index, sparse=sparse)  # B x V x H

        # Normalize node features
        if not sparse:
            h = (
                self.norm_h(h.view(batch_size * num_nodes, hidden_dim)).view(batch_size, num_nodes, hidden_dim)
                if self.norm_h
                else h
            )
        else:
            h = self.norm_h(h) if self.norm_h else h

        # Normalize edge features
        if not sparse:
            e = (
                self.norm_e(e.view(batch_size * num_nodes * num_nodes, hidden_dim)).view(
                    batch_size, num_nodes, num_nodes, hidden_dim
                )
                if self.norm_e
                else e
            )
        else:
            e = self.norm_e(e) if self.norm_e else e

        # Apply non-linearity
        h = relu(h)
        e = relu(e)

        # Make residual connection
        if mode == "residual":
            h = h_in + h
            e = e_in + e

        return h, e

    def aggregate(
        self,
        Vh: torch.Tensor,
        graph: torch.Tensor | torch_sparse.SparseTensor,
        gates: torch.Tensor,
        mode: str | None = None,
        edge_index: torch.Tensor | None = None,
        sparse: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            In Dense version:
              Vh: Neighborhood features (B x V x V x H)
              graph: Graph adjacency matrices (B x V x V)
              gates: Edge gates (B x V x V x H)
              mode: str
            In Sparse version:
              Vh: Neighborhood features (E x H)
              graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
              gates: Edge gates (E x H)
              mode: str
              edge_index: Edge indices (2 x E)
            sparse: Whether to use sparse tensors (True/False)
        Returns:
            Aggregated neighborhood features (B x V x H)
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H

        # Enforce graph structure through masking
        # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

        # Aggregate neighborhood features
        if not sparse:
            if (mode or self.aggregation) == "mean":
                return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
            if (mode or self.aggregation) == "max":
                return torch.max(Vh, dim=2)[0]
            return torch.sum(Vh, dim=2)

        sparseVh = SparseTensor(
            row=edge_index[0], col=edge_index[1], value=Vh, sparse_sizes=(graph.size(0), graph.size(1))
        )

        if (mode or self.aggregation) == "mean":
            return sparse_mean(sparseVh, dim=1)

        if (mode or self.aggregation) == "max":
            return sparse_max(sparseVh, dim=1)

        return sparse_sum(sparseVh, dim=1)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is All You Need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: float | None = None
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            error_msg = "Normalize should be True if scale is passed"
            raise ValueError(error_msg)
        if scale is None:
            scale = 2 * pi
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_embed = x[:, :, 0]
        x_embed = x[:, :, 1]
        if self.normalize:
            # eps = 1e-6
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode="trunc")) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x), dim=2).contiguous()


class ScalarEmbeddingSine(nn.Module):
    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: float | None = None
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            error_msg = "Normalize should be True if scale is passed"
            raise ValueError(error_msg)
        if scale is None:
            scale = 2 * pi
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_embed = x
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        return torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)


class ScalarEmbeddingSine1D(nn.Module):
    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: float | None = None
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            error_msg = "Normalize should be True if scale is passed"
            raise ValueError(error_msg)
        if scale is None:
            scale = 2 * pi
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_embed = x
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        return torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)


def run_sparse_layer(
    layer: callable,
    time_layer: callable,
    out_layer: callable,
    adj_matrix: torch.Tensor,
    edge_index: torch.Tensor,
    add_time_on_edge: bool = True,
) -> callable:
    def custom_forward(*inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = inputs[0]
        e_in = inputs[1]
        time_emb = inputs[2]
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
        if add_time_on_edge:
            e = e + time_layer(time_emb)
        else:
            x = x + time_layer(time_emb)
        x = x_in + x
        e = e_in + out_layer(e)
        return x, e

    return custom_forward


class GNNEncoder(nn.Module):
    """Configurable GNN Encoder"""

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        out_channels: int = 1,
        aggregation: str = "sum",
        norm: str = "layer",
        learn_norm: bool = True,
        track_norm: bool = False,
        gated: bool = True,
        sparse: bool = False,
        use_activation_checkpoint: bool = False,
        node_feature_only: bool = False,
    ) -> None:
        super().__init__()
        self.sparse = sparse
        self.node_feature_only = node_feature_only
        self.hidden_dim = hidden_dim
        time_embed_dim = hidden_dim // 2
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

        if not node_feature_only:
            self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
            self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
        else:
            self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.out = nn.Sequential(
            normalization(hidden_dim), nn.ReLU(), nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        )

        self.layers = nn.ModuleList(
            [GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated) for _ in range(n_layers)]
        )

        self.time_embed_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(time_embed_dim, hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )

        self.per_layer_out = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
                    nn.SiLU(),
                    zero_module(nn.Linear(hidden_dim, hidden_dim)),
                )
                for _ in range(n_layers)
            ]
        )
        self.use_activation_checkpoint = use_activation_checkpoint

    def dense_forward(
        self, x: torch.Tensor, graph: torch.Tensor, timesteps: torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input node coordinates (B x V x 2)
            graph: Graph adjacency matrices (B x V x V)
            timesteps: Input node timesteps (B)
            edge_index: Edge indices (2 x E)
        Returns:
            Updated edge features (B x V x V)
        """
        del edge_index
        x = self.node_embed(self.pos_embed(x))
        e = self.edge_embed(self.edge_pos_embed(graph))
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
        graph = torch.ones_like(graph).long()

        for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
            x_in, e_in = x, e

            if self.use_activation_checkpoint:
                raise NotImplementedError

            x, e = layer(x, e, graph, mode="direct")
            if not self.node_feature_only:
                e = e + time_layer(time_emb)[:, None, None, :]
            else:
                x = x + time_layer(time_emb)[:, None, :]
            x = x_in + x
            e = e_in + out_layer(e)

        return self.out(e.permute((0, 3, 1, 2)))

    def sparse_forward(
        self, x: torch.Tensor, graph: torch.Tensor, timesteps: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input node coordinates (V x 2)
            graph: Graph edge features (E)
            timesteps: Input edge timestep features (E)
            edge_index: Adjacency matrix for the graph (2 x E)
        Returns:
            Updated edge features (E x H)
        """
        x = self.node_embed(self.pos_embed(x.unsqueeze(0)).squeeze(0))
        e = self.edge_embed(self.edge_pos_embed(graph.expand(1, 1, -1)).squeeze())
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
        edge_index = edge_index.long()

        x, e = self.sparse_encoding(x, e, edge_index, time_emb)
        return e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))

    def sparse_forward_node_feature_only(
        self, x: torch.Tensor, timesteps: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        x = self.node_embed(self.pos_embed(x))
        x_shape = x.shape
        e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
        edge_index = edge_index.long()

        x, e = self.sparse_encoding(x, e, edge_index, time_emb)
        x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
        return self.out(x).reshape(-1, x_shape[0]).permute((1, 0))

    def sparse_encoding(
        self, x: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor, time_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adj_matrix = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=torch.ones_like(edge_index[0].float()),
            sparse_sizes=(x.shape[0], x.shape[0]),
        )
        adj_matrix = adj_matrix.to(x.device)

        for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
            x_in, e_in = x, e

            if self.use_activation_checkpoint:
                single_time_emb = time_emb[:1]

                run_sparse_layer_fn = functools.partial(run_sparse_layer, add_time_on_edge=not self.node_feature_only)

                out = activation_checkpoint.checkpoint(
                    run_sparse_layer_fn(layer, time_layer, out_layer, adj_matrix, edge_index),
                    x_in,
                    e_in,
                    single_time_emb,
                )
                x = out[0]
                e = out[1]
            else:
                x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
                if not self.node_feature_only:
                    e = e + time_layer(time_emb)
                else:
                    x = x + time_layer(time_emb)
                x = x_in + x
                e = e_in + out_layer(e)
        return x, e

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        graph: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.node_feature_only:
            if self.sparse:
                return self.sparse_forward_node_feature_only(x, timesteps, edge_index)
            error_msg = "Dense node feature only is not supported"
            raise NotImplementedError(error_msg)
        if self.sparse:
            return self.sparse_forward(x, graph, timesteps, edge_index)
        return self.dense_forward(x, graph, timesteps, edge_index)
