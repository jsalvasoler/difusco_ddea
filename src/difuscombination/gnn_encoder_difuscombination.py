from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from difusco.gnn_encoder import (
    GNNEncoder,
    PositionEmbeddingSine,
    ScalarEmbeddingSine,
    ScalarEmbeddingSine1D,
)
from difusco.nn_utils import timestep_embedding

if TYPE_CHECKING:
    from config.myconfig import Config


class GNNEncoderDifuscombination(GNNEncoder):
    def __init__(self, config: Config) -> None:
        sparse = bool(config.sparse_factor > 0 or config.node_feature_only)
        super().__init__(
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            out_channels=1 if config.diffusion_type == "gaussian" else 2,
            aggregation=config.aggregation,
            norm="layer",
            learn_norm=True,
            track_norm=False,
            gated=True,
            sparse=sparse,
            use_activation_checkpoint=config.use_activation_checkpoint,
            node_feature_only=config.node_feature_only,
        )

        self.node_embed = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.node_embed_feat0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.node_embed_feat1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        if not self.node_feature_only:
            self.pos_embed = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)
            self.pos_embed_feat0 = PositionEmbeddingSine(
                self.hidden_dim // 2, normalize=True
            )
            self.pos_embed_feat1 = PositionEmbeddingSine(
                self.hidden_dim // 2, normalize=True
            )

            self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dim, normalize=False)
        else:
            self.pos_embed = ScalarEmbeddingSine1D(self.hidden_dim, normalize=False)
            self.pos_embed_feat0 = ScalarEmbeddingSine1D(
                self.hidden_dim, normalize=False
            )
            self.pos_embed_feat1 = ScalarEmbeddingSine1D(
                self.hidden_dim, normalize=False
            )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        features: torch.Tensor,
        graph: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.node_feature_only:
            if self.sparse:
                return self.sparse_forward_node_feature_only(
                    x, features, timesteps, edge_index
                )
            error_msg = "Dense node feature only is not supported"
            raise NotImplementedError(error_msg)
        # TODO: implement features for other cases
        raise NotImplementedError(
            "Features are not supported for non-node feature only cases"
        )
        if self.sparse:
            return self.sparse_forward(x, graph, timesteps, edge_index)
        return self.dense_forward(x, graph, timesteps, edge_index)

    def sparse_forward_node_feature_only(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
        timesteps: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Assume x is of shape (num_nodes,), features is of shape (num_nodes, 2)"""
        x = self.node_embed(self.pos_embed(x))
        x += self.node_embed_feat0(self.pos_embed_feat0(features[:, 0]))
        x += self.node_embed_feat1(self.pos_embed_feat1(features[:, 1]))

        x_shape = x.shape
        e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
        edge_index = edge_index.long()

        x, e = self.sparse_encoding(x, e, edge_index, time_emb)
        x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
        return self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
