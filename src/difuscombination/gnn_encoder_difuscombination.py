import torch
from config.myconfig import Config
from torch import nn

from difusco.gnn_encoder import GNNEncoder, PositionEmbeddingSine, ScalarEmbeddingSine, ScalarEmbeddingSine1D
from difusco.nn_utils import timestep_embedding


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
        self.node_embed_feat1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.node_embed_feat2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        if not self.node_feature_only:
            self.pos_embed = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)
            self.pos_embed_feat1 = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)
            self.pos_embed_feat2 = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)

            self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dim, normalize=False)
        else:
            self.pos_embed = ScalarEmbeddingSine1D(self.hidden_dim, normalize=False)
            self.pos_embed_feat1 = ScalarEmbeddingSine1D(self.hidden_dim, normalize=False)
            self.pos_embed_feat2 = ScalarEmbeddingSine1D(self.hidden_dim, normalize=False)

    def sparse_forward_node_feature_only(
        self, x: torch.Tensor, timesteps: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        x0 = self.node_embed(self.pos_embed(x[:, 0]))
        x1 = self.node_embed_feat1(self.pos_embed_feat1(x[:, 1]))
        x2 = self.node_embed_feat2(self.pos_embed_feat2(x[:, 2]))
        x = x0 + x1 + x2

        x_shape = x.shape
        e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
        edge_index = edge_index.long()

        x, e = self.sparse_encoding(x, e, edge_index, time_emb)
        x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
        return self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
