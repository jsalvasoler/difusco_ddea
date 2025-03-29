from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding, BatchNorm1d, Sequential, ReLU, ModuleList
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.conv import GPSConv
from torch_geometric.datasets import ZINC
from problems.mis.mis_dataset import MISDataset
import torch_geometric.transforms as T
from typing import Any, Optional


class PerformerAttention:
    """Implementation of Performer Attention for efficient attention computation."""
    
    def __init__(self):
        # Placeholder for initialization
        pass
    
    def redraw_projection_matrix(self) -> None:
        """Redraws projection matrices in performer attention."""
        # Placeholder for redrawing projection matrices
        pass


class RedrawProjection:
    """Helper class to handle redrawing projections for attention mechanisms."""
    
    def __init__(self, model: torch.nn.Module,
                redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self) -> None:
        """Redraws projection matrices for performer attention if necessary."""
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class GPS(torch.nn.Module):
    """Graph Positional Embedding with self-attention and message passing layers."""
    
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                attn_type: str, attn_kwargs: dict[str, Any], 
                heads: int = 4, node_features: int = 28, edge_features: int = 4,
                walk_length: int = 20, dropout: float = 0.0):
        super().__init__()

        self.node_emb = Embedding(node_features, channels - pe_dim)
        self.pe_lin = Linear(walk_length, pe_dim)
        self.pe_norm = BatchNorm1d(walk_length)
        self.edge_emb = Embedding(edge_features, channels)
        self.dropout = dropout

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=heads,
                        attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        # MLP for final prediction
        mlp_layers = []
        prev_dim = channels
        for i, dim_factor in enumerate([2, 4]):
            next_dim = channels // dim_factor
            mlp_layers.append(Linear(prev_dim, next_dim))
            mlp_layers.append(ReLU())
            if self.dropout > 0 and i == 0:  # Only add dropout after first layer
                mlp_layers.append(torch.nn.Dropout(self.dropout))
            prev_dim = next_dim
        mlp_layers.append(Linear(prev_dim, 1))
        
        self.mlp = Sequential(*mlp_layers)
        
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, pe, edge_index, edge_attr, batch) -> torch.Tensor:
        """Forward pass through the GPS model.
        
        Args:
            x: Node features
            pe: Positional encodings
            edge_index: Graph connectivity
            edge_attr: Edge attributes
            batch: Batch assignment vector
            
        Returns:
            Graph-level predictions
        """
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)


class GraphTransformerTrainer:
    """Trainer class for Graph Transformer models."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the trainer with arguments.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set parameters from args
        self.dataset_path = args.dataset_path
        self.walk_length = args.walk_length
        self.channels = args.channels
        self.pe_dim = args.pe_dim
        self.num_layers = args.num_layers
        self.attn_heads = args.heads
        self.model_dropout = args.model_dropout
        self.attn_dropout = args.attn_dropout
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.batch_size_train = args.batch_size_train
        self.batch_size_eval = args.batch_size_eval
        self.patience = args.patience
        self.min_lr = args.min_lr
        self.num_epochs = args.num_epochs
        
        # Setup datasets, model, optimizer and scheduler
        self.setup_datasets()
        self.setup_model()
        self.setup_optimizer()
        
    def setup_datasets(self) -> None:
        """Setup datasets and dataloaders."""
        self.train_dataset = MISDataset(data_dir=f"{self.dataset_path}/train",
                                        data_label_dir=f"{self.dataset_path}/train_labels")
        self.val_dataset = MISDataset(data_dir=f"{self.dataset_path}/test",
                                        data_label_dir=f"{self.dataset_path}/test_labels")
        self.test_dataset = MISDataset(data_dir=f"{self.dataset_path}/test",
                                        data_label_dir=f"{self.dataset_path}/test_labels")

        print(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size_eval)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_eval)
    
    def setup_model(self) -> None:
        """Setup the model architecture."""
        attn_kwargs = {'dropout': self.attn_dropout}
        self.model = GPS(
            channels=self.channels, 
            pe_dim=self.pe_dim, 
            num_layers=self.num_layers, 
            attn_type=self.args.attn_type,
            attn_kwargs=attn_kwargs,
            heads=self.attn_heads,
            dropout=self.model_dropout,
            walk_length=self.walk_length
        ).to(self.device)
        
        # Print model architecture summary
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
    
    def setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=self.patience,
            min_lr=self.min_lr
        )
        print(f"Optimizer: Adam with LR={self.lr}, weight_decay={self.weight_decay}")
        print(f"Scheduler: ReduceLROnPlateau with patience={self.patience}, min_lr={self.min_lr}")
    
    def train_epoch(self) -> float:
        """Run a single training epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()

        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            self.model.redraw_projection.redraw_projections()
            out = self.model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
            loss = (out.squeeze() - data.y).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            self.optimizer.step()
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate the model on a given dataloader.
        
        Args:
            loader: DataLoader to evaluate on
            
        Returns:
            Mean absolute error
        """
        self.model.eval()

        total_error = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
            total_error += (out.squeeze() - data.y).abs().sum().item()
        return total_error / len(loader.dataset)
    
    def train(self) -> None:
        """Train the model for the specified number of epochs."""
        best_val_mae = float('inf')
        best_test_mae = float('inf')
        
        print(f"Starting training for {self.num_epochs} epochs...")
        for epoch in range(1, self.num_epochs + 1):
            loss = self.train_epoch()
            val_mae = self.evaluate(self.val_loader)
            test_mae = self.evaluate(self.test_loader)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_test_mae = test_mae
                
            self.scheduler.step(val_mae)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'Epoch: {epoch:02d}/{self.num_epochs}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
                  f'Test: {test_mae:.4f}, LR: {current_lr:.6f}')
                  
        print(f"Training completed. Best validation MAE: {best_val_mae:.4f}, corresponding test MAE: {best_test_mae:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Graph Transformer for ZINC')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, default='./data/mis/er_700_800',
                        help='Path to dataset')
    parser.add_argument('--walk_length', type=int, default=20,
                        help='Length of random walks for positional encoding')
    # Model parameters
    parser.add_argument('--attn_type', type=str, default='transformer', 
                        choices=['transformer', 'performer'],
                        help='Type of attention to use')
    parser.add_argument('--channels', type=int, default=64,
                        help='Number of channels in the model')
    parser.add_argument('--pe_dim', type=int, default=8,
                        help='Dimension of positional encoding')
    parser.add_argument('--num_layers', type=int, default=10,
                        help='Number of layers in the model')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--model_dropout', type=float, default=0.0,
                        help='Dropout rate in the model')
    parser.add_argument('--attn_dropout', type=float, default=0.5,
                        help='Dropout rate in attention')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size_train', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--batch_size_eval', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--min_lr', type=float, default=0.00001,
                        help='Minimum learning rate')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the training process."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and train the model
    trainer = GraphTransformerTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()