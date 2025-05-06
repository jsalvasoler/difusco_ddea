from __future__ import annotations

"""
Graph Transformer model for node classification on the MIS dataset.

This module implements a Graph Transformer model using the Graph Positional Embedding with 
self-attention (GPS) architecture for solving the Maximal Independent Set (MIS) node
classification problem. Each node is classified as either part of the MIS (1) or not (0)
using a binary cross-entropy loss.

Key components:
- GPS: Graph Positional Embedding model with self-attention for node classification
- NodeFeatureTransform: Custom transform to prepare node features and labels
- GraphTransformerTrainer: Trainer class for the GPS model

The model uses a combination of graph attention and message passing to learn node representations,
followed by an MLP classifier to predict node-level scores.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import os
from torch.nn import Linear, Embedding, BatchNorm1d, Sequential, ReLU, ModuleList
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
from torch_geometric.nn.conv import GPSConv
from problems.mis.mis_dataset import MISDataset
from problems.mis.mis_instance import MISInstance
import torch_geometric.transforms as T
from typing import Any, Optional
from tqdm import tqdm


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
    """Graph Positional Embedding with self-attention and message passing layers for node classification."""
    
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                attn_type: str, attn_kwargs: dict[str, Any], 
                heads: int = 4, num_classes: int = 2, walk_length: int = 20,
                dropout: float = 0.0):
        super().__init__()

        # Initial node feature embedding
        self.node_emb = Linear(num_classes, channels - pe_dim)
        self.pe_lin = Linear(walk_length, pe_dim)
        self.pe_norm = BatchNorm1d(walk_length)
        # We still need edge embeddings for GINE
        self.edge_emb = Linear(1, channels)  # Simple linear layer for dummy edge features
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

        # MLP for node classification
        mlp_layers = []
        prev_dim = channels
        for i, dim_factor in enumerate([2, 2]):
            next_dim = channels // dim_factor
            mlp_layers.append(Linear(prev_dim, next_dim))
            mlp_layers.append(ReLU())
            if self.dropout > 0 and i == 0:  # Only add dropout after first layer
                mlp_layers.append(torch.nn.Dropout(self.dropout))
            prev_dim = next_dim
        mlp_layers.append(Linear(prev_dim, num_classes))
        
        self.mlp = Sequential(*mlp_layers)
        
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, pe, edge_index, batch, edge_attr=None) -> torch.Tensor:
        """Forward pass through the GPS model.
        
        Args:
            x: Node features
            pe: Positional encodings
            edge_index: Graph connectivity
            batch: Batch assignment vector
            edge_attr: Edge attributes (or None)
            
        Returns:
            Node-level class logits
        """
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        
        # Create dummy edge attributes if none provided
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), 1), device=edge_index.device)
        
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr, batch=batch)
        
        # Node-level classification (no pooling)
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
        self.num_classes = args.num_classes
        
        # Setup checkpoint directory
        self.checkpoint_dir = args.checkpoint_dir
        if self.checkpoint_dir is not None:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print(f"Created checkpoint directory: {self.checkpoint_dir}")
        else:
            print("No checkpoint directory specified. Models will not be saved.")
        
        # Initialize wandb
        self.setup_wandb(args)
        
        # Setup datasets, model, optimizer and scheduler
        self.setup_datasets()
        self.setup_model()
        self.setup_optimizer()
        
    def setup_wandb(self, args: argparse.Namespace) -> None:
        """Initialize wandb for logging.
        
        Args:
            args: Command line arguments
        """
        if not args.use_wandb:
            self.use_wandb = False
            return
            
        self.use_wandb = True
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "architecture": "GPS",
                "dataset": self.dataset_path,
                "num_layers": self.num_layers,
                "channels": self.channels,
                "pe_dim": self.pe_dim,
                "heads": self.attn_heads,
                "attn_type": args.attn_type,
                "model_dropout": self.model_dropout,
                "attn_dropout": self.attn_dropout,
                "learning_rate": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size_train": self.batch_size_train,
                "batch_size_eval": self.batch_size_eval,
                "num_epochs": self.num_epochs,
                "walk_length": self.walk_length,
                "num_classes": self.num_classes,
                "seed": args.seed,
            }
        )
        
        print(f"Wandb initialized. Project: {args.wandb_project}, Run: {args.wandb_name}")
    
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

        self.test_loader_batch_one = DataLoader(self.test_dataset, batch_size=1)
    
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
            num_classes=self.num_classes,
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
    
    def preprocess_batch(self, batch_data) -> tuple:
        """Preprocess the batch data from MISDataset.
        
        Args:
            batch_data: Batch data from MISDataset
            
        Returns:
            Processed input features, positional encodings, edge indices, batch indices, edge attributes, and target labels
        """
        # Unpack batch data
        indices, graph_data, point_indicator = batch_data
        
        # Current MISDataset format puts node labels in x
        node_labels = graph_data.x
        
        # Create dummy node features (one-hot encoding of node indices)
        batch_size = len(indices)
        num_nodes_total = node_labels.size(0)
        
        # Create dummy features as one-hot vectors
        node_features = torch.zeros((num_nodes_total, self.num_classes), device=self.device)
        node_features[:, 0] = 1.0  # All ones in the first feature
        
        # Create batch assignment vector
        batch_idx = torch.zeros(num_nodes_total, dtype=torch.long, device=self.device)
        start_idx = 0
        for i, num_nodes in enumerate(point_indicator):
            batch_idx[start_idx:start_idx + num_nodes] = i
            start_idx += num_nodes
                
        # Create random walk positional encodings
        # In practice, we would use a proper transform
        # This is a placeholder that creates random values
        pe = torch.randn((num_nodes_total, self.walk_length), device=self.device)
        
        # Create dummy edge features (just ones)
        edge_attr = torch.ones((graph_data.edge_index.size(1), 1), device=self.device)
        
        return node_features, pe, graph_data.edge_index, batch_idx, edge_attr, node_labels
        
    def train_epoch(self) -> float:
        """Run a single training epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()

        total_loss = 0
        num_nodes = 0
        
        # Create a tqdm progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f"Training", 
            leave=True,
            ncols=100,
            unit="batch",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            indices, graph_data, point_indicator = batch_data
            graph_data = graph_data.to(self.device)
            point_indicator = point_indicator.to(self.device)
            
            # Preprocess batch
            node_features, pe, edge_index, batch_idx, edge_attr, node_labels = self.preprocess_batch((indices, graph_data, point_indicator))
            
            self.optimizer.zero_grad()
            self.model.redraw_projection.redraw_projections()
            
            # Forward pass
            logits = self.model(node_features, pe, edge_index, batch_idx, edge_attr)
            
            # CrossEntropyLoss for node classification
            loss = F.cross_entropy(logits, node_labels)
            
            loss.backward()
            batch_loss = loss.item()
            total_loss += batch_loss * node_labels.size(0)
            num_nodes += node_labels.size(0)
            self.optimizer.step()
            
            # Update progress bar with current loss and average loss
            if num_nodes > 0:
                avg_loss = total_loss / num_nodes
                pbar.set_postfix({
                    'batch_loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'nodes': num_nodes
                })
            
        return total_loss / num_nodes if num_nodes > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluate the model on a given dataloader.
        
        Args:
            loader: DataLoader to evaluate on
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()

        total_loss = 0
        correct = 0
        total_nodes = 0
        
        # Create a progress bar for evaluation
        desc = "Validating" if loader == self.val_loader else "Testing"
        pbar = tqdm(
            loader, 
            desc=desc,
            leave=False,
            ncols=100,
            unit="batch",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for batch_data in pbar:
            # Move data to device
            indices, graph_data, point_indicator = batch_data
            graph_data = graph_data.to(self.device)
            point_indicator = point_indicator.to(self.device)
            
            # Preprocess batch
            node_features, pe, edge_index, batch_idx, edge_attr, node_labels = self.preprocess_batch((indices, graph_data, point_indicator))
            
            # Forward pass
            logits = self.model(node_features, pe, edge_index, batch_idx, edge_attr)
            
            # Compute loss
            loss = F.cross_entropy(logits, node_labels)
            batch_loss = loss.item()
            total_loss += batch_loss * node_labels.size(0)
            
            # Compute accuracy
            pred = logits.argmax(dim=1)
            batch_correct = (pred == node_labels).sum().item()
            correct += batch_correct
            total_nodes += node_labels.size(0)
            
            # Update progress bar
            if total_nodes > 0:
                avg_loss = total_loss / total_nodes
                accuracy = correct / total_nodes
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'acc': f'{accuracy:.4f}'
                })
            
        return total_loss / total_nodes, correct / total_nodes
    
    @torch.no_grad()
    def evaluate_mis_size(self, loader: DataLoader) -> float:
        """Evaluate the average MIS size on a given dataloader using actual MIS construction.
        
        Args:
            loader: DataLoader to evaluate on
            
        Returns:
            Average MIS size across all graphs
        """
        self.model.eval()
        
        total_mis_size = 0
        num_graphs = 0
        
        # Create a progress bar for MIS evaluation
        pbar = tqdm(
            loader, 
            desc="Evaluating MIS",
            leave=False,
            ncols=100,
            unit="graph",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for batch_data in pbar:
            # Move data to device
            indices, graph_data, point_indicator = batch_data
            graph_data = graph_data.to(self.device)
            point_indicator = point_indicator.to(self.device)
            
            # Preprocess batch
            node_features, pe, edge_index, batch_idx, edge_attr, node_labels = self.preprocess_batch((indices, graph_data, point_indicator))
            
            # Forward pass to get node scores
            logits = self.model(node_features, pe, edge_index, batch_idx, edge_attr)
            
            # Get node scores (probability of being in MIS)
            node_scores = F.softmax(logits, dim=1)[:, 1].cpu()  # Probability of class 1 (in MIS)
            
            # Create MIS instance
            mis_instance = MISInstance.create_from_batch_sample(batch_data, "cpu")
            
            # Get feasible MIS solution
            mis_solution = mis_instance.get_feasible_from_individual(node_scores)
            mis_size = mis_solution.sum().item()
            
            total_mis_size += mis_size
            num_graphs += 1
            
            # Update progress bar
            if num_graphs > 0:
                avg_mis_size = total_mis_size / num_graphs
                pbar.set_postfix({
                    'current_mis': f'{mis_size:.2f}',
                    'avg_mis': f'{avg_mis_size:.2f}'
                })
        
        return total_mis_size / num_graphs if num_graphs > 0 else 0.0
    
    def save_checkpoint(self, epoch: int, val_acc: float, test_acc: float, 
                    mis_size: Optional[float] = None, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_acc: Validation accuracy
            test_acc: Test accuracy
            mis_size: Average MIS size (optional)
            is_best: Whether this is the best model so far
        """
        # Skip saving if checkpoint_dir is None
        if self.checkpoint_dir is None:
            if is_best:
                print(f"New best model at epoch {epoch} with MIS size: {mis_size:.4f}, but not saving (checkpoint_dir is None)")
            return
            
        # Create a checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'mis_size': mis_size,
            # Save configuration for model reconstruction
            'config': {
                'channels': self.channels,
                'pe_dim': self.pe_dim,
                'num_layers': self.num_layers,
                'attn_type': self.args.attn_type,
                'attn_dropout': self.attn_dropout,
                'heads': self.attn_heads,
                'num_classes': self.num_classes,
                'model_dropout': self.model_dropout,
                'walk_length': self.walk_length
            }
        }
        
        # Save regular checkpoint
        run_name = self.args.wandb_name if self.args.wandb_name else "model"
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{run_name}_checkpoint_epoch{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model separately
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, f"{run_name}_best_model.pt")
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model to {best_model_path}")
            
    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Loaded checkpoint dictionary
        """
        # Check if file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
        
        # Recreate model with the same configuration
        config = checkpoint['config']
        attn_kwargs = {'dropout': config['attn_dropout']}
        self.model = GPS(
            channels=config['channels'],
            pe_dim=config['pe_dim'],
            num_layers=config['num_layers'],
            attn_type=config['attn_type'],
            attn_kwargs=attn_kwargs,
            heads=config['heads'],
            num_classes=config['num_classes'],
            dropout=config['model_dropout'],
            walk_length=config['walk_length']
        ).to(self.device)

        self.model = torch.compile(self.model)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler if they exist
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint
        
    def train(self) -> None:
        """Train the model for the specified number of epochs."""
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_epoch = 0
        best_mis_size = 0.0
        
        print(f"\n{'='*80}\nStarting training for {self.num_epochs} epochs...\n{'='*80}")
        
        # Evaluate MIS size at epoch 0
        avg_mis_size = self.evaluate_mis_size(self.test_loader_batch_one)
        best_mis_size = avg_mis_size  # Set initial MIS size as best
        print(f'Epoch 0: Average MIS size on test set: {avg_mis_size:.4f}')
        
        # Log initial metrics
        if self.use_wandb:
            wandb.log({
                "epoch": 0,
                "avg_mis_size": avg_mis_size,
            })
        
        total_training_time = 0.0
        
        # Create epoch progress bar
        epoch_pbar = tqdm(
            range(1, self.num_epochs + 1),
            desc="Epochs",
            position=0,
            leave=True,
            ncols=100,
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for epoch in epoch_pbar:
            # Start timing the epoch
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader)
            test_loss, test_acc = self.evaluate(self.test_loader)
            
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time
            total_training_time += epoch_duration
            
            # Check if this is the best model in terms of validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            
            # Evaluate MIS size every epoch (since this is our primary metric)
            avg_mis_size = self.evaluate_mis_size(self.test_loader_batch_one)
            
            # Check if this is the best model in terms of MIS size
            is_best_mis = avg_mis_size > best_mis_size
            if is_best_mis:
                best_mis_size = avg_mis_size
                best_epoch = epoch
                # Print notification of best model (save_checkpoint will handle whether to actually save)
                if self.checkpoint_dir is not None:
                    print(f'✅ New best model at epoch {epoch} with MIS size: {avg_mis_size:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}')
                else:
                    print(f'✅ New best model at epoch {epoch} with MIS size: {avg_mis_size:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f} (not saving)')
                
                # Try to save the best model based on MIS size (will be skipped if checkpoint_dir is None)
                self.save_checkpoint(epoch, val_acc, test_acc, avg_mis_size, is_best=True)
            
            # Save regular checkpoint every 10 epochs (will be skipped if checkpoint_dir is None)
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_acc, test_acc, avg_mis_size)
                
            self.scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_acc': f'{val_acc:.4f}',
                'MIS': f'{avg_mis_size:.2f}',
                'LR': f'{current_lr:.6f}',
                'time': f'{epoch_duration:.2f}s'
            })
            
            # Log more detailed metrics to console every 10 epochs
            if epoch % 10 == 0 or epoch == 1 or is_best_mis:
                print(f"\nEpoch: {epoch:02d}/{self.num_epochs}, Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                      f"MIS size: {avg_mis_size:.4f}, LR: {current_lr:.6f}, "
                      f"Time: {epoch_duration:.2f}s\n")
            
            # Log metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "avg_mis_size": avg_mis_size,
                    "learning_rate": current_lr,
                    "epoch_duration": epoch_duration,
                    "total_training_time": total_training_time,
                })
                  
        # Print final statistics
        avg_epoch_time = total_training_time / self.num_epochs
        print(f"\n{'='*80}")
        print(f"Training completed. Best MIS size: {best_mis_size:.4f} at epoch {best_epoch}, "
              f"best validation accuracy: {best_val_acc:.4f}, "
              f"corresponding test accuracy: {best_test_acc:.4f}")
        print(f"Total training time: {total_training_time:.2f}s, Average epoch time: {avg_epoch_time:.2f}s")
        print(f"{'='*80}\n")
        
        # Final MIS size evaluation (using current model, not best model)
        avg_mis_size = self.evaluate_mis_size(self.test_loader_batch_one)
        print(f'Final average MIS size on test set: {avg_mis_size:.4f}')
        
        # Log final metrics
        if self.use_wandb:
            wandb.log({
                "epoch": self.num_epochs,
                "final_avg_mis_size": avg_mis_size,
                "best_val_accuracy": best_val_acc,
                "best_test_accuracy": best_test_acc,
                "best_epoch": best_epoch,
                "best_mis_size": best_mis_size,
                "total_training_time": total_training_time,
                "avg_epoch_time": avg_epoch_time,
            })
            wandb.finish()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Graph Transformer for Node Classification on MIS Dataset')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, default='./data/mis/er_700_800',
                        help='Path to dataset')
    parser.add_argument('--walk_length', type=int, default=20,
                        help='Length of random walks for positional encoding')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of node classes')
    
    # Model parameters
    parser.add_argument('--attn_type', type=str, default='performer', 
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
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save model checkpoints. If None, no checkpoints are saved.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='graph-transformer-mis',
                        help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Wandb run name')
    
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
    
    # Create trainer
    trainer = GraphTransformerTrainer(args)
    
    # Resume from checkpoint if specified
    if args.resume_checkpoint:
        try:
            trainer.load_checkpoint(args.resume_checkpoint)
            print(f"Resumed training from checkpoint: {args.resume_checkpoint}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch instead.")
    
    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()