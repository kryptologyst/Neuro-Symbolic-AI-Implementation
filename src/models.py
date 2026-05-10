"""
Core neural network models for neuro-symbolic AI.

This module contains various neural network architectures that can be combined
with symbolic reasoning systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class SimpleNN(nn.Module):
    """
    A simple neural network for feature learning in neuro-symbolic systems.
    
    This network learns representations from input data that can then be
    processed by symbolic reasoning modules.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1):
        """
        Initialize the simple neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        super(SimpleNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SymbolicNeuralModule(nn.Module):
    """
    A neural module that incorporates symbolic reasoning capabilities.
    
    This module combines learned representations with explicit symbolic rules
    to make interpretable decisions.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, 
                 symbolic_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize the symbolic neural module.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            symbolic_rules: Dictionary of symbolic rules to apply
        """
        super(SymbolicNeuralModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.symbolic_rules = symbolic_rules or {}
        
        # Neural component for feature learning
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Symbolic reasoning component
        self.symbolic_layer = nn.Linear(1, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining neural and symbolic reasoning.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (neural_output, symbolic_output)
        """
        # Neural feature extraction
        neural_features = self.feature_extractor(x)
        
        # Apply symbolic reasoning
        symbolic_output = self.apply_symbolic_rules(x, neural_features)
        
        return neural_features, symbolic_output
    
    def apply_symbolic_rules(self, x: torch.Tensor, neural_features: torch.Tensor) -> torch.Tensor:
        """
        Apply symbolic rules to the neural features.
        
        Args:
            x: Original input features
            neural_features: Learned neural features
            
        Returns:
            Symbolic reasoning output
        """
        # Simple symbolic rule: if sum of inputs > threshold, then positive
        if len(x.shape) == 2 and x.shape[1] >= 2:
            symbolic_input = torch.sum(x[:, :2], dim=1, keepdim=True)
        else:
            symbolic_input = neural_features
            
        symbolic_output = self.symbolic_layer(symbolic_input)
        return torch.sigmoid(symbolic_output)


class GraphNeuralSymbolic(nn.Module):
    """
    A graph neural network with symbolic reasoning capabilities.
    
    This module processes graph-structured data and applies symbolic rules
    to the learned node/edge representations.
    """
    
    def __init__(self, node_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        """
        Initialize the graph neural symbolic module.
        
        Args:
            node_dim: Dimension of node features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
        """
        super(GraphNeuralSymbolic, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(nn.Linear(node_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Symbolic reasoning head
        self.symbolic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for graph neural symbolic reasoning.
        
        Args:
            node_features: Node feature matrix (num_nodes, node_dim)
            edge_index: Edge connectivity matrix (2, num_edges)
            
        Returns:
            Node-level predictions with symbolic reasoning
        """
        x = node_features
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = F.relu(layer(x))
            
        # Apply symbolic reasoning
        symbolic_output = self.symbolic_head(x)
        
        return torch.sigmoid(symbolic_output)


def create_model(model_type: str = "simple", **kwargs) -> nn.Module:
    """
    Factory function to create different types of neuro-symbolic models.
    
    Args:
        model_type: Type of model to create ("simple", "symbolic", "graph")
        **kwargs: Additional arguments for model creation
        
    Returns:
        Initialized model
    """
    if model_type == "simple":
        return SimpleNN(**kwargs)
    elif model_type == "symbolic":
        return SymbolicNeuralModule(**kwargs)
    elif model_type == "graph":
        return GraphNeuralSymbolic(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def symbolic_reasoning(predictions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Apply symbolic reasoning to neural network predictions.
    
    Args:
        predictions: Raw predictions from neural network
        threshold: Threshold for binary classification
        
    Returns:
        Binary decisions based on symbolic rules
    """
    return (predictions > threshold).float()
