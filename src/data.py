"""
Data handling and generation for neuro-symbolic AI experiments.

This module provides data loaders, synthetic dataset generation, and
preprocessing utilities for neuro-symbolic AI tasks.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class NeuroSymbolicDataset(Dataset):
    """
    A dataset class for neuro-symbolic AI experiments.
    
    This dataset can generate synthetic data with known symbolic rules
    or load real-world data for neuro-symbolic reasoning tasks.
    """
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, 
                 symbolic_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize the neuro-symbolic dataset.
        
        Args:
            data: Input features
            targets: Target labels
            symbolic_rules: Dictionary containing symbolic rules for the data
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.symbolic_rules = symbolic_rules or {}
        
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset."""
        return self.data[idx], self.targets[idx]
    
    def get_symbolic_explanation(self, idx: int) -> str:
        """
        Get symbolic explanation for a given sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            String explanation of the symbolic rule applied
        """
        if not self.symbolic_rules:
            return "No symbolic rules available"
            
        sample = self.data[idx]
        target = self.targets[idx]
        
        # Simple symbolic explanation based on the rule
        if "threshold" in self.symbolic_rules:
            threshold = self.symbolic_rules["threshold"]
            if len(sample) >= 2:
                sum_features = torch.sum(sample[:2]).item()
                if sum_features > threshold:
                    return f"Sum of features ({sum_features:.2f}) > threshold ({threshold}) → Class 1"
                else:
                    return f"Sum of features ({sum_features:.2f}) ≤ threshold ({threshold}) → Class 0"
        
        return f"Sample {idx}: Features {sample.tolist()} → Class {target.item()}"


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 2, 
                          rule_type: str = "sum_threshold", 
                          noise_level: float = 0.1, 
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic data with known symbolic rules for neuro-symbolic experiments.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        rule_type: Type of symbolic rule to apply
        noise_level: Amount of noise to add to the data
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (features, targets, symbolic_rules)
    """
    np.random.seed(random_state)
    
    # Generate random features
    X = np.random.rand(n_samples, n_features)
    
    # Apply symbolic rules to generate targets
    if rule_type == "sum_threshold":
        threshold = 1.0
        y = (np.sum(X[:, :2], axis=1) > threshold).astype(int)
        symbolic_rules = {
            "rule_type": "sum_threshold",
            "threshold": threshold,
            "description": f"Class 1 if sum of first two features > {threshold}"
        }
    elif rule_type == "xor":
        # XOR rule: class 1 if exactly one of first two features > 0.5
        y = ((X[:, 0] > 0.5) != (X[:, 1] > 0.5)).astype(int)
        symbolic_rules = {
            "rule_type": "xor",
            "description": "Class 1 if exactly one of first two features > 0.5"
        }
    elif rule_type == "circle":
        # Circle rule: class 1 if point is inside unit circle
        center = np.array([0.5, 0.5])
        distances = np.sqrt(np.sum((X[:, :2] - center) ** 2, axis=1))
        y = (distances < 0.3).astype(int)
        symbolic_rules = {
            "rule_type": "circle",
            "center": center.tolist(),
            "radius": 0.3,
            "description": "Class 1 if point is inside circle centered at (0.5, 0.5) with radius 0.3"
        }
    else:
        raise ValueError(f"Unknown rule type: {rule_type}")
    
    # Add noise to targets
    if noise_level > 0:
        noise_mask = np.random.rand(n_samples) < noise_level
        y[noise_mask] = 1 - y[noise_mask]  # Flip labels for noisy samples
    
    logger.info(f"Generated {n_samples} samples with {rule_type} rule")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y, symbolic_rules


def generate_graph_data(n_nodes: int = 100, n_features: int = 10, 
                       edge_prob: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic graph data for graph neural symbolic experiments.
    
    Args:
        n_nodes: Number of nodes in the graph
        n_features: Number of features per node
        edge_prob: Probability of edge existence between nodes
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (node_features, edge_index, graph_info)
    """
    np.random.seed(random_state)
    
    # Generate node features
    node_features = np.random.rand(n_nodes, n_features)
    
    # Generate edges
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < edge_prob:
                edges.append([i, j])
                edges.append([j, i])  # Undirected graph
    
    edge_index = np.array(edges).T if edges else np.array([[], []], dtype=int)
    
    # Generate node labels based on simple symbolic rule
    # Class 1 if sum of features > threshold
    threshold = n_features * 0.5
    node_labels = (np.sum(node_features, axis=1) > threshold).astype(int)
    
    graph_info = {
        "n_nodes": n_nodes,
        "n_edges": len(edges) // 2,  # Undirected edges
        "n_features": n_features,
        "edge_prob": edge_prob,
        "threshold": threshold
    }
    
    logger.info(f"Generated graph with {n_nodes} nodes and {len(edges)//2} edges")
    
    return node_features, edge_index, node_labels, graph_info


def create_data_loaders(X: np.ndarray, y: np.ndarray, 
                       batch_size: int = 32, test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        X: Input features
        y: Target labels
        batch_size: Batch size for data loaders
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split train into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
    )
    
    # Create datasets
    train_dataset = NeuroSymbolicDataset(X_train, y_train)
    val_dataset = NeuroSymbolicDataset(X_val, y_val)
    test_dataset = NeuroSymbolicDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def load_real_world_data(data_path: str, target_column: str, 
                        feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real-world data for neuro-symbolic experiments.
    
    Args:
        data_path: Path to the data file (CSV)
        target_column: Name of the target column
        feature_columns: List of feature column names (if None, use all except target)
        
    Returns:
        Tuple of (features, targets)
    """
    try:
        df = pd.read_csv(data_path)
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        logger.info(f"Loaded data from {data_path}: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise


def preprocess_data(X: np.ndarray, y: np.ndarray, 
                   normalize: bool = True, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the data for neuro-symbolic experiments.
    
    Args:
        X: Input features
        y: Target labels
        normalize: Whether to normalize features
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (processed_features, processed_targets)
    """
    X_processed = X.copy()
    y_processed = y.copy()
    
    if normalize:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_processed)
        logger.info("Normalized features using StandardScaler")
    
    return X_processed, y_processed
