"""
Training and evaluation utilities for neuro-symbolic AI models.

This module provides training loops, evaluation metrics, and model
comparison utilities for neuro-symbolic AI experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
import json
import os

logger = logging.getLogger(__name__)


class NeuroSymbolicTrainer:
    """
    Trainer class for neuro-symbolic AI models.
    
    This trainer handles the training loop, validation, and model checkpointing
    for neuro-symbolic AI experiments.
    """
    
    def __init__(self, model: nn.Module, device: str = "auto", 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model to train
            device: Device to use for training ("auto", "cuda", "mps", "cpu")
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and len(self.model.forward(data)) == 2:
                # Handle models that return both neural and symbolic outputs
                neural_out, symbolic_out = self.model(data)
                output = symbolic_out.squeeze()
            else:
                output = self.model(data).squeeze()
            
            # Calculate loss
            loss = self.criterion(output, target.squeeze())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == target.squeeze()).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and len(self.model.forward(data)) == 2:
                    neural_out, symbolic_out = self.model(data)
                    output = symbolic_out.squeeze()
                else:
                    output = self.model(data).squeeze()
                
                loss = self.criterion(output, target.squeeze())
                total_loss += loss.item()
                
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == target.squeeze()).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, patience: int = 10, 
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Number of epochs to wait before early stopping
            save_path: Path to save the best model
            
        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }, save_path)
                    logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1
            
            # Log progress
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                       f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }


class ModelEvaluator:
    """
    Evaluator class for comparing different neuro-symbolic models.
    
    This evaluator provides comprehensive evaluation metrics and
    comparison utilities for neuro-symbolic AI experiments.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the evaluator.
        
        Args:
            device: Device to use for evaluation
        """
        self.device = self._get_device(device)
        self.results = {}
        
        logger.info(f"Initialized evaluator on device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                      model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate a single model on test data.
        
        Args:
            model: The model to evaluate
            test_loader: Test data loader
            model_name: Name of the model for storing results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        model.to(self.device)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if hasattr(model, 'forward') and len(model.forward(data)) == 2:
                    neural_out, symbolic_out = model(data)
                    output = symbolic_out.squeeze()
                else:
                    output = model(data).squeeze()
                
                probabilities = torch.sigmoid(output).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        # Store results
        self.results[model_name] = metrics
        
        logger.info(f"Evaluated {model_name}: Accuracy={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, targets: List[float], predictions: List[int], 
                          probabilities: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        targets = np.array(targets)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'f1': f1_score(targets, predictions, zero_division=0),
            'auc': roc_auc_score(targets, probabilities) if len(np.unique(targets)) > 1 else 0.0
        }
        
        return metrics
    
    def evaluate_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate baseline models for comparison.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing baseline model results
        """
        baselines = {}
        
        # Logistic Regression
        try:
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train, y_train)
            lr_pred = lr_model.predict(X_test)
            lr_prob = lr_model.predict_proba(X_test)[:, 1]
            
            baselines['Logistic Regression'] = self._calculate_metrics(y_test, lr_pred, lr_prob)
            logger.info("Evaluated Logistic Regression baseline")
        except Exception as e:
            logger.warning(f"Failed to evaluate Logistic Regression: {e}")
        
        # Random Forest
        try:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_prob = rf_model.predict_proba(X_test)[:, 1]
            
            baselines['Random Forest'] = self._calculate_metrics(y_test, rf_pred, rf_prob)
            logger.info("Evaluated Random Forest baseline")
        except Exception as e:
            logger.warning(f"Failed to evaluate Random Forest: {e}")
        
        # Store baseline results
        for name, metrics in baselines.items():
            self.results[name] = metrics
        
        return baselines
    
    def create_leaderboard(self) -> pd.DataFrame:
        """
        Create a leaderboard comparing all evaluated models.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.results:
            logger.warning("No results available for leaderboard")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results).T
        df = df.sort_values('accuracy', ascending=False)
        
        logger.info("Created model leaderboard")
        return df
    
    def save_results(self, filepath: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            filepath: Path to save the results
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    logger.info(f"Set random seed to {seed}")


def create_experiment_config(model_type: str, data_type: str, 
                           epochs: int = 100, batch_size: int = 32,
                           learning_rate: float = 0.001) -> Dict[str, Any]:
    """
    Create a configuration dictionary for experiments.
    
    Args:
        model_type: Type of model to use
        data_type: Type of data to generate
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Configuration dictionary
    """
    config = {
        'model_type': model_type,
        'data_type': data_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': 'auto',
        'random_seed': 42,
        'patience': 10,
        'weight_decay': 1e-4
    }
    
    return config
