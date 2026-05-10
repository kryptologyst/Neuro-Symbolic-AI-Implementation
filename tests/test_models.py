"""
Test suite for neuro-symbolic AI implementation.

This module contains unit tests for all components of the
neuro-symbolic AI system.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import SimpleNN, SymbolicNeuralModule, GraphNeuralSymbolic, create_model, symbolic_reasoning
from src.data import generate_synthetic_data, create_data_loaders, NeuroSymbolicDataset, preprocess_data
from src.train import NeuroSymbolicTrainer, ModelEvaluator, set_seed


class TestModels:
    """Test cases for neural network models."""
    
    def test_simple_nn_creation(self):
        """Test SimpleNN model creation."""
        model = SimpleNN(input_dim=2, hidden_dim=64, output_dim=1)
        assert model.input_dim == 2
        assert model.hidden_dim == 64
        assert model.output_dim == 1
    
    def test_simple_nn_forward(self):
        """Test SimpleNN forward pass."""
        model = SimpleNN(input_dim=2, hidden_dim=64, output_dim=1)
        x = torch.randn(10, 2)
        output = model(x)
        assert output.shape == (10, 1)
    
    def test_symbolic_neural_module_creation(self):
        """Test SymbolicNeuralModule creation."""
        model = SymbolicNeuralModule(input_dim=2, hidden_dim=64)
        assert model.input_dim == 2
        assert model.hidden_dim == 64
    
    def test_symbolic_neural_module_forward(self):
        """Test SymbolicNeuralModule forward pass."""
        model = SymbolicNeuralModule(input_dim=2, hidden_dim=64)
        x = torch.randn(10, 2)
        neural_out, symbolic_out = model(x)
        assert neural_out.shape == (10, 1)
        assert symbolic_out.shape == (10, 1)
    
    def test_graph_neural_symbolic_creation(self):
        """Test GraphNeuralSymbolic creation."""
        model = GraphNeuralSymbolic(node_dim=10, hidden_dim=64, num_layers=2)
        assert model.node_dim == 10
        assert model.hidden_dim == 64
        assert model.num_layers == 2
    
    def test_graph_neural_symbolic_forward(self):
        """Test GraphNeuralSymbolic forward pass."""
        model = GraphNeuralSymbolic(node_dim=10, hidden_dim=64, num_layers=2)
        node_features = torch.randn(20, 10)
        edge_index = torch.randint(0, 20, (2, 30))
        output = model(node_features, edge_index)
        assert output.shape == (20, 1)
    
    def test_create_model_factory(self):
        """Test model factory function."""
        simple_model = create_model("simple", input_dim=2)
        assert isinstance(simple_model, SimpleNN)
        
        symbolic_model = create_model("symbolic", input_dim=2)
        assert isinstance(symbolic_model, SymbolicNeuralModule)
        
        graph_model = create_model("graph", node_dim=10)
        assert isinstance(graph_model, GraphNeuralSymbolic)
    
    def test_symbolic_reasoning_function(self):
        """Test symbolic reasoning function."""
        predictions = torch.tensor([0.3, 0.7, 0.5, 0.9])
        results = symbolic_reasoning(predictions, threshold=0.5)
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
        assert torch.equal(results, expected)


class TestData:
    """Test cases for data handling."""
    
    def test_generate_synthetic_data_sum_threshold(self):
        """Test synthetic data generation with sum threshold rule."""
        X, y, rules = generate_synthetic_data(
            n_samples=100, 
            rule_type="sum_threshold", 
            random_state=42
        )
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert rules["rule_type"] == "sum_threshold"
        assert "threshold" in rules
    
    def test_generate_synthetic_data_xor(self):
        """Test synthetic data generation with XOR rule."""
        X, y, rules = generate_synthetic_data(
            n_samples=100, 
            rule_type="xor", 
            random_state=42
        )
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert rules["rule_type"] == "xor"
    
    def test_generate_synthetic_data_circle(self):
        """Test synthetic data generation with circle rule."""
        X, y, rules = generate_synthetic_data(
            n_samples=100, 
            rule_type="circle", 
            random_state=42
        )
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert rules["rule_type"] == "circle"
        assert "center" in rules
        assert "radius" in rules
    
    def test_neuro_symbolic_dataset(self):
        """Test NeuroSymbolicDataset class."""
        X = np.random.rand(50, 2)
        y = np.random.randint(0, 2, 50)
        dataset = NeuroSymbolicDataset(X, y)
        
        assert len(dataset) == 50
        sample_x, sample_y = dataset[0]
        assert sample_x.shape == (2,)
        assert sample_y.shape == ()
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        X = np.random.rand(100, 2)
        y = np.random.randint(0, 2, 100)
        
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y, batch_size=16, random_state=42
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # Test batch
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] <= 16
        assert batch_y.shape[0] <= 16
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        X = np.random.rand(100, 2) * 10  # Scale up
        y = np.random.randint(0, 2, 100)
        
        X_processed, y_processed = preprocess_data(X, y, normalize=True)
        
        assert X_processed.shape == X.shape
        assert y_processed.shape == y.shape
        # Check if normalization was applied (mean should be close to 0)
        assert np.abs(np.mean(X_processed)) < 0.1


class TestTraining:
    """Test cases for training and evaluation."""
    
    def test_neuro_symbolic_trainer_creation(self):
        """Test NeuroSymbolicTrainer creation."""
        model = SimpleNN(input_dim=2, hidden_dim=64)
        trainer = NeuroSymbolicTrainer(model, device="cpu")
        assert trainer.model == model
        assert trainer.device == "cpu"
    
    def test_model_evaluator_creation(self):
        """Test ModelEvaluator creation."""
        evaluator = ModelEvaluator(device="cpu")
        assert evaluator.device == "cpu"
        assert evaluator.results == {}
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
    
    def test_trainer_train_epoch(self):
        """Test trainer train epoch."""
        model = SimpleNN(input_dim=2, hidden_dim=64)
        trainer = NeuroSymbolicTrainer(model, device="cpu")
        
        # Create dummy data
        X = torch.randn(32, 2)
        y = torch.randint(0, 2, (32,)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        loss, acc = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
    
    def test_trainer_validate(self):
        """Test trainer validation."""
        model = SimpleNN(input_dim=2, hidden_dim=64)
        trainer = NeuroSymbolicTrainer(model, device="cpu")
        
        # Create dummy data
        X = torch.randn(32, 2)
        y = torch.randint(0, 2, (32,)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        loss, acc = trainer.validate(loader)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete neuro-symbolic AI pipeline."""
        # Set seed for reproducibility
        set_seed(42)
        
        # Generate data
        X, y, rules = generate_synthetic_data(
            n_samples=100, 
            rule_type="sum_threshold", 
            random_state=42
        )
        
        # Preprocess data
        X, y = preprocess_data(X, y, normalize=True)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y, batch_size=16, random_state=42
        )
        
        # Create model
        model = create_model("symbolic", input_dim=X.shape[1])
        
        # Create trainer
        trainer = NeuroSymbolicTrainer(model, device="cpu", learning_rate=0.01)
        
        # Train for a few epochs
        history = trainer.train(train_loader, val_loader, epochs=5, patience=3)
        
        # Check history
        assert "train_losses" in history
        assert "val_losses" in history
        assert "train_accuracies" in history
        assert "val_accuracies" in history
        
        # Create evaluator
        evaluator = ModelEvaluator(device="cpu")
        
        # Evaluate model
        results = evaluator.evaluate_model(model, test_loader, "test_model")
        
        # Check results
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "auc" in results
        
        # All metrics should be between 0 and 1
        for metric, value in results.items():
            assert 0 <= value <= 1


if __name__ == "__main__":
    pytest.main([__file__])
