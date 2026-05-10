#!/usr/bin/env python3
"""
Simple demo script for neuro-symbolic AI implementation.

This script demonstrates the core functionality without requiring Streamlit.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import create_model, symbolic_reasoning
from src.data import generate_synthetic_data, create_data_loaders, preprocess_data
from src.train import NeuroSymbolicTrainer, ModelEvaluator, set_seed
from src.viz import NeuroSymbolicVisualizer


def main():
    """Run a simple neuro-symbolic AI demo."""
    print("Neuro-Symbolic AI Demo")
    print("=" * 50)
    print("Author: kryptologyst")
    print("GitHub: https://github.com/kryptologyst")
    print("Research/Education Demo Only - Not for production decisions")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    print("Set random seed for reproducibility")
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X, y, symbolic_rules = generate_synthetic_data(
        n_samples=500,
        n_features=2,
        rule_type="sum_threshold",
        noise_level=0.1,
        random_state=42
    )
    
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Symbolic rule: {symbolic_rules['description']}")
    
    # Preprocess data
    X, y = preprocess_data(X, y, normalize=True)
    print("Data preprocessing completed")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y, batch_size=32, random_state=42
    )
    print(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    # Create models
    print("\nCreating models...")
    models = {
        "Simple Neural": create_model("simple", input_dim=X.shape[1]),
        "Symbolic Neural": create_model("symbolic", input_dim=X.shape[1])
    }
    
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name}: {param_count:,} parameters")
    
    # Train models
    print("\nTraining models...")
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create trainer
        trainer = NeuroSymbolicTrainer(
            model=model,
            device="auto",
            learning_rate=0.01
        )
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,
            patience=5
        )
        
        # Evaluate model
        evaluator = ModelEvaluator()
        model_results = evaluator.evaluate_model(
            model=model,
            test_loader=test_loader,
            model_name=name
        )
        
        results[name] = {
            "model": model,
            "history": history,
            "results": model_results
        }
        
        print(f"{name} - Accuracy: {model_results['accuracy']:.3f}, F1: {model_results['f1']:.3f}")
    
    # Model comparison
    print("\nModel Performance Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        metrics = result["results"]
        print(f"{name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['auc']:<10.3f}")
    
    # Symbolic reasoning demonstration
    print("\nSymbolic Reasoning Demonstration:")
    print("-" * 50)
    
    test_samples = np.array([
        [0.7, 0.3],  # Sum = 1.0, should be class 1
        [0.2, 0.9],  # Sum = 1.1, should be class 1
        [0.4, 0.5],  # Sum = 0.9, should be class 0
        [0.1, 0.2]   # Sum = 0.3, should be class 0
    ])
    
    symbolic_model = results["Symbolic Neural"]["model"]
    symbolic_model.eval()
    device = next(symbolic_model.parameters()).device
    
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            # Get neural prediction
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
            if hasattr(symbolic_model, 'forward') and len(symbolic_model.forward(sample_tensor)) == 2:
                neural_out, symbolic_out = symbolic_model(sample_tensor)
                neural_prob = torch.sigmoid(neural_out).item()
                symbolic_prob = torch.sigmoid(symbolic_out).item()
            else:
                output = symbolic_model(sample_tensor)
                neural_prob = torch.sigmoid(output).item()
                symbolic_prob = neural_prob
            
            # Apply symbolic rule
            sum_features = np.sum(sample)
            threshold = symbolic_rules['threshold']
            symbolic_rule_result = 1 if sum_features > threshold else 0
            
            print(f"\nSample {i+1}: {sample}")
            print(f"  Sum of features: {sum_features:.2f}")
            print(f"  Threshold: {threshold}")
            print(f"  Symbolic rule result: {symbolic_rule_result}")
            print(f"  Neural prediction: {neural_prob:.3f}")
            print(f"  Symbolic prediction: {symbolic_prob:.3f}")
            print(f"  Final prediction: {'Class 1' if symbolic_prob > 0.5 else 'Class 0'}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = NeuroSymbolicVisualizer()
    
    # Plot training history
    symbolic_history = results["Symbolic Neural"]["history"]
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(symbolic_history['train_losses'], label='Train Loss', color='blue')
    plt.plot(symbolic_history['val_losses'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(symbolic_history['train_accuracies'], label='Train Accuracy', color='blue')
    plt.plot(symbolic_history['val_accuracies'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_training_history.png', dpi=300, bbox_inches='tight')
    print("Saved training history plot: demo_training_history.png")
    
    # Plot decision boundary
    visualizer.plot_decision_boundary(symbolic_model, X, y)
    plt.savefig('demo_decision_boundary.png', dpi=300, bbox_inches='tight')
    print("Saved decision boundary plot: demo_decision_boundary.png")
    
    # Summary
    print("\nSummary:")
    print("=" * 50)
    print("Successfully demonstrated neuro-symbolic AI concepts")
    print("Combined neural learning with symbolic reasoning")
    print("Generated interpretable explanations for predictions")
    print("Created visualizations of model behavior")
    print("\nKey Insights:")
    print("• Neural networks learn complex patterns from data")
    print("• Symbolic rules provide interpretable reasoning")
    print("• Hybrid approach balances learning and interpretability")
    print("• Clear explanations enhance trust and understanding")
    
    print("\nResearch Focus:")
    print("This implementation demonstrates how neuro-symbolic AI can:")
    print("• Combine data-driven learning with rule-based reasoning")
    print("• Provide transparent explanations for AI decisions")
    print("• Balance performance with interpretability")
    print("• Enable trustworthy AI systems")
    
    print("\nNext Steps:")
    print("• Run interactive demo: streamlit run demo.py")
    print("• Experiment with different symbolic rules")
    print("• Try different model architectures")
    print("• Explore real-world applications")
    
    print("\nDisclaimer:")
    print("This is a research and educational demonstration.")
    print("Not for production decisions or real-world applications.")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
