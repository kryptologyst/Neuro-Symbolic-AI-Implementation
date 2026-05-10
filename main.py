"""
Main experiment script for neuro-symbolic AI implementation.

This script demonstrates the complete neuro-symbolic AI pipeline including
data generation, model training, evaluation, and visualization.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
import pandas as pd

from src.models import create_model, symbolic_reasoning
from src.data import generate_synthetic_data, create_data_loaders, preprocess_data
from src.train import NeuroSymbolicTrainer, ModelEvaluator, set_seed, create_experiment_config
from src.viz import NeuroSymbolicVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(config: Dict[str, Any], output_dir: str = "results") -> Dict[str, Any]:
    """
    Run a complete neuro-symbolic AI experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing experiment results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(config['random_seed'])
    
    logger.info(f"Starting experiment with config: {config}")
    
    # Generate synthetic data
    X, y, symbolic_rules = generate_synthetic_data(
        n_samples=1000,
        n_features=2,
        rule_type=config['data_type'],
        random_state=config['random_seed']
    )
    
    # Preprocess data
    X, y = preprocess_data(X, y, normalize=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y, 
        batch_size=config['batch_size'],
        random_state=config['random_seed']
    )
    
    # Create model
    model = create_model(
        model_type=config['model_type'],
        input_dim=X.shape[1],
        hidden_dim=64
    )
    
    # Initialize trainer
    trainer = NeuroSymbolicTrainer(
        model=model,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    model_save_path = os.path.join(output_dir, f"best_model_{config['model_type']}.pth")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        patience=config['patience'],
        save_path=model_save_path
    )
    
    # Evaluate model
    evaluator = ModelEvaluator(device=config['device'])
    
    # Load best model
    device = trainer.device
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate neural model
    neural_results = evaluator.evaluate_model(
        model=model,
        test_loader=test_loader,
        model_name=f"Neural_{config['model_type']}"
    )
    
    # Evaluate baseline models
    baseline_results = evaluator.evaluate_baseline_models(
        X_train=X[train_loader.dataset.indices] if hasattr(train_loader.dataset, 'indices') else X,
        y_train=y[train_loader.dataset.indices] if hasattr(train_loader.dataset, 'indices') else y,
        X_test=X[test_loader.dataset.indices] if hasattr(test_loader.dataset, 'indices') else X,
        y_test=y[test_loader.dataset.indices] if hasattr(test_loader.dataset, 'indices') else y
    )
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard()
    
    # Save results
    results = {
        'config': config,
        'symbolic_rules': symbolic_rules,
        'history': history,
        'results': evaluator.results,
        'leaderboard': leaderboard.to_dict() if not leaderboard.empty else {}
    }
    
    results_path = os.path.join(output_dir, f"results_{config['model_type']}.json")
    evaluator.save_results(results_path)
    
    # Create visualizations
    visualizer = NeuroSymbolicVisualizer()
    
    # Plot training history
    history_plot_path = os.path.join(output_dir, f"training_history_{config['model_type']}.png")
    visualizer.plot_training_history(history, save_path=history_plot_path)
    
    # Plot decision boundary
    decision_boundary_path = os.path.join(output_dir, f"decision_boundary_{config['model_type']}.png")
    visualizer.plot_decision_boundary(model, X, y, save_path=decision_boundary_path)
    
    # Plot model comparison
    comparison_path = os.path.join(output_dir, f"model_comparison_{config['model_type']}.png")
    visualizer.plot_model_comparison(evaluator.results, save_path=comparison_path)
    
    # Plot symbolic explanations
    explanations_path = os.path.join(output_dir, f"symbolic_explanations_{config['model_type']}.png")
    visualizer.plot_symbolic_explanations(model, X, y, save_path=explanations_path)
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")
    
    return results


def main():
    """Main function to run neuro-symbolic AI experiments."""
    parser = argparse.ArgumentParser(description="Neuro-symbolic AI Experiment")
    parser.add_argument("--model_type", type=str, default="simple", 
                       choices=["simple", "symbolic", "graph"],
                       help="Type of model to train")
    parser.add_argument("--data_type", type=str, default="sum_threshold",
                       choices=["sum_threshold", "xor", "circle"],
                       help="Type of symbolic rule for data generation")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for training")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = create_experiment_config(
        model_type=args.model_type,
        data_type=args.data_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    config.update({
        'device': args.device,
        'random_seed': args.random_seed
    })
    
    # Run experiment
    try:
        results = run_experiment(config, args.output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Model Type: {config['model_type']}")
        print(f"Data Type: {config['data_type']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Device: {config['device']}")
        print(f"Random Seed: {config['random_seed']}")
        
        if 'leaderboard' in results and results['leaderboard']:
            print("\nModel Performance:")
            leaderboard_df = pd.DataFrame(results['leaderboard'])
            print(leaderboard_df.to_string(index=True))
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
