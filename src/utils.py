"""
Utility functions for neuro-symbolic AI experiments.

This module provides common utility functions used across
the neuro-symbolic AI implementation.
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_device(device: str = "auto") -> str:
    """
    Determine the best available device.
    
    Args:
        device: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        Available device string
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def set_deterministic(seed: int = 42) -> None:
    """
    Set deterministic behavior for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Set Python random seed
    import random
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set MPS random seed
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def create_directories(base_dir: Union[str, Path], subdirs: list) -> None:
    """
    Create directory structure.
    
    Args:
        base_dir: Base directory path
        subdirs: List of subdirectories to create
    """
    base_dir = Path(base_dir)
    
    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)


def save_results(results: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary
        filepath: Path to save results
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def format_time(seconds: float) -> str:
    """
    Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> str:
    """
    Get the size of a model in a human-readable format.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size string
    """
    param_count = count_parameters(model)
    
    if param_count < 1000:
        return f"{param_count} parameters"
    elif param_count < 1000000:
        return f"{param_count/1000:.1f}K parameters"
    else:
        return f"{param_count/1000000:.1f}M parameters"


def print_model_summary(model: torch.nn.Module, input_shape: tuple) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (excluding batch dimension)
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape}")
    print(f"Parameters: {get_model_size(model)}")
    print(f"Device: {next(model.parameters()).device}")
    print("-" * 50)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['model', 'data', 'training']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required configuration key: {key}")
            return False
    
    # Validate model configuration
    model_config = config['model']
    if 'type' not in model_config:
        print("Missing model type in configuration")
        return False
    
    # Validate data configuration
    data_config = config['data']
    if 'type' not in data_config:
        print("Missing data type in configuration")
        return False
    
    # Validate training configuration
    training_config = config['training']
    if 'epochs' not in training_config:
        print("Missing epochs in training configuration")
        return False
    
    return True


def create_experiment_name(config: Dict[str, Any]) -> str:
    """
    Create a descriptive name for the experiment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment name string
    """
    model_type = config.get('model', {}).get('type', 'unknown')
    data_type = config.get('data', {}).get('type', 'unknown')
    epochs = config.get('training', {}).get('epochs', 0)
    
    return f"{model_type}_{data_type}_{epochs}epochs"


def print_experiment_info(config: Dict[str, Any]) -> None:
    """
    Print experiment information.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 60)
    print("NEURO-SYMBOLIC AI EXPERIMENT")
    print("=" * 60)
    print(f"Model Type: {config.get('model', {}).get('type', 'N/A')}")
    print(f"Data Type: {config.get('data', {}).get('type', 'N/A')}")
    print(f"Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
    print(f"Batch Size: {config.get('training', {}).get('batch_size', 'N/A')}")
    print(f"Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
    print(f"Device: {config.get('training', {}).get('device', 'N/A')}")
    print(f"Random Seed: {config.get('reproducibility', {}).get('random_seed', 'N/A')}")
    print("=" * 60)


def check_gpu_memory() -> Dict[str, Any]:
    """
    Check GPU memory usage.
    
    Returns:
        Dictionary with GPU memory information
    """
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA not available"}
    
    memory_info = {
        "available": True,
        "total_memory": torch.cuda.get_device_properties(0).total_memory,
        "allocated_memory": torch.cuda.memory_allocated(0),
        "cached_memory": torch.cuda.memory_reserved(0),
    }
    
    memory_info["free_memory"] = (
        memory_info["total_memory"] - memory_info["allocated_memory"]
    )
    
    # Convert to GB
    for key in ["total_memory", "allocated_memory", "cached_memory", "free_memory"]:
        memory_info[f"{key}_gb"] = memory_info[key] / (1024**3)
    
    return memory_info


def print_gpu_info() -> None:
    """Print GPU information."""
    memory_info = check_gpu_memory()
    
    if not memory_info["available"]:
        print("GPU: Not available")
        return
    
    print("GPU Information:")
    print(f"  Total Memory: {memory_info['total_memory_gb']:.2f} GB")
    print(f"  Allocated Memory: {memory_info['allocated_memory_gb']:.2f} GB")
    print(f"  Cached Memory: {memory_info['cached_memory_gb']:.2f} GB")
    print(f"  Free Memory: {memory_info['free_memory_gb']:.2f} GB")
