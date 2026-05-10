"""
Visualization and demo utilities for neuro-symbolic AI experiments.

This module provides plotting functions, interactive demos, and
visualization tools for neuro-symbolic AI research and education.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class NeuroSymbolicVisualizer:
    """
    Visualization class for neuro-symbolic AI experiments.
    
    This class provides various visualization methods for understanding
    neuro-symbolic model behavior and results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot training history including loss and accuracy curves.
        
        Args:
            history: Dictionary containing training history
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot losses
        ax1.plot(history['train_losses'], label='Train Loss', color=self.colors[0])
        ax1.plot(history['val_losses'], label='Validation Loss', color=self.colors[1])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax2.plot(history['train_accuracies'], label='Train Accuracy', color=self.colors[0])
        ax2.plot(history['val_accuracies'], label='Validation Accuracy', color=self.colors[1])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
        
        plt.show()
    
    def plot_decision_boundary(self, model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
                             resolution: int = 100, save_path: Optional[str] = None) -> None:
        """
        Plot decision boundary for 2D data.
        
        Args:
            model: Trained model
            X: Input features (2D)
            y: Target labels
            resolution: Resolution of the decision boundary grid
            save_path: Optional path to save the plot
        """
        if X.shape[1] != 2:
            logger.warning("Decision boundary plot only works for 2D data")
            return
        
        model.eval()
        device = next(model.parameters()).device
        
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_tensor = torch.tensor(mesh_points, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if hasattr(model, 'forward') and len(model.forward(mesh_tensor)) == 2:
                _, predictions = model(mesh_tensor)
                predictions = torch.sigmoid(predictions).squeeze()
            else:
                predictions = torch.sigmoid(model(mesh_tensor)).squeeze()
        
        Z = predictions.cpu().numpy().reshape(xx.shape)
        
        # Create the plot
        plt.figure(figsize=self.figsize)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        plt.colorbar(label='Prediction Probability')
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                            edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='True Label')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary Visualization')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved decision boundary plot to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot comparison of different models.
        
        Args:
            results: Dictionary containing model evaluation results
            save_path: Optional path to save the plot
        """
        if not results:
            logger.warning("No results to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            if metric in df.columns:
                bars = ax.bar(df.index, df[metric], color=self.colors[:len(df)])
                ax.set_title(title)
                ax.set_ylabel(metric.capitalize())
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {save_path}")
        
        plt.show()
    
    def plot_symbolic_explanations(self, model: torch.nn.Module, X: np.ndarray, 
                                 y: np.ndarray, n_samples: int = 10,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot symbolic explanations for model predictions.
        
        Args:
            model: Trained model
            X: Input features
            y: Target labels
            n_samples: Number of samples to show explanations for
            save_path: Optional path to save the plot
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Select random samples
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            sample = X[idx]
            target = y[idx]
            
            # Get prediction
            with torch.no_grad():
                sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
                if hasattr(model, 'forward') and len(model.forward(sample_tensor)) == 2:
                    neural_out, symbolic_out = model(sample_tensor)
                    prediction = torch.sigmoid(symbolic_out).item()
                else:
                    prediction = torch.sigmoid(model(sample_tensor)).item()
            
            # Create explanation visualization
            ax = axes[i]
            
            # Plot feature values
            features = sample[:2] if len(sample) >= 2 else sample
            bars = ax.bar(range(len(features)), features, 
                         color=['red' if f > 0.5 else 'blue' for f in features])
            
            ax.set_title(f'Sample {idx}\nTrue: {target}, Pred: {prediction:.3f}')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Value')
            ax.set_ylim(0, 1)
            
            # Add threshold line
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Symbolic Explanations for Sample Predictions', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved symbolic explanations plot to {save_path}")
        
        plt.show()
    
    def create_interactive_demo(self, model: torch.nn.Module, 
                              data_info: Dict[str, Any]) -> None:
        """
        Create an interactive Streamlit demo.
        
        Args:
            model: Trained model
            data_info: Information about the dataset
        """
        st.set_page_config(page_title="Neuro-Symbolic AI Demo", layout="wide")
        
        st.title("🧠 Neuro-Symbolic AI Interactive Demo")
        st.markdown("**Author:** [kryptologyst](https://github.com/kryptologyst)")
        
        # Sidebar
        st.sidebar.header("Model Information")
        st.sidebar.write(f"**Model Type:** {data_info.get('model_type', 'Unknown')}")
        st.sidebar.write(f"**Data Type:** {data_info.get('data_type', 'Unknown')}")
        st.sidebar.write(f"**Rule Type:** {data_info.get('rule_type', 'Unknown')}")
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["🔍 Interactive Prediction", "📊 Model Analysis", "📈 Training History"])
        
        with tab1:
            st.header("Interactive Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input Features")
                if data_info.get('n_features', 2) >= 2:
                    feature1 = st.slider("Feature 1", 0.0, 1.0, 0.5, 0.01)
                    feature2 = st.slider("Feature 2", 0.0, 1.0, 0.5, 0.01)
                    features = [feature1, feature2]
                else:
                    feature1 = st.slider("Feature 1", 0.0, 1.0, 0.5, 0.01)
                    features = [feature1]
                
                # Add more features if needed
                for i in range(2, data_info.get('n_features', 2)):
                    features.append(st.slider(f"Feature {i+1}", 0.0, 1.0, 0.5, 0.01))
            
            with col2:
                st.subheader("Prediction Results")
                
                # Make prediction
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    if hasattr(model, 'forward') and len(model.forward(input_tensor)) == 2:
                        neural_out, symbolic_out = model(input_tensor)
                        neural_prob = torch.sigmoid(neural_out).item()
                        symbolic_prob = torch.sigmoid(symbolic_out).item()
                    else:
                        output = model(input_tensor)
                        neural_prob = torch.sigmoid(output).item()
                        symbolic_prob = neural_prob
                
                st.metric("Neural Prediction", f"{neural_prob:.3f}")
                st.metric("Symbolic Prediction", f"{symbolic_prob:.3f}")
                
                # Show symbolic explanation
                st.subheader("Symbolic Explanation")
                if len(features) >= 2:
                    sum_features = sum(features[:2])
                    threshold = data_info.get('threshold', 1.0)
                    if sum_features > threshold:
                        st.success(f"Sum of features ({sum_features:.2f}) > threshold ({threshold}) → Class 1")
                    else:
                        st.info(f"Sum of features ({sum_features:.2f}) ≤ threshold ({threshold}) → Class 0")
        
        with tab2:
            st.header("Model Analysis")
            
            # Model comparison
            if 'results' in data_info:
                st.subheader("Model Performance Comparison")
                results_df = pd.DataFrame(data_info['results']).T
                st.dataframe(results_df, use_container_width=True)
                
                # Performance chart
                fig = px.bar(results_df.reset_index(), 
                           x='index', y='accuracy',
                           title="Model Accuracy Comparison")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Training History")
            
            if 'history' in data_info:
                history = data_info['history']
                
                # Loss plot
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=history['train_losses'], 
                                            name='Train Loss', line=dict(color='blue')))
                fig_loss.add_trace(go.Scatter(y=history['val_losses'], 
                                            name='Validation Loss', line=dict(color='red')))
                fig_loss.update_layout(title="Training and Validation Loss",
                                     xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Accuracy plot
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(y=history['train_accuracies'], 
                                           name='Train Accuracy', line=dict(color='blue')))
                fig_acc.add_trace(go.Scatter(y=history['val_accuracies'], 
                                           name='Validation Accuracy', line=dict(color='red')))
                fig_acc.update_layout(title="Training and Validation Accuracy",
                                    xaxis_title="Epoch", yaxis_title="Accuracy")
                st.plotly_chart(fig_acc, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown("**⚠️ Disclaimer:** This is a research/education demo. Not for production decisions.")
        st.markdown("**🔬 Research Focus:** Neuro-symbolic AI combines neural learning with symbolic reasoning.")


def create_plotly_visualizations(results: Dict[str, Dict[str, float]]) -> List[go.Figure]:
    """
    Create interactive Plotly visualizations.
    
    Args:
        results: Model evaluation results
        
    Returns:
        List of Plotly figures
    """
    figures = []
    
    if not results:
        return figures
    
    df = pd.DataFrame(results).T
    
    # Model comparison bar chart
    fig_bar = px.bar(df.reset_index(), x='index', y='accuracy',
                     title="Model Accuracy Comparison",
                     labels={'index': 'Model', 'accuracy': 'Accuracy'})
    figures.append(fig_bar)
    
    # Radar chart for multiple metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) > 2:
        fig_radar = go.Figure()
        
        for model_name in df.index:
            values = [df.loc[model_name, metric] for metric in available_metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics,
                fill='toself',
                name=model_name
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        figures.append(fig_radar)
    
    return figures
