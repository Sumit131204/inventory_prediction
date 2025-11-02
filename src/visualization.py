"""
Visualization utilities for inventory prediction project.

This module contains functions for creating visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple


def plot_time_series(df: pd.DataFrame,
                    date_col: str,
                    value_col: str,
                    title: str = "Time Series Plot",
                    figsize: Tuple[int, int] = (15, 6)) -> None:
    """
    Plot time series data.
    
    Args:
        df: DataFrame containing the data
        date_col: Name of date column
        value_col: Name of value column
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(df[date_col], df[value_col], linewidth=1)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(value_col, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               title: str = "Predictions vs Actual",
                               figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot predictions against actual values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot residual analysis.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        figsize: Figure size
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 15,
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        figsize: Figure size
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame,
                         metric: str = 'RMSE',
                         figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot model comparison.
    
    Args:
        results_df: DataFrame with model comparison results
        metric: Metric to compare
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.barh(results_df['Model'], results_df[metric], color='steelblue')
    plt.xlabel(metric, fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame,
                           columns: List[str] = None,
                           figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: Input DataFrame
        columns: List of columns to include (None for all numeric columns)
        figsize: Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    correlation_matrix = df[columns].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
