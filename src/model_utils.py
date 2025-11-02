"""
Model utilities for inventory prediction project.

This module contains functions for:
- Model training and evaluation
- Model persistence
- Prediction generation
- Performance metrics calculation
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            r2_score, mean_absolute_percentage_error)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'Max_Error': np.max(np.abs(y_true - y_pred)),
        'Mean_Error': np.mean(y_true - y_pred),
        'Median_Error': np.median(np.abs(y_true - y_pred))
    }


def evaluate_model(model: Any,
                  X_train: np.ndarray,
                  X_test: np.ndarray,
                  y_train: np.ndarray,
                  y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a model on training and test sets.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        Dictionary with train and test metrics
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    return {
        'train': train_metrics,
        'test': test_metrics
    }


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def predict_inventory(model: Any,
                     features: pd.DataFrame,
                     scaler: Any = None) -> np.ndarray:
    """
    Generate inventory predictions.
    
    Args:
        model: Trained model
        features: Input features
        scaler: Feature scaler (optional)
        
    Returns:
        Array of predictions
    """
    if scaler is not None:
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
    else:
        predictions = model.predict(features)
    
    return predictions


def get_feature_importance(model: Any,
                          feature_names: List[str],
                          top_n: int = 20) -> pd.DataFrame:
    """
    Get feature importance from tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def calculate_prediction_intervals(predictions: np.ndarray,
                                  historical_errors: np.ndarray,
                                  confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals based on historical errors.
    
    Args:
        predictions: Model predictions
        historical_errors: Historical prediction errors
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Calculate error quantiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    error_lower = np.percentile(historical_errors, lower_percentile)
    error_upper = np.percentile(historical_errors, upper_percentile)
    
    lower_bound = predictions + error_lower
    upper_bound = predictions + error_upper
    
    return lower_bound, upper_bound


def compare_models(models: Dict[str, Any],
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  X_test_scaled: np.ndarray = None) -> pd.DataFrame:
    """
    Compare multiple models on test data.
    
    Args:
        models: Dictionary of model name to model object
        X_test: Test features
        y_test: Test target
        X_test_scaled: Scaled test features (for models requiring scaling)
        
    Returns:
        DataFrame with model comparison results
    """
    results = []
    
    for name, model in models.items():
        # Determine if model needs scaled features
        if name in ['Linear Regression', 'Ridge', 'Lasso', 'Neural Network'] and X_test_scaled is not None:
            predictions = model.predict(X_test_scaled)
        else:
            predictions = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, predictions)
        
        results.append({
            'Model': name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2'],
            'MAPE': metrics['MAPE'],
            'Max_Error': metrics['Max_Error']
        })
    
    results_df = pd.DataFrame(results).sort_values('RMSE')
    return results_df


def detect_anomalies(predictions: np.ndarray,
                    actuals: np.ndarray,
                    threshold_std: float = 3.0) -> np.ndarray:
    """
    Detect anomalies in predictions based on error threshold.
    
    Args:
        predictions: Model predictions
        actuals: Actual values
        threshold_std: Number of standard deviations for threshold
        
    Returns:
        Boolean array indicating anomalies
    """
    errors = actuals - predictions
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    threshold = threshold_std * std_error
    anomalies = np.abs(errors - mean_error) > threshold
    
    return anomalies


def create_ensemble_predictions(predictions_dict: Dict[str, np.ndarray],
                               weights: Dict[str, float] = None) -> np.ndarray:
    """
    Create ensemble predictions from multiple models.
    
    Args:
        predictions_dict: Dictionary of model name to predictions
        weights: Dictionary of model name to weight (optional)
        
    Returns:
        Ensemble predictions
    """
    if weights is None:
        # Equal weights
        weights = {name: 1.0/len(predictions_dict) for name in predictions_dict.keys()}
    
    ensemble = np.zeros_like(list(predictions_dict.values())[0])
    
    for name, predictions in predictions_dict.items():
        ensemble += predictions * weights[name]
    
    return ensemble


def optimize_inventory(predictions: np.ndarray,
                      safety_stock_factor: float = 1.5,
                      min_stock: int = 10) -> np.ndarray:
    """
    Calculate optimal inventory levels based on predictions.
    
    Args:
        predictions: Predicted sales
        safety_stock_factor: Multiplier for safety stock
        min_stock: Minimum stock level
        
    Returns:
        Optimal inventory levels
    """
    # Calculate safety stock
    safety_stock = predictions * (safety_stock_factor - 1.0)
    
    # Optimal inventory = predicted demand + safety stock
    optimal_inventory = predictions + safety_stock
    
    # Apply minimum stock constraint
    optimal_inventory = np.maximum(optimal_inventory, min_stock)
    
    return optimal_inventory.astype(int)


def calculate_inventory_costs(actual_demand: np.ndarray,
                             inventory_levels: np.ndarray,
                             holding_cost_per_unit: float = 1.0,
                             stockout_cost_per_unit: float = 5.0) -> float:
    """
    Calculate total inventory costs.
    
    Args:
        actual_demand: Actual demand/sales
        inventory_levels: Inventory levels maintained
        holding_cost_per_unit: Cost to hold one unit
        stockout_cost_per_unit: Cost of one unit stockout
        
    Returns:
        Total inventory cost
    """
    # Calculate excess inventory (holding cost)
    excess = np.maximum(inventory_levels - actual_demand, 0)
    holding_cost = np.sum(excess * holding_cost_per_unit)
    
    # Calculate stockouts (shortage cost)
    stockouts = np.maximum(actual_demand - inventory_levels, 0)
    stockout_cost = np.sum(stockouts * stockout_cost_per_unit)
    
    total_cost = holding_cost + stockout_cost
    
    return total_cost


def print_model_summary(model: Any,
                       metrics: Dict[str, float],
                       model_name: str = "Model") -> None:
    """
    Print a formatted summary of model performance.
    
    Args:
        model: Trained model
        metrics: Dictionary of performance metrics
        model_name: Name of the model
    """
    print("\n" + "=" * 70)
    print(f"{model_name} PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"RMSE:        {metrics['RMSE']:.2f}")
    print(f"MAE:         {metrics['MAE']:.2f}")
    print(f"R² Score:    {metrics['R2']:.4f}")
    print(f"MAPE:        {metrics['MAPE']:.2f}%")
    print(f"Max Error:   {metrics['Max_Error']:.2f}")
    print(f"Mean Error:  {metrics['Mean_Error']:.2f}")
    print("=" * 70)
