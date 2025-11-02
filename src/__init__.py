"""
Inventory Prediction Utilities Package
"""

from .data_processing import (
    load_data,
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_price_features,
    encode_categorical_features,
    split_train_test,
    scale_features,
    handle_missing_values,
    generate_sample_data
)

from .model_utils import (
    calculate_metrics,
    evaluate_model,
    save_model,
    load_model,
    predict_inventory,
    get_feature_importance,
    calculate_prediction_intervals,
    compare_models,
    detect_anomalies,
    create_ensemble_predictions,
    optimize_inventory,
    calculate_inventory_costs,
    print_model_summary
)

from .visualization import (
    plot_time_series,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_feature_importance,
    plot_model_comparison,
    plot_correlation_matrix
)

__version__ = '1.0.0'
__all__ = [
    # Data processing
    'load_data',
    'create_time_features',
    'create_lag_features',
    'create_rolling_features',
    'create_price_features',
    'encode_categorical_features',
    'split_train_test',
    'scale_features',
    'handle_missing_values',
    'generate_sample_data',
    
    # Model utilities
    'calculate_metrics',
    'evaluate_model',
    'save_model',
    'load_model',
    'predict_inventory',
    'get_feature_importance',
    'calculate_prediction_intervals',
    'compare_models',
    'detect_anomalies',
    'create_ensemble_predictions',
    'optimize_inventory',
    'calculate_inventory_costs',
    'print_model_summary',
    
    # Visualization
    'plot_time_series',
    'plot_predictions_vs_actual',
    'plot_residuals',
    'plot_feature_importance',
    'plot_model_comparison',
    'plot_correlation_matrix',
]
