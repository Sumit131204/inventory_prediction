"""
Example script for making inventory predictions using trained models.

This script demonstrates how to:
1. Load a trained model
2. Prepare input features
3. Generate predictions
4. Calculate optimal inventory levels
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_utils import predict_inventory, optimize_inventory
from src.data_processing import create_time_features


def load_trained_model(model_name='xgboost_tuned_model'):
    """Load a trained model and scaler."""
    try:
        model = joblib.load(f'models/{model_name}.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        print(f"✓ Loaded model: {model_name}")
        return model, scaler, feature_names
    except FileNotFoundError:
        print(f"✗ Model not found. Please run the training notebooks first.")
        return None, None, None


def prepare_prediction_features(product, store, date, price, promotion, 
                                competitor_price, recent_sales):
    """
    Prepare features for prediction.
    
    Args:
        product: Product name/code
        store: Store name/code
        date: Date for prediction
        price: Product price
        promotion: Promotion flag (0 or 1)
        competitor_price: Competitor's price
        recent_sales: Dictionary with recent sales data
                     e.g., {'lag_1': 45, 'lag_7': 48, 'lag_30': 50}
    """
    # Create base dataframe
    df = pd.DataFrame({
        'date': [pd.to_datetime(date)],
        'product': [product],
        'store': [store],
        'price': [price],
        'promotion': [promotion],
        'competitor_price': [competitor_price]
    })
    
    # Add time features
    df = create_time_features(df)
    
    # Add price features
    df['price_ratio'] = df['price'] / df['competitor_price']
    df['price_difference'] = df['price'] - df['competitor_price']
    df['discounted_price'] = df['price'] * (1 - 0.1 * df['promotion'])
    
    # Add lag features (from recent sales)
    df['sales_lag_1'] = recent_sales.get('lag_1', 50)
    df['sales_lag_3'] = recent_sales.get('lag_3', 50)
    df['sales_lag_7'] = recent_sales.get('lag_7', 50)
    df['sales_lag_14'] = recent_sales.get('lag_14', 50)
    df['sales_lag_30'] = recent_sales.get('lag_30', 50)
    
    # Add rolling features (from recent sales)
    df['sales_rolling_mean_7'] = recent_sales.get('rolling_mean_7', 50)
    df['sales_rolling_std_7'] = recent_sales.get('rolling_std_7', 5)
    df['sales_rolling_mean_14'] = recent_sales.get('rolling_mean_14', 50)
    df['sales_rolling_std_14'] = recent_sales.get('rolling_std_14', 5)
    df['sales_rolling_mean_30'] = recent_sales.get('rolling_mean_30', 50)
    df['sales_rolling_std_30'] = recent_sales.get('rolling_std_30', 5)
    
    # Encode categorical variables (simplified)
    product_map = {'Product_A': 0, 'Product_B': 1, 'Product_C': 2, 
                   'Product_D': 3, 'Product_E': 4}
    store_map = {'Store_1': 0, 'Store_2': 1, 'Store_3': 2, 'Store_4': 3}
    
    df['product_encoded'] = df['product'].map(product_map).fillna(0)
    df['store_encoded'] = df['store'].map(store_map).fillna(0)
    
    return df


def predict_single_inventory(model, scaler, feature_names, 
                            product, store, date, price, promotion,
                            competitor_price, recent_sales):
    """Make a single inventory prediction."""
    
    # Prepare features
    features_df = prepare_prediction_features(
        product, store, date, price, promotion, 
        competitor_price, recent_sales
    )
    
    # Select only the features used in training
    # Note: In production, ensure feature alignment
    available_features = [f for f in feature_names if f in features_df.columns]
    X = features_df[available_features]
    
    # Make prediction
    predicted_sales = predict_inventory(model, X, scaler=None)
    
    # Calculate optimal inventory with safety stock
    optimal_inventory = optimize_inventory(
        predicted_sales, 
        safety_stock_factor=1.5,  # 50% safety stock
        min_stock=10
    )
    
    return {
        'predicted_sales': predicted_sales[0],
        'recommended_inventory': optimal_inventory[0],
        'safety_stock': optimal_inventory[0] - predicted_sales[0]
    }


def main():
    """Main execution function with example usage."""
    
    print("=" * 70)
    print("INVENTORY PREDICTION EXAMPLE")
    print("=" * 70)
    
    # Load model
    model, scaler, feature_names = load_trained_model('xgboost')
    
    if model is None:
        return
    
    # Example prediction inputs
    example_inputs = {
        'product': 'Product_A',
        'store': 'Store_1',
        'date': '2025-01-15',
        'price': 29.99,
        'promotion': 1,  # Has promotion
        'competitor_price': 27.50,
        'recent_sales': {
            'lag_1': 52,
            'lag_3': 48,
            'lag_7': 50,
            'lag_14': 47,
            'lag_30': 45,
            'rolling_mean_7': 49,
            'rolling_std_7': 4.2,
            'rolling_mean_14': 48,
            'rolling_std_14': 4.8,
            'rolling_mean_30': 47,
            'rolling_std_30': 5.1
        }
    }
    
    print("\nInput Parameters:")
    print(f"  Product: {example_inputs['product']}")
    print(f"  Store: {example_inputs['store']}")
    print(f"  Date: {example_inputs['date']}")
    print(f"  Price: ${example_inputs['price']}")
    print(f"  Promotion: {'Yes' if example_inputs['promotion'] else 'No'}")
    print(f"  Competitor Price: ${example_inputs['competitor_price']}")
    print(f"  Recent Sales (lag_7): {example_inputs['recent_sales']['lag_1']}")
    
    # Make prediction
    result = predict_single_inventory(
        model, scaler, feature_names,
        **example_inputs
    )
    
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"  Predicted Sales: {result['predicted_sales']:.0f} units")
    print(f"  Recommended Inventory: {result['recommended_inventory']:.0f} units")
    print(f"  Safety Stock: {result['safety_stock']:.0f} units")
    print("=" * 70)
    
    # Calculate inventory cost comparison
    holding_cost = 1.0  # $ per unit
    stockout_cost = 5.0  # $ per unit
    
    print("\nCost Analysis:")
    print(f"  Holding Cost (per unit): ${holding_cost}")
    print(f"  Stockout Cost (per unit): ${stockout_cost}")
    print(f"  Safety Stock Cost: ${result['safety_stock'] * holding_cost:.2f}")
    
    print("\n💡 Recommendation:")
    if example_inputs['promotion']:
        print("  ⚠️  Promotion active - consider higher safety stock")
    if result['predicted_sales'] > example_inputs['recent_sales']['rolling_mean_7']:
        print("  📈 Sales trending up - monitor closely")
    else:
        print("  📊 Sales stable - standard inventory levels")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
