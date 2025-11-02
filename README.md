# Inventory Prediction System

Machine Learning-based inventory prediction system for retail demand forecasting with full-stack web application.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-blue)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Project Overview

This project implements multiple machine learning models to predict inventory needs, helping businesses optimize stock levels, reduce costs, and prevent stockouts. The system analyzes historical sales patterns, seasonality, pricing, promotions, and other factors to generate accurate inventory forecasts.

### Key Features
✅ **6 ML Models** - Compare Linear Regression, Ridge, Random Forest, XGBoost, LightGBM, Neural Network  
✅ **60+ Engineered Features** - Temporal, lag, rolling window, and statistical features  
✅ **FastAPI Backend** - Production-ready REST API with async support  
✅ **React Frontend** - Modern UI with Material-UI and interactive charts  
✅ **Easy Deployment** - Deploy backend on Render, frontend on Vercel (free tier available)  
✅ **Research Paper** - Complete academic documentation

## 📁 Project Structure

```
inventory_prediction/
│
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data ready for modeling
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and preprocessing
│   ├── 02_model_training.ipynb        # Model training and comparison
│   └── 03_model_evaluation.ipynb     # Evaluation and tuning
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data processing utilities
│   ├── model_utils.py          # Model training and evaluation utilities
│   └── visualization.py        # Visualization utilities
│
├── models/                     # Saved trained models
│
├── reports/                    # Analysis reports and results
│
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository or navigate to the project directory:
```bash
cd inventory_prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Data Exploration and Preprocessing**:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```
   - Run all cells to generate sample data and perform EDA
   - Creates processed dataset with engineered features

2. **Model Training**:
   ```bash
   jupyter notebook notebooks/02_model_training.ipynb
   ```
   - Trains 6 different ML models
   - Compares performance metrics
   - Saves trained models

3. **Model Evaluation**:
   ```bash
   jupyter notebook notebooks/03_model_evaluation.ipynb
   ```
   - Performs cross-validation
   - Tunes hyperparameters
   - Generates detailed evaluation reports

## 📊 Models Implemented

1. **Linear Regression**: Baseline model
2. **Ridge Regression**: L2 regularization
3. **Random Forest**: Ensemble of decision trees
4. **XGBoost**: Gradient boosting (typically best performer)
5. **LightGBM**: Fast gradient boosting
6. **Neural Network**: Multi-layer perceptron

## 🔧 Key Features Generated

### Time-Based Features
- Year, Month, Day, Day of Week, Week of Year, Quarter
- Weekend indicator
- Month start/end indicators
- Cyclical encodings (sin/cos transformations)

### Lag Features
- Previous 1, 3, 7, 14, 30 days sales

### Rolling Window Features
- 7, 14, 30-day rolling mean and standard deviation

### Price Features
- Price ratio (vs competitors)
- Price difference
- Discounted price
- Inventory-to-sales ratio

## 📈 Performance Metrics

Models are evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score**
- **MAPE** (Mean Absolute Percentage Error)
- **Max Error**
- **Cross-validation scores**

## 💾 Using the Utility Functions

```python
from src.data_processing import load_data, create_time_features
from src.model_utils import evaluate_model, predict_inventory
from src.visualization import plot_predictions_vs_actual

# Load and process data
df = load_data('data/raw/sales_data.csv')
df = create_time_features(df)

# Make predictions
predictions = predict_inventory(model, features, scaler)

# Visualize results
plot_predictions_vs_actual(y_test, predictions)
```

## 📝 Model Usage Example

```python
import joblib
import pandas as pd
from src.model_utils import predict_inventory

# Load trained model
model = joblib.load('models/xgboost_tuned_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare features
features = pd.DataFrame({
    'price': [29.99],
    'promotion': [1],
    'sales_lag_7': [45.2],
    # ... other features
})

# Generate prediction
inventory_prediction = predict_inventory(model, features, scaler)
print(f"Recommended inventory: {inventory_prediction[0]:.0f} units")
```

## 🎯 Results Summary

Typical performance metrics (on sample data):
- **Test RMSE**: 8-12 units
- **Test R²**: 0.85-0.92
- **Test MAPE**: 12-18%

Best performing models: XGBoost and LightGBM (after hyperparameter tuning)

## 📦 Dependencies

Main libraries used:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib

See `requirements.txt` for complete list with versions.

## 🔄 Workflow

1. **Data Collection**: Historical sales data with relevant features
2. **Preprocessing**: Feature engineering, handling missing values, encoding
3. **Model Training**: Train multiple models with proper validation
4. **Evaluation**: Compare models, tune hyperparameters
5. **Deployment**: Save best model for production use
6. **Monitoring**: Track performance, retrain periodically

## 📌 Best Practices

- **Time-based splitting**: Use chronological split for train/test
- **Feature scaling**: Normalize features for linear models
- **Cross-validation**: Use TimeSeriesSplit for robust evaluation
- **Regular retraining**: Update models with new data monthly
- **Error monitoring**: Track prediction errors by product/store

## 🚧 Future Enhancements

- [ ] Integration with real-time data sources
- [ ] API endpoint for predictions
- [ ] Automated retraining pipeline
- [ ] Dashboard for monitoring
- [ ] Multi-step ahead forecasting
- [ ] Incorporation of external factors (weather, holidays, events)
- [ ] Deep learning models (LSTM, Transformer)

## 📞 Support

For questions or issues, please check:
1. Notebook documentation and comments
2. Utility function docstrings
3. Error messages and logs

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- Built using industry-standard ML libraries
- Follows best practices for time series forecasting
- Designed for production deployment

---

**Last Updated**: October 2025
**Version**: 1.0.0
# inventory_prediction
# inventory_prediction
