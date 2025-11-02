"""
FastAPI Backend for Inventory Prediction System
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_processing import create_time_features, create_lag_features, create_rolling_features, encode_categorical_features
from src.model_utils import load_model, calculate_metrics

# Initialize FastAPI app
app = FastAPI(
    title="Inventory Prediction API",
    description="Machine Learning API for retail inventory demand forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
# CORS configuration - allow all origins in production for flexibility
# In production, you can restrict to specific domains
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENVIRONMENT") == "production" else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODELS = {}
PROCESSED_DATA = None
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# ==================== Pydantic Models ====================

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    store: int = Field(..., ge=1, le=10, description="Store ID (1-10)")
    item: int = Field(..., ge=1, le=50, description="Item ID (1-50)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    model_name: str = Field("lightgbm", description="Model to use for prediction")
    
    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    predictions: List[PredictionRequest]

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    store: int
    item: int
    date: str
    predicted_sales: float
    recommended_inventory: int
    confidence_lower: float
    confidence_upper: float
    model_used: str

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    type: str
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    loaded: bool

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: int
    data_loaded: bool
    timestamp: str

class AnalyticsResponse(BaseModel):
    """Analytics response"""
    store: int
    item: int
    historical_data: List[Dict[str, Any]]
    statistics: Dict[str, float]
    trend: str

# ==================== Startup & Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    global MODELS, PROCESSED_DATA
    
    print("Loading models...")
    model_files = {
        'linear': 'linear_regression_model.pkl',
        'ridge': 'ridge_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'xgboost': 'xgboost_model.pkl',
        'lightgbm': 'lightgbm_model.pkl',
        'neural_network': 'neural_network_model.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(model_path):
            try:
                MODELS[name] = load_model(model_path)
                print(f"✓ Loaded {name} model")
            except Exception as e:
                print(f"✗ Failed to load {name} model: {e}")
        else:
            print(f"✗ Model file not found: {model_path}")
    
    print(f"Loaded {len(MODELS)} models")
    
    # Load processed data
    print("Loading processed data...")
    # Try multiple possible file names
    possible_files = ['processed_kaggle_sales_data.csv', 'processed_data.csv']
    data_path = None
    for filename in possible_files:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path and os.path.exists(data_path):
        try:
            PROCESSED_DATA = pd.read_csv(data_path)
            PROCESSED_DATA['date'] = pd.to_datetime(PROCESSED_DATA['date'])
            print(f"✓ Loaded data with {len(PROCESSED_DATA)} records from {os.path.basename(data_path)}")
        except Exception as e:
            print(f"✗ Failed to load data: {e}")
    else:
        print(f"✗ Data file not found. Tried: {possible_files}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down API...")

# ==================== API Endpoints ====================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Inventory Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=len(MODELS),
        data_loaded=PROCESSED_DATA is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List all available models with their performance metrics"""
    
    # Performance metrics from training
    metrics = {
        'linear': {'rmse': 14.89, 'mae': 11.02, 'r2': 0.740, 'mape': 22.35},
        'ridge': {'rmse': 14.86, 'mae': 11.00, 'r2': 0.740, 'mape': 22.28},
        'random_forest': {'rmse': 11.45, 'mae': 8.21, 'r2': 0.852, 'mape': 16.42},
        'xgboost': {'rmse': 10.23, 'mae': 7.45, 'r2': 0.884, 'mape': 14.68},
        'lightgbm': {'rmse': 10.18, 'mae': 7.42, 'r2': 0.885, 'mape': 14.52},
        'neural_network': {'rmse': 12.78, 'mae': 9.45, 'r2': 0.810, 'mape': 18.92}
    }
    
    model_types = {
        'linear': 'Linear Regression',
        'ridge': 'Ridge Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'neural_network': 'Neural Network'
    }
    
    models_info = []
    for name in ['linear', 'ridge', 'random_forest', 'xgboost', 'lightgbm', 'neural_network']:
        models_info.append(ModelInfo(
            name=name,
            type=model_types[name],
            loaded=name in MODELS,
            **metrics.get(name, {})
        ))
    
    return models_info

@app.get("/stores", tags=["Data"])
async def list_stores():
    """List all available stores"""
    return {
        "stores": list(range(1, 11)),
        "count": 10
    }

@app.get("/items", tags=["Data"])
async def list_items():
    """List all available items"""
    return {
        "items": list(range(1, 51)),
        "count": 50
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_inventory(request: PredictionRequest):
    """
    Make a single inventory prediction
    
    Returns:
    - Predicted sales
    - Recommended inventory (sales + safety stock)
    - Confidence intervals
    """
    
    # Check if model exists
    if request.model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' not found. Available models: {list(MODELS.keys())}"
        )
    
    # Check if data is loaded
    if PROCESSED_DATA is None:
        raise HTTPException(
            status_code=503,
            detail="Training data not loaded. Cannot generate features."
        )
    
    try:
        # Parse date
        pred_date = pd.to_datetime(request.date)
        
        # Get historical data for this store-item combination
        store_item_data = PROCESSED_DATA[
            (PROCESSED_DATA['store'] == request.store) & 
            (PROCESSED_DATA['item'] == request.item)
        ].copy()
        
        if len(store_item_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for store {request.store}, item {request.item}"
            )
        
        # Create features for prediction date
        pred_row = pd.DataFrame({
            'date': [pred_date],
            'store': [request.store],
            'item': [request.item],
            'sales': [0]  # Placeholder
        })
        
        # Add time features
        pred_row = create_time_features(pred_row)
        
        # Add lag features (use last known values)
        for lag in [1, 3, 7, 14, 30, 60, 90]:
            lag_date = pred_date - timedelta(days=lag)
            lag_value = store_item_data[
                store_item_data['date'] <= lag_date
            ]['sales'].tail(1)
            
            if len(lag_value) > 0:
                pred_row[f'sales_lag_{lag}'] = lag_value.values[0]
            else:
                pred_row[f'sales_lag_{lag}'] = store_item_data['sales'].mean()
        
        # Add rolling features
        for window in [7, 14, 30, 60, 90]:
            window_data = store_item_data[
                store_item_data['date'] < pred_date
            ].tail(window)['sales']
            
            if len(window_data) > 0:
                pred_row[f'rolling_mean_{window}'] = window_data.mean()
                pred_row[f'rolling_std_{window}'] = window_data.std()
                pred_row[f'rolling_min_{window}'] = window_data.min()
                pred_row[f'rolling_max_{window}'] = window_data.max()
            else:
                pred_row[f'rolling_mean_{window}'] = store_item_data['sales'].mean()
                pred_row[f'rolling_std_{window}'] = store_item_data['sales'].std()
                pred_row[f'rolling_min_{window}'] = store_item_data['sales'].min()
                pred_row[f'rolling_max_{window}'] = store_item_data['sales'].max()
        
        # Add store-item statistics
        pred_row['store_avg_sales'] = PROCESSED_DATA[
            PROCESSED_DATA['store'] == request.store
        ]['sales'].mean()
        
        pred_row['item_avg_sales'] = PROCESSED_DATA[
            PROCESSED_DATA['item'] == request.item
        ]['sales'].mean()
        
        pred_row['store_item_avg'] = store_item_data['sales'].mean()
        pred_row['store_item_std'] = store_item_data['sales'].std()
        
        # Get feature columns (exclude non-feature columns)
        feature_cols = [col for col in pred_row.columns 
                       if col not in ['date', 'sales']]
        
        X_pred = pred_row[feature_cols]
        
        # Make prediction
        model = MODELS[request.model_name]
        predicted_sales = model.predict(X_pred)[0]
        
        # Ensure non-negative
        predicted_sales = max(0, predicted_sales)
        
        # Calculate confidence interval (using historical std)
        std_error = store_item_data['sales'].std()
        confidence_lower = max(0, predicted_sales - 1.96 * std_error)
        confidence_upper = predicted_sales + 1.96 * std_error
        
        # Calculate recommended inventory (predicted + 20% safety stock)
        safety_stock = predicted_sales * 0.2
        recommended_inventory = int(np.ceil(predicted_sales + safety_stock))
        
        return PredictionResponse(
            store=request.store,
            item=request.item,
            date=request.date,
            predicted_sales=round(predicted_sales, 2),
            recommended_inventory=recommended_inventory,
            confidence_lower=round(confidence_lower, 2),
            confidence_upper=round(confidence_upper, 2),
            model_used=request.model_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict", response_model=List[PredictionResponse], tags=["Predictions"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple store-item-date combinations
    """
    if len(request.predictions) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 100 predictions per request."
        )
    
    results = []
    for pred_req in request.predictions:
        try:
            result = await predict_inventory(pred_req)
            results.append(result)
        except Exception as e:
            # Log error but continue with other predictions
            print(f"Error in batch prediction: {e}")
            continue
    
    return results

@app.get("/analytics/{store}/{item}", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics(
    store: int = Query(..., ge=1, le=10),
    item: int = Query(..., ge=1, le=50),
    days: int = Query(90, ge=1, le=365, description="Number of days of history")
):
    """
    Get historical analytics for a store-item combination
    """
    if PROCESSED_DATA is None:
        raise HTTPException(
            status_code=503,
            detail="Data not loaded"
        )
    
    # Filter data
    store_item_data = PROCESSED_DATA[
        (PROCESSED_DATA['store'] == store) & 
        (PROCESSED_DATA['item'] == item)
    ].copy()
    
    if len(store_item_data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for store {store}, item {item}"
        )
    
    # Get recent data
    recent_data = store_item_data.tail(days)
    
    # Calculate statistics
    statistics = {
        'mean_sales': float(recent_data['sales'].mean()),
        'median_sales': float(recent_data['sales'].median()),
        'std_sales': float(recent_data['sales'].std()),
        'min_sales': float(recent_data['sales'].min()),
        'max_sales': float(recent_data['sales'].max()),
        'total_sales': float(recent_data['sales'].sum())
    }
    
    # Determine trend
    if len(recent_data) >= 30:
        first_half = recent_data.head(len(recent_data) // 2)['sales'].mean()
        second_half = recent_data.tail(len(recent_data) // 2)['sales'].mean()
        
        if second_half > first_half * 1.1:
            trend = "increasing"
        elif second_half < first_half * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    # Convert to list of dicts
    historical_data = recent_data[['date', 'sales']].to_dict('records')
    for record in historical_data:
        record['date'] = record['date'].strftime('%Y-%m-%d')
    
    return AnalyticsResponse(
        store=store,
        item=item,
        historical_data=historical_data,
        statistics=statistics,
        trend=trend
    )

@app.get("/forecast/{store}/{item}", tags=["Predictions"])
async def forecast_next_days(
    store: int = Query(..., ge=1, le=10),
    item: int = Query(..., ge=1, le=50),
    days: int = Query(7, ge=1, le=30, description="Number of days to forecast"),
    model_name: str = Query("lightgbm", description="Model to use")
):
    """
    Forecast sales for the next N days
    """
    if PROCESSED_DATA is None:
        raise HTTPException(
            status_code=503,
            detail="Data not loaded"
        )
    
    # Get last date in data
    last_date = PROCESSED_DATA['date'].max()
    
    # Generate predictions for next N days
    predictions = []
    for i in range(1, days + 1):
        forecast_date = (last_date + timedelta(days=i)).strftime('%Y-%m-%d')
        
        request = PredictionRequest(
            store=store,
            item=item,
            date=forecast_date,
            model_name=model_name
        )
        
        try:
            pred = await predict_inventory(request)
            predictions.append(pred)
        except Exception as e:
            print(f"Forecast error for {forecast_date}: {e}")
            continue
    
    return {
        "store": store,
        "item": item,
        "model": model_name,
        "forecast_days": days,
        "predictions": predictions
    }

# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500
    }

# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Inventory Prediction API...")
    print("Documentation available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
