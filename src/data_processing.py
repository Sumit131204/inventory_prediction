"""
Data processing utilities for inventory prediction project.

This module contains functions for:
- Data loading and validation
- Feature engineering
- Data preprocessing
- Train-test splitting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file and perform basic validation.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(filepath)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"Data loaded successfully: {df.shape}")
    return df


def create_time_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Create time-based features from date column.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Basic time features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_lag_features(df: pd.DataFrame, 
                       target_col: str,
                       group_cols: List[str],
                       lags: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
    """
    Create lag features for time series forecasting.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        group_cols: List of columns to group by
        lags: List of lag periods
        
    Returns:
        DataFrame with added lag features
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           target_col: str,
                           group_cols: List[str],
                           windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        group_cols: List of columns to group by
        windows: List of window sizes
        
    Returns:
        DataFrame with added rolling features
    """
    df = df.copy()
    
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'{target_col}_rolling_std_{window}'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price-related features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added price features
    """
    df = df.copy()
    
    if 'price' in df.columns and 'competitor_price' in df.columns:
        df['price_ratio'] = df['price'] / df['competitor_price']
        df['price_difference'] = df['price'] - df['competitor_price']
    
    if 'price' in df.columns and 'promotion' in df.columns:
        df['discounted_price'] = df['price'] * (1 - 0.1 * df['promotion'])
    
    if 'inventory_level' in df.columns and 'sales' in df.columns:
        df['inventory_to_sales_ratio'] = df['inventory_level'] / (df['sales'] + 1)
    
    return df


def encode_categorical_features(df: pd.DataFrame,
                                categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features using Label Encoding.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        
    Returns:
        Tuple of (DataFrame with encoded features, dictionary of encoders)
    """
    df = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
    
    return df, encoders


def split_train_test(df: pd.DataFrame,
                     date_col: str = 'date',
                     test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets based on time.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_date = df[date_col].quantile(1 - test_size)
    
    train_df = df[df[date_col] < split_date].copy()
    test_df = df[df[date_col] >= split_date].copy()
    
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df


def scale_features(X_train: pd.DataFrame,
                  X_test: pd.DataFrame = None) -> Tuple:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        
    Returns:
        Tuple of (scaler, X_train_scaled, X_test_scaled)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return scaler, X_train_scaled, X_test_scaled
    
    return scaler, X_train_scaled, None


def handle_missing_values(df: pd.DataFrame,
                         strategy: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('ffill', 'bfill', 'mean', 'median', 'zero')
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if strategy == 'ffill':
        df = df.fillna(method='ffill')
    elif strategy == 'bfill':
        df = df.fillna(method='bfill')
    elif strategy == 'mean':
        df = df.fillna(df.mean())
    elif strategy == 'median':
        df = df.fillna(df.median())
    elif strategy == 'zero':
        df = df.fillna(0)
    
    # Fill any remaining NaN with 0
    df = df.fillna(0)
    
    return df


def generate_sample_data(start_date: str = '2023-01-01',
                        end_date: str = '2024-12-31',
                        products: List[str] = None,
                        stores: List[str] = None) -> pd.DataFrame:
    """
    Generate sample sales data for testing.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        products: List of product names
        stores: List of store names
        
    Returns:
        DataFrame with sample sales data
    """
    if products is None:
        products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    
    if stores is None:
        stores = ['Store_1', 'Store_2', 'Store_3', 'Store_4']
    
    np.random.seed(42)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    for date in date_range:
        for product in products:
            for store in stores:
                base_sales = np.random.poisson(50)
                month_factor = 1 + 0.3 * np.sin(2 * np.pi * date.month / 12)
                weekend_factor = 1.2 if date.weekday() >= 5 else 1.0
                holiday_factor = 1.5 if date.month == 12 else 1.0
                
                sales = int(base_sales * month_factor * weekend_factor * holiday_factor)
                
                base_price = {'Product_A': 29.99, 'Product_B': 49.99, 'Product_C': 19.99,
                             'Product_D': 39.99, 'Product_E': 24.99}[product]
                price = base_price * np.random.uniform(0.9, 1.1)
                promotion = np.random.choice([0, 1], p=[0.8, 0.2])
                competitor_price = price * np.random.uniform(0.85, 1.15)
                
                data.append({
                    'date': date,
                    'product': product,
                    'store': store,
                    'sales': sales,
                    'price': round(price, 2),
                    'promotion': promotion,
                    'competitor_price': round(competitor_price, 2),
                    'inventory_level': sales + np.random.randint(10, 50)
                })
    
    return pd.DataFrame(data)
