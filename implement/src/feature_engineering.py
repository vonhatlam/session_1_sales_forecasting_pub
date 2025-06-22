import pandas as pd
import numpy as np

def create_features(df):
    """
    Engineers features from the sales data to prepare it for modeling.

    This function creates:
    - Time-based features (day of week, month, year).
    - Cyclical features for day of week and month to represent time's circular nature.
    - Lag features to give the model memory of past sales.
    - Rolling average features to capture recent trends and smooth out noise.

    Args:
        df (pd.DataFrame): The input DataFrame with 'date' and 'daily_revenue' columns.

    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """
    df = df.copy()
    # df['date'] = pd.to_datetime(df['date'])
    # df = df.set_index('date')
    # Check if 'date' column exists, otherwise use 'week_start_date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif 'week_start_date' in df.columns:
        df['week_start_date'] = pd.to_datetime(df['week_start_date'])
        df = df.set_index('week_start_date')
    else:
        raise ValueError("Neither 'date' nor 'week_start_date' column found in DataFrame")

    # Simple time-based features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear

    # Cyclical features
    # These help the model understand the cyclical nature of time (e.g., Dec is close to Jan)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features (memory of past values)
    # Using past sales data to predict future sales
    df['lag_1'] = df['daily_revenue'].shift(1)  # Previous day's sales
    df['lag_7'] = df['daily_revenue'].shift(7)  # Sales from the same day last week
    df['lag_30'] = df['daily_revenue'].shift(30) # Sales from the same day last month

    # Rolling average features (smoothed trends)
    # These help capture the underlying trend by smoothing out daily fluctuations
    df['rolling_mean_7'] = df['daily_revenue'].rolling(window=7).mean().shift(1)
    df['rolling_mean_30'] = df['daily_revenue'].rolling(window=30).mean().shift(1)
    
    # Drop rows with NaN values created by lags and rolling averages
    df = df.dropna()

    return df 