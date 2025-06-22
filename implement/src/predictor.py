import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from .feature_engineering import create_features

def load_model(model_path):
    """
    Load a saved model from a pickle file.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: The loaded model object
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_future_dates(last_date, days_ahead=30):
    """
    Create a DataFrame with future dates for prediction.
    
    Args:
        last_date (str or datetime): The last date in the historical data
        days_ahead (int): Number of days to predict ahead
        
    Returns:
        pd.DataFrame: DataFrame with future dates
    """
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    future_dates = []
    for i in range(1, days_ahead + 1):
        future_dates.append(last_date + timedelta(days=i))
    
    future_df = pd.DataFrame({
        'date': future_dates,
        'daily_revenue': np.nan  # Placeholder for target variable
    })
    
    return future_df

def create_future_weeks(last_date, weeks_ahead=10):
    """
    Create a DataFrame with future weekly dates for prediction.
    
    Args:
        last_date (str or datetime): The last date in the historical data
        weeks_ahead (int): Number of weeks to predict ahead
        
    Returns:
        pd.DataFrame: DataFrame with future weekly dates
    """
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    future_dates = []
    for i in range(1, weeks_ahead + 1):
        future_dates.append(last_date + timedelta(weeks=i))
    
    future_df = pd.DataFrame({
        'date': future_dates,
        'daily_revenue': np.nan  # Placeholder for target variable
    })
    
    return future_df

def create_future_months(last_date, months_ahead=5):
    """
    Create a DataFrame with future monthly dates for prediction.
    
    Args:
        last_date (str or datetime): The last date in the historical data
        months_ahead (int): Number of months to predict ahead
        
    Returns:
        pd.DataFrame: DataFrame with future monthly dates
    """
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    future_dates = []
    for i in range(1, months_ahead + 1):
        # Use pandas date arithmetic to properly handle month increments
        next_month = last_date + pd.DateOffset(months=i)
        future_dates.append(next_month)
    
    future_df = pd.DataFrame({
        'date': future_dates,
        'daily_revenue': np.nan  # Placeholder for target variable
    })
    
    return future_df

def prepare_future_features(historical_df, future_df):
    """
    Prepare features for future predictions by combining historical and future data.
    
    Args:
        historical_df (pd.DataFrame): Historical data with features
        future_df (pd.DataFrame): Future dates DataFrame
        
    Returns:
        pd.DataFrame: Future data with engineered features
    """
    # Combine historical and future data temporarily for feature engineering
    combined_df = pd.concat([historical_df, future_df], ignore_index=True)
    
    # Create features for the combined dataset
    featured_combined = create_features(combined_df)
    
    # Extract only the future portion
    future_featured = featured_combined.tail(len(future_df)).copy()
    
    # Handle any remaining NaN values in lag features by forward filling
    # This is a simplification - in production, you might want more sophisticated handling
    future_featured = future_featured.fillna(method='ffill')
    
    return future_featured

def predict_future_sales(model, historical_df, days_ahead=30, target_column='daily_revenue'):
    """
    Predict future sales using a trained model.
    
    Args:
        model: Trained model object
        historical_df (pd.DataFrame): Historical data with date and revenue
        days_ahead (int): Number of days to predict ahead
        target_column (str): Name of the target variable column
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Get the last date from historical data
    if 'date' in historical_df.columns:
        last_date = historical_df['date'].max()
    else:
        last_date = historical_df.index.max()
    
    # Create future dates
    future_df = create_future_dates(last_date, days_ahead)
    
    # Prepare features for prediction
    future_featured = prepare_future_features(historical_df, future_df)
    
    # Get feature columns (exclude target column)
    feature_columns = [col for col in future_featured.columns if col != target_column and col != 'date']
    X_future = future_featured[feature_columns]
    
    # Make predictions
    predictions = model.predict(X_future)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'date': future_df['date'],
        'predicted_revenue': predictions
    })
    
    return results_df

def predict_future_weekly_sales(model, historical_df, weeks_ahead=10, target_column='daily_revenue'):
    """
    Predict future weekly sales using a trained model.
    
    Args:
        model: Trained model object
        historical_df (pd.DataFrame): Historical data with date and revenue
        weeks_ahead (int): Number of weeks to predict ahead
        target_column (str): Name of the target variable column
        
    Returns:
        pd.DataFrame: DataFrame with weekly predictions
    """
    # Get the last date from historical data
    if 'date' in historical_df.columns:
        last_date = historical_df['date'].max()
    else:
        last_date = historical_df.index.max()
    
    # Create future weekly dates
    future_df = create_future_weeks(last_date, weeks_ahead)
    
    # Prepare features for prediction
    future_featured = prepare_future_features(historical_df, future_df)
    
    # Get feature columns (exclude target column)
    feature_columns = [col for col in future_featured.columns if col != target_column and col != 'date']
    X_future = future_featured[feature_columns]
    
    # Make predictions
    predictions = model.predict(X_future)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'date': future_df['date'],
        'predicted_revenue': predictions
    })
    
    return results_df

def predict_future_monthly_sales(model, historical_df, months_ahead=5, target_column='daily_revenue'):
    """
    Predict future monthly sales using a trained model.
    
    Args:
        model: Trained model object
        historical_df (pd.DataFrame): Historical data with date and revenue
        months_ahead (int): Number of months to predict ahead
        target_column (str): Name of the target variable column
        
    Returns:
        pd.DataFrame: DataFrame with monthly predictions
    """
    # Get the last date from historical data
    if 'date' in historical_df.columns:
        last_date = historical_df['date'].max()
    else:
        last_date = historical_df.index.max()
    
    # Create future monthly dates
    future_df = create_future_months(last_date, months_ahead)
    
    # Prepare features for prediction
    future_featured = prepare_future_features(historical_df, future_df)
    
    # Get feature columns (exclude target column)
    feature_columns = [col for col in future_featured.columns if col != target_column and col != 'date']
    X_future = future_featured[feature_columns]
    
    # Make predictions
    predictions = model.predict(X_future)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'date': future_df['date'],
        'predicted_revenue': predictions
    })
    
    return results_df

def plot_predictions_with_history(historical_df, predictions_df, days_to_show=60):
    """
    Plot historical data alongside future predictions.
    
    Args:
        historical_df (pd.DataFrame): Historical data
        predictions_df (pd.DataFrame): Future predictions
        days_to_show (int): Number of historical days to show in the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Get recent historical data
    recent_history = historical_df.tail(days_to_show)
    
    # Plot historical data
    if 'date' in recent_history.columns:
        plt.plot(recent_history['date'], recent_history['daily_revenue'], 
                label='Historical Sales', color='blue', linewidth=2)
    else:
        plt.plot(recent_history.index, recent_history['daily_revenue'], 
                label='Historical Sales', color='blue', linewidth=2)
    
    # Plot predictions
    plt.plot(predictions_df['date'], predictions_df['predicted_revenue'], 
            label='Predicted Sales', color='red', linewidth=2, linestyle='--')
    
    # Add vertical line to separate history from predictions
    if 'date' in recent_history.columns:
        last_historical_date = recent_history['date'].max()
    else:
        last_historical_date = recent_history.index.max()
    
    plt.axvline(x=last_historical_date, color='gray', linestyle=':', alpha=0.7, 
                label='Prediction Start')
    
    plt.title("Historical Sales vs. Future Predictions", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Revenue ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def summarize_predictions(predictions_df):
    """
    Provide a summary of the predictions.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions
    """
    print("🔮 Future Sales Forecast Summary")
    print("=" * 50)
    print(f"Prediction Period: {predictions_df['date'].min().strftime('%Y-%m-%d')} to {predictions_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Number of Days: {len(predictions_df)}")
    print()
    print(f"📊 Revenue Predictions:")
    print(f"  • Average Daily Revenue: ${predictions_df['predicted_revenue'].mean():.2f}")
    print(f"  • Minimum Daily Revenue: ${predictions_df['predicted_revenue'].min():.2f}")
    print(f"  • Maximum Daily Revenue: ${predictions_df['predicted_revenue'].max():.2f}")
    print(f"  • Total Predicted Revenue: ${predictions_df['predicted_revenue'].sum():.2f}")
    print()
    print(f"📈 Weekly Breakdown:")
    predictions_with_week = predictions_df.copy()
    predictions_with_week['week'] = predictions_with_week['date'].dt.isocalendar().week
    weekly_summary = predictions_with_week.groupby('week')['predicted_revenue'].agg(['mean', 'sum']).round(2)
    for week, row in weekly_summary.iterrows():
        print(f"  • Week {week}: Avg ${row['mean']:.2f}/day, Total ${row['sum']:.2f}") 