import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_linear_regression(df, target_column='daily_revenue', test_size=90): # test_size is the number of days to hold out for testing
    """
    Splits the data and trains a Linear Regression model.
    """
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]

    # Time-based split
    # Total number of days is 1095, so we hold out the last 90 days for testing
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Training complete.")

    return model, X_test, y_test

def train_random_forest(df, target_column='daily_revenue', test_size=90): # test_size is the number of days to hold out for testing
    """
    Splits the data, trains a Random Forest Regressor, and returns the model.
    """
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]

    # Time-based split
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    print(f"Training Random Forest with data shape: {X_train.shape}")
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)

    print("Model training completed.")
    return model, X_test, y_test

def train_final_model(df, model_type='rf', target_column='daily_revenue'):
    """
    Trains a final model on the entire dataset.
    """
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]

    print(f"Training final {model_type.upper()} model on entire dataset (shape: {X.shape})...")

    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    elif model_type == 'lr':
        model = LinearRegression()
    else:
        model = LinearRegression()
    
    model.fit(X, y)
    print("Final model training completed.")
    return model

def save_model(model, file_path="models/maya_coffee_forecaster.pkl"):
    """
    Saves the trained model to a file using pickle.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {file_path}") 