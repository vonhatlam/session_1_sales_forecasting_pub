import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

def perform_time_series_cv(model, df, n_splits=5, target_column='daily_revenue'):
    """
    Performs time series cross-validation using a given model.

    Args:
        model: An unfitted scikit-learn compatible model instance.
        df (pd.DataFrame): The feature-engineered DataFrame.
        n_splits (int): The number of splits for cross-validation.
        target_column (str): The name of the column to predict.

    Returns:
        pd.DataFrame: A DataFrame with the combined out-of-sample predictions.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]

    mae_scores = []
    rmse_scores = []
    all_results = []

    print(f"Performing Time Series Cross-Validation with {n_splits} splits...")
    print("-" * 60)
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"Fold {i+1}/{n_splits} | Train: {len(X_train)} | Test: {len(X_test)} | MAE: ${mae:<5.2f} | RMSE: ${rmse:<5.2f}")

        fold_results = pd.DataFrame({
            'actual_revenue': y_test,
            'predicted_revenue': predictions
        })
        all_results.append(fold_results)

    print("-" * 60)
    print(f"Average CV MAE: ${np.mean(mae_scores):.2f}")
    print(f"Average CV RMSE: ${np.mean(rmse_scores):.2f}")
    
    return pd.concat(all_results)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on the test set.

    Calculates and prints MAE and RMSE.

    Args:
        model: The trained machine learning model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target values.

    Returns:
        pd.DataFrame: A DataFrame with the actual values, predicted values, and error.
    """
    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Model Evaluation Results:")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print("-" * 30)
    print("Interpretation:")
    print(f"On average, the model's predictions are off by about ${mae:.2f}.")
    print("RMSE is higher because it penalizes larger errors more heavily.")

    # Create results dataframe
    results_df = pd.DataFrame({
        'actual_revenue': y_test,
        'predicted_revenue': predictions,
        'error': y_test - predictions
    })
    
    return results_df

def plot_predictions(results_df, title="Actual vs. Predicted Daily Revenue"):
    """
    Plots the actual vs. predicted revenue over time.

    Args:
        results_df (pd.DataFrame): DataFrame containing 'actual_revenue' and 'predicted_revenue'.
        title (str): Custom title for the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 6))

    results_df['actual_revenue'].plot(ax=ax, label='Actual Revenue', color='blue', lw=2)
    results_df['predicted_revenue'].plot(ax=ax, label='Predicted Revenue', color='orange', ls='--')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily Revenue ($)', fontsize=12)
    ax.legend()
    ax.grid(True)
    
    plt.show() 