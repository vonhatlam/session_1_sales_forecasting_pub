import pandas as pd
import numpy as np
import os

def generate_sales_data(file_path="data/maya_coffee_sales.csv"):
    """
    Generates synthetic daily sales data for Maya's Coffee Chain for three years.

    The data includes:
    - A baseline daily revenue.
    - A steady growth trend over time.
    - Weekly seasonality (higher revenue on weekends).
    - Yearly seasonality (higher revenue in colder months).
    - Random noise to simulate real-world unpredictability.

    Args:
        file_path (str): The path to save the generated CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the generated sales data.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Generate a date range for three years
    dates = pd.to_datetime(pd.date_range(start="2021-01-01", end="2023-12-31", freq='D'))
    num_days = len(dates)

    # Base revenue and growth trend
    base_revenue = 500
    growth_rate = 0.1
    time_trend = np.linspace(0, num_days * growth_rate, num_days)

    # Weekly seasonality (higher on weekends)
    day_of_week_effect = dates.dayofweek.isin([5, 6]) * 150  # Sat, Sun

    # Yearly seasonality (higher in colder months)
    month_effect = np.sin((dates.month - 1) * (2 * np.pi / 12)) * -100 + 50

    # Random noise
    noise = np.random.normal(0, 50, num_days)

    # Combine all components to create the daily revenue
    daily_revenue = base_revenue + time_trend + day_of_week_effect + month_effect + noise
    daily_revenue = np.maximum(daily_revenue, 100) # Ensure revenue is not negative

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'daily_revenue': daily_revenue.round(2)
    })
    
    df.to_csv(file_path, index=False)
    print(f"Successfully generated and saved sales data to {file_path}")
    return df

if __name__ == '__main__':
    # When running this script directly, generate the data in the default location
    generate_sales_data("session_1_sales_forecasting/data/maya_coffee_sales.csv") 