import pandas as pd
import os
from datetime import datetime

def create_weekly_sales_data():
    """
    Convert daily sales data to weekly aggregated data.
    Reads from sales_by_order_date.csv and creates sales_by_order_week.csv
    """
    # Define file paths
    input_file = '../sales_by_order_date.csv'
    output_file = '../sales_by_order_week.csv'
    
    try:
        # Read the daily sales data
        daily_sales = pd.read_csv(input_file)
        
        # Convert date column to datetime
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        
        # Create week start date (Monday of each week)
        daily_sales['week_start'] = daily_sales['date'].dt.to_period('W-MON').dt.start_time
        
        # Group by week and sum the daily revenue
        weekly_sales = daily_sales.groupby('week_start').agg({
            'daily_revenue': 'sum'
        }).reset_index()
        
        # Rename columns for clarity
        weekly_sales.columns = ['week_start_date', 'weekly_revenue']
        
        # Format the week_start_date as string for better readability
        weekly_sales['week_start_date'] = weekly_sales['week_start_date'].dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        weekly_sales.to_csv(output_file, index=False)
        
        print(f"Successfully created {output_file}")
        print(f"Original daily data points: {len(daily_sales)}")
        print(f"Weekly data points: {len(weekly_sales)}")
        print(f"Date range: {daily_sales['date'].min()} to {daily_sales['date'].max()}")
        
        # Display first few rows
        print("\nFirst 5 rows of weekly data:")
        print(weekly_sales.head())
        
        return weekly_sales
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return None
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

if __name__ == "__main__":
    create_weekly_sales_data() 