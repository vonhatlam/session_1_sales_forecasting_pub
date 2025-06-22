"""
Utility script to convert daily sales data to monthly aggregated data.
"""

import pandas as pd
from datetime import datetime
from typing import Optional
import os


def create_monthly_sales_data() -> Optional[pd.DataFrame]:
    """
    Convert daily sales data to monthly aggregated data.
    
    Reads from sales_by_order_date.csv and creates sales_by_order_month.csv
    with monthly revenue totals.
    
    Returns:
        Optional[pd.DataFrame]: Monthly sales dataframe if successful, None if failed
        
    Raises:
        FileNotFoundError: If the input CSV file is not found
        pd.errors.EmptyDataError: If the input CSV file is empty
        ValueError: If date conversion fails
    """
    # Define file paths
    input_file = '../sales_by_order_date.csv'
    output_file = '../sales_by_order_month.csv'
    
    try:
        # Read the daily sales data
        print(f"Reading daily sales data from {input_file}...")
        daily_sales = pd.read_csv(input_file)
        
        if daily_sales.empty:
            raise pd.errors.EmptyDataError("Input CSV file is empty")
        
        # Convert date column to datetime
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        
        # Create month start date (first day of each month)
        daily_sales['month_start'] = daily_sales['date'].dt.to_period('M').dt.start_time
        
        # Group by month and sum the daily revenue
        monthly_sales = daily_sales.groupby('month_start').agg({
            'daily_revenue': 'sum'
        }).reset_index()
        
        # Rename columns for clarity
        monthly_sales.columns = ['month_start_date', 'monthly_revenue']
        
        # Format the month_start_date as string for better readability
        monthly_sales['month_start_date'] = monthly_sales['month_start_date'].dt.strftime('%Y-%m-%d')
        
        # Round the monthly revenue to 2 decimal places
        monthly_sales['monthly_revenue'] = monthly_sales['monthly_revenue'].round(2)
        
        # Save to CSV
        monthly_sales.to_csv(output_file, index=False)
        
        print(f"✓ Successfully created {output_file}")
        print(f"📊 Original daily data points: {len(daily_sales)}")
        print(f"📊 Monthly data points: {len(monthly_sales)}")
        print(f"📅 Date range: {daily_sales['date'].min().strftime('%Y-%m-%d')} to {daily_sales['date'].max().strftime('%Y-%m-%d')}")
        
        # Display first few rows
        print("\n📋 First 5 rows of monthly data:")
        print(monthly_sales.head().to_string(index=False))
        
        # Display summary statistics
        print(f"\n📈 Monthly Revenue Statistics:")
        print(f"   Average: ${monthly_sales['monthly_revenue'].mean():,.2f}")
        print(f"   Minimum: ${monthly_sales['monthly_revenue'].min():,.2f}")
        print(f"   Maximum: ${monthly_sales['monthly_revenue'].max():,.2f}")
        print(f"   Total: ${monthly_sales['monthly_revenue'].sum():,.2f}")
        
        return monthly_sales
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        print("   Please ensure the sales_by_order_date.csv file exists in the parent directory.")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"❌ Error: {str(e)}")
        return None
    except ValueError as e:
        print(f"❌ Error converting dates: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error processing data: {str(e)}")
        return None


if __name__ == "__main__":
    create_monthly_sales_data() 