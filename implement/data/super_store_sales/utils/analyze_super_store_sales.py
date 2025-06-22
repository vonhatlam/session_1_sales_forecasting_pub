import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_super_store_sales():
    """
    Analyze super store sales data and extract sales by order date
    """
    try:
        # Read the CSV file
        print("Loading super store sales data...")
        df = pd.read_csv('super_store_sales.csv')
        
        # Display basic info about the dataset
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Check for date columns
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'order' in col.lower()]
        print(f"\nPotential date columns: {date_columns}")
        
        # Check data types
        print(f"\nData types:")
        print(df.dtypes)
        
        # Look for sales columns
        sales_columns = [col for col in df.columns if 'sale' in col.lower() or 'revenue' in col.lower() or 'amount' in col.lower()]
        print(f"\nPotential sales columns: {sales_columns}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_sales_by_order_date(df):
    """
    Extract and aggregate sales by order date
    """
    if df is None:
        return None
        
    try:
        # Try to identify the order date column
        order_date_col = None
        for col in df.columns:
            if 'order' in col.lower() and 'date' in col.lower():
                order_date_col = col
                break
        
        if order_date_col is None:
            # Look for any date column
            for col in df.columns:
                if 'date' in col.lower():
                    order_date_col = col
                    break
        
        if order_date_col is None:
            print("No date column found. Available columns:")
            print(list(df.columns))
            return None
        
        print(f"Using '{order_date_col}' as order date column")
        
        # Convert to datetime
        df[order_date_col] = pd.to_datetime(df[order_date_col])
        
        # Try to identify sales column
        sales_col = None
        for col in df.columns:
            if 'sale' in col.lower() or 'revenue' in col.lower() or 'amount' in col.lower():
                sales_col = col
                break
        
        if sales_col is None:
            print("No sales column found. Available columns:")
            print(list(df.columns))
            return None
        
        print(f"Using '{sales_col}' as sales column")
        
        # Aggregate sales by order date
        sales_by_date = df.groupby(order_date_col.date if hasattr(df[order_date_col], 'date') else order_date_col)[sales_col].sum().reset_index()
        sales_by_date.columns = ['Order_Date', 'Total_Sales']
        
        # Sort by date
        sales_by_date = sales_by_date.sort_values('Order_Date')
        
        print(f"\nSales by Order Date (first 10 rows):")
        print(sales_by_date.head(10))
        
        print(f"\nSales by Order Date (last 10 rows):")
        print(sales_by_date.tail(10))
        
        print(f"\nSummary Statistics:")
        print(f"Total Sales: ${sales_by_date['Total_Sales'].sum():,.2f}")
        print(f"Average Daily Sales: ${sales_by_date['Total_Sales'].mean():,.2f}")
        print(f"Date Range: {sales_by_date['Order_Date'].min()} to {sales_by_date['Order_Date'].max()}")
        print(f"Number of unique dates: {len(sales_by_date)}")
        
        # Save results to CSV
        output_file = 'sales_by_order_date.csv'
        sales_by_date.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return sales_by_date
        
    except Exception as e:
        print(f"Error processing sales by date: {e}")
        return None

def create_sales_visualization(sales_by_date):
    """
    Create visualizations for sales by order date
    """
    if sales_by_date is None or len(sales_by_date) == 0:
        return
    
    try:
        # Convert Order_Date to datetime for plotting
        sales_by_date['Order_Date'] = pd.to_datetime(sales_by_date['Order_Date'])
        
        plt.figure(figsize=(15, 8))
        
        # Time series plot
        plt.subplot(2, 1, 1)
        plt.plot(sales_by_date['Order_Date'], sales_by_date['Total_Sales'], linewidth=1)
        plt.title('Sales by Order Date - Time Series')
        plt.xlabel('Order Date')
        plt.ylabel('Total Sales ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Monthly aggregation
        sales_by_date['Year_Month'] = sales_by_date['Order_Date'].dt.to_period('M')
        monthly_sales = sales_by_date.groupby('Year_Month')['Total_Sales'].sum().reset_index()
        monthly_sales['Year_Month'] = monthly_sales['Year_Month'].astype(str)
        
        plt.subplot(2, 1, 2)
        plt.bar(range(len(monthly_sales)), monthly_sales['Total_Sales'])
        plt.title('Monthly Sales Totals')
        plt.xlabel('Month')
        plt.ylabel('Total Sales ($)')
        plt.xticks(range(len(monthly_sales)), monthly_sales['Year_Month'], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sales_by_order_date_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as: sales_by_order_date_visualization.png")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    print("=== Super Store Sales Analysis ===")
    
    # Load and analyze the data
    df = analyze_super_store_sales()
    
    if df is not None:
        # Get sales by order date
        sales_by_date = get_sales_by_order_date(df)
        
        # Create visualization
        if sales_by_date is not None:
            create_sales_visualization(sales_by_date)
        
        print("\n=== Analysis Complete ===")
    else:
        print("Failed to load the data. Please check the file path and format.") 