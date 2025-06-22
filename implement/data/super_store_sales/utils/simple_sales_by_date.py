import pandas as pd
import numpy as np

def get_sales_by_order_date():
    """
    Extract sales by order date from super store sales data
    """
    try:
        # Read the CSV file
        print("Loading super store sales data...")
        df = pd.read_csv('super_store_sales.csv')
        
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Convert Order Date to datetime with proper parsing
        print("Converting Order Date to datetime...")
        df['Order Date'] = pd.to_datetime(df['Order Date'], infer_datetime_format=True)
        
        # Group by Order Date and sum sales
        print("Aggregating sales by order date...")
        sales_by_date = df.groupby('Order Date')['Sales'].agg(['sum', 'count']).reset_index()
        sales_by_date.columns = ['Order_Date', 'Total_Sales', 'Number_of_Orders']
        
        # Sort by date
        sales_by_date = sales_by_date.sort_values('Order_Date')
        
        # Display results
        print(f"\n=== SALES BY ORDER DATE ===")
        print(f"Total unique order dates: {len(sales_by_date)}")
        print(f"Date range: {sales_by_date['Order_Date'].min().strftime('%Y-%m-%d')} to {sales_by_date['Order_Date'].max().strftime('%Y-%m-%d')}")
        print(f"Total sales: ${sales_by_date['Total_Sales'].sum():,.2f}")
        print(f"Average daily sales: ${sales_by_date['Total_Sales'].mean():,.2f}")
        
        print(f"\n=== TOP 10 SALES DAYS ===")
        top_days = sales_by_date.sort_values('Total_Sales', ascending=False).head(10)
        for _, row in top_days.iterrows():
            print(f"{row['Order_Date'].strftime('%Y-%m-%d')}: ${row['Total_Sales']:,.2f} ({row['Number_of_Orders']} orders)")
        
        print(f"\n=== FIRST 10 DAYS ===")
        for _, row in sales_by_date.head(10).iterrows():
            print(f"{row['Order_Date'].strftime('%Y-%m-%d')}: ${row['Total_Sales']:,.2f} ({row['Number_of_Orders']} orders)")
        
        print(f"\n=== LAST 10 DAYS ===")
        for _, row in sales_by_date.tail(10).iterrows():
            print(f"{row['Order_Date'].strftime('%Y-%m-%d')}: ${row['Total_Sales']:,.2f} ({row['Number_of_Orders']} orders)")
        
        # Save to CSV
        output_file = 'sales_by_order_date.csv'
        sales_by_date.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Monthly summary
        print(f"\n=== MONTHLY SALES SUMMARY ===")
        sales_by_date['Year_Month'] = sales_by_date['Order_Date'].dt.to_period('M')
        monthly_sales = sales_by_date.groupby('Year_Month').agg({
            'Total_Sales': 'sum',
            'Number_of_Orders': 'sum'
        }).reset_index()
        
        for _, row in monthly_sales.iterrows():
            print(f"{row['Year_Month']}: ${row['Total_Sales']:,.2f} ({row['Number_of_Orders']} orders)")
        
        return sales_by_date
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    sales_data = get_sales_by_order_date() 