"""
Simple Cyclical Encoding Demo
Focus on showing the power of cyclical features for seasonal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

def create_simple_seasonal_data():
    """
    Create simple seasonal sales data that clearly shows cyclical patterns
    """
    print("🎯 STEP 1: Creating Seasonal Sales Data")
    
    # Create 2 years of monthly data
    months = list(range(1, 25))  # 24 months (2 years)
    
    # Simple seasonal pattern: higher sales in winter (Nov, Dec, Jan, Feb)
    seasonal_sales = []
    for month in months:
        month_in_year = ((month - 1) % 12) + 1  # Convert to 1-12
        
        if month_in_year in [11, 12, 1, 2]:  # Winter months
            base_sales = 1500  # Higher winter sales
        elif month_in_year in [6, 7, 8]:     # Summer months  
            base_sales = 800   # Lower summer sales
        else:                                # Spring/Fall
            base_sales = 1200  # Medium sales
            
        # Add some random variation
        sales = base_sales + np.random.normal(0, 100)
        seasonal_sales.append(sales)
    
    # Create DataFrame
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] * 2
    
    df = pd.DataFrame({
        'month_number': months,
        'month_name': month_names,
        'sales': seasonal_sales
    })
    
    print("✅ Created 24 months of seasonal sales data")
    print("\nData preview:")
    print(df.head(8))
    
    return df

def show_seasonal_pattern(df):
    """
    Visualize the seasonal pattern to make it clear
    """
    print("\n📊 STEP 2: Visualizing the Seasonal Pattern")
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Sales over time
    plt.subplot(1, 2, 1)
    plt.plot(df['month_number'], df['sales'], 'o-', linewidth=2, markersize=6)
    plt.title('Sales Over Time\n(Notice the repeating pattern!)')
    plt.xlabel('Month Number')
    plt.ylabel('Sales ($)')
    plt.grid(True, alpha=0.3)
    
    # Add season labels
    for i in range(0, 24, 6):
        plt.axvline(x=i+1, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Average sales by month of year
    monthly_avg = df.groupby(df['month_number'].apply(lambda x: ((x-1) % 12) + 1))['sales'].mean()
    
    plt.subplot(1, 2, 2)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    bars = plt.bar(range(1, 13), monthly_avg.values)
    plt.title('Average Sales by Month\n(Winter = High, Summer = Low)')
    plt.xlabel('Month')
    plt.ylabel('Average Sales ($)')
    plt.xticks(range(1, 13), month_labels, rotation=45)
    
    # Color code seasons
    colors = ['blue', 'blue', 'green', 'green', 'green', 'red', 
              'red', 'red', 'orange', 'orange', 'blue', 'blue']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ You can clearly see the seasonal pattern!")
    print("   🔵 Blue = Winter (High sales)")
    print("   🔴 Red = Summer (Low sales)")
    print("   🟢 Green = Spring (Medium sales)")
    print("   🟠 Orange = Fall (Medium sales)")

def create_regular_features(df):
    """
    Create regular (non-cyclical) month features
    """
    print("\n🔢 STEP 3: Creating Regular Month Features")
    
    df_regular = df.copy()
    df_regular['month_regular'] = df_regular['month_number'].apply(lambda x: ((x-1) % 12) + 1)
    
    print("Regular encoding:")
    print("January = 1, February = 2, ..., December = 12")
    print("\n❌ Problem: December (12) and January (1) seem completely different!")
    print("But in business, they're both winter months with similar sales patterns.")
    
    return df_regular

def create_cyclical_features(df):
    """
    Create cyclical month features
    """
    print("\n⭕ STEP 4: Creating Cyclical Month Features")
    
    df_cyclical = df.copy()
    
    # Convert month to cyclical features
    df_cyclical['month_regular'] = df_cyclical['month_number'].apply(lambda x: ((x-1) % 12) + 1)
    df_cyclical['month_sin'] = df_cyclical['month_regular'].apply(lambda x: np.sin(2 * np.pi * x / 12))
    df_cyclical['month_cos'] = df_cyclical['month_regular'].apply(lambda x: np.cos(2 * np.pi * x / 12))
    
    print("Cyclical encoding using sin and cos:")
    print("Think of months arranged on a clock face!")
    
    # Show the transformation for key months
    key_months = [1, 6, 12]  # Jan, Jun, Dec
    for month in key_months:
        sin_val = np.sin(2 * np.pi * month / 12)
        cos_val = np.cos(2 * np.pi * month / 12)
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
        print(f"{month_name}: sin={sin_val:.2f}, cos={cos_val:.2f}")
    
    print("\n✅ Now December and January have similar coordinates!")
    
    return df_cyclical

def compare_encoding_methods(df_regular, df_cyclical):
    """
    Compare regular vs cyclical encoding visually
    """
    print("\n🔍 STEP 5: Comparing Encoding Methods")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Regular encoding
    plt.subplot(2, 2, 1)
    months_in_year = df_regular['month_regular']
    sales = df_regular['sales']
    
    plt.scatter(months_in_year, sales, alpha=0.7, s=50)
    plt.title('Regular Encoding\n(December=12, January=1 seem unrelated)')
    plt.xlabel('Month Number')
    plt.ylabel('Sales ($)')
    plt.xticks(range(1, 13))
    
    # Plot 2: Cyclical encoding visualization
    plt.subplot(2, 2, 2)
    month_sin = df_cyclical['month_sin']
    month_cos = df_cyclical['month_cos']
    
    # Create a scatter plot in cyclical space
    scatter = plt.scatter(month_cos, month_sin, c=sales, s=80, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Sales ($)')
    plt.title('Cyclical Encoding\n(December and January are close!)')
    plt.xlabel('Cosine (Winter↔Summer axis)')
    plt.ylabel('Sine (Spring↔Fall axis)')
    
    # Add month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i in range(12):
        month_num = i + 1
        sin_val = np.sin(2 * np.pi * month_num / 12)
        cos_val = np.cos(2 * np.pi * month_num / 12)
        plt.annotate(month_labels[i], (cos_val, sin_val), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Distance comparison
    plt.subplot(2, 1, 2)
    
    # Calculate distances between December and January
    dec_regular = 12
    jan_regular = 1
    regular_distance = abs(dec_regular - jan_regular)
    
    dec_sin, dec_cos = np.sin(2 * np.pi * 12 / 12), np.cos(2 * np.pi * 12 / 12)
    jan_sin, jan_cos = np.sin(2 * np.pi * 1 / 12), np.cos(2 * np.pi * 1 / 12)
    cyclical_distance = np.sqrt((dec_sin - jan_sin)**2 + (dec_cos - jan_cos)**2)
    
    distances = [regular_distance, cyclical_distance]
    methods = ['Regular\nEncoding', 'Cyclical\nEncoding']
    colors = ['red', 'green']
    
    bars = plt.bar(methods, distances, color=colors, alpha=0.7)
    plt.title('Distance Between December and January\n(Lower = More Similar)')
    plt.ylabel('Distance')
    
    # Add value labels
    for bar, distance in zip(bars, distances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{distance:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Distance Analysis:")
    print(f"   Regular encoding: {regular_distance:.2f} (December and January seem very different)")
    print(f"   Cyclical encoding: {cyclical_distance:.2f} (December and January are close neighbors)")

def train_and_compare_models(df_regular, df_cyclical):
    """
    Train models with both encoding methods and compare performance
    """
    print("\n🤖 STEP 6: Training Models to Compare Performance")
    
    # Prepare data for both methods
    X_regular = df_regular[['month_regular']].values
    X_cyclical = df_cyclical[['month_sin', 'month_cos']].values
    y = df_regular['sales'].values
    
    # Split data: first 18 months for training, last 6 months for testing
    split_point = 18
    
    X_regular_train, X_regular_test = X_regular[:split_point], X_regular[split_point:]
    X_cyclical_train, X_cyclical_test = X_cyclical[:split_point], X_cyclical[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Train models
    model_regular = LinearRegression()
    model_cyclical = LinearRegression()
    
    model_regular.fit(X_regular_train, y_train)
    model_cyclical.fit(X_cyclical_train, y_train)
    
    # Make predictions
    pred_regular = model_regular.predict(X_regular_test)
    pred_cyclical = model_cyclical.predict(X_cyclical_test)
    
    # Calculate performance
    mae_regular = mean_absolute_error(y_test, pred_regular)
    mae_cyclical = mean_absolute_error(y_test, pred_cyclical)
    
    r2_regular = r2_score(y_test, pred_regular)
    r2_cyclical = r2_score(y_test, pred_cyclical)
    
    # Show results
    print("🏆 PERFORMANCE COMPARISON:")
    print(f"Regular Encoding:")
    print(f"   Average Error: ${mae_regular:.0f}")
    print(f"   R² Score: {r2_regular:.3f} ({r2_regular*100:.1f}% patterns explained)")
    
    print(f"\nCyclical Encoding:")
    print(f"   Average Error: ${mae_cyclical:.0f}")
    print(f"   R² Score: {r2_cyclical:.3f} ({r2_cyclical*100:.1f}% patterns explained)")
    
    improvement = ((mae_regular - mae_cyclical) / mae_regular) * 100
    print(f"\n🎯 Improvement: {improvement:.1f}% better accuracy with cyclical encoding!")
    
    # Visualize predictions
    plt.figure(figsize=(15, 5))
    
    test_months = range(19, 25)  # Last 6 months
    
    plt.subplot(1, 2, 1)
    plt.plot(test_months, y_test, 'o-', label='Actual Sales', linewidth=2, markersize=8)
    plt.plot(test_months, pred_regular, 's-', label='Regular Encoding', linewidth=2, markersize=8)
    plt.plot(test_months, pred_cyclical, '^-', label='Cyclical Encoding', linewidth=2, markersize=8)
    plt.title('Predictions Comparison\n(Last 6 months)')
    plt.xlabel('Month Number')
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    methods = ['Regular\nEncoding', 'Cyclical\nEncoding']
    errors = [mae_regular, mae_cyclical]
    colors = ['red', 'green']
    
    bars = plt.bar(methods, errors, color=colors, alpha=0.7)
    plt.title('Average Prediction Error\n(Lower is Better)')
    plt.ylabel('Mean Absolute Error ($)')
    
    for bar, error in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'${error:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'regular_mae': mae_regular,
        'cyclical_mae': mae_cyclical,
        'improvement': improvement
    }

def main():
    """
    Run the complete cyclical encoding demonstration
    """
    print("🚀 CYCLICAL ENCODING DEMO")
    print("=" * 50)
    print("Learn why cyclical features are powerful for seasonal business patterns!")
    print("=" * 50)
    
    # Step 1: Create data
    df = create_simple_seasonal_data()
    
    # Step 2: Show the pattern
    show_seasonal_pattern(df)
    
    # Step 3: Create regular features
    df_regular = create_regular_features(df)
    
    # Step 4: Create cyclical features
    df_cyclical = create_cyclical_features(df)
    
    # Step 5: Compare visually
    compare_encoding_methods(df_regular, df_cyclical)
    
    # Step 6: Train models and compare
    results = train_and_compare_models(df_regular, df_cyclical)
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 KEY TAKEAWAYS:")
    print("=" * 50)
    print("1. 📅 Time has natural cycles - seasons, weeks, days")
    print("2. 🔢 Regular encoding treats December(12) and January(1) as very different")
    print("3. ⭕ Cyclical encoding understands they're neighboring months")
    print(f"4. 📈 Result: {results['improvement']:.1f}% better prediction accuracy!")
    print("5. 💼 Perfect for seasonal businesses: retail, restaurants, tourism")
    
    print(f"\n💡 Bottom Line: Cyclical features help AI understand that time repeats!")
    print("   December → January is like 11:59 PM → 12:01 AM")
    print("=" * 50)

if __name__ == "__main__":
    main() 