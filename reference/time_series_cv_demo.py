"""
Time Series Cross-Validation Demo
Focus on showing why the walk-forward approach is the ONLY correct way
to evaluate time series models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def create_trending_sales_data():
    """
    Create sales data with a clear trend and seasonal pattern
    This will make it obvious when we're cheating with future data
    """
    print("🎯 STEP 1: Creating Business Sales Data with Trend")
    
    # Create 24 months of data
    months = np.arange(1, 25)
    
    # Base trend: business is growing over time
    trend = months * 50 + 1000  # Growing by $50 per month, starting at $1000
    
    # Seasonal pattern: higher in Q4, lower in Q2
    seasonal = []
    for month in months:
        month_in_year = ((month - 1) % 12) + 1
        if month_in_year in [10, 11, 12]:  # Q4 - holiday season
            seasonal_boost = 400
        elif month_in_year in [4, 5, 6]:   # Q2 - slow season
            seasonal_boost = -200
        else:
            seasonal_boost = 0
        seasonal.append(seasonal_boost)
    
    # Combine trend + seasonal + some noise
    sales = trend + np.array(seasonal) + np.random.normal(0, 100, len(months))
    
    # Create DataFrame
    dates = pd.date_range('2022-01-01', periods=24, freq='M')
    df = pd.DataFrame({
        'month': months,
        'date': dates,
        'sales': sales,
        'trend': trend,
        'seasonal': seasonal
    })
    
    print("✅ Created 24 months of business sales data")
    print("   📈 Growing trend: +$50 per month")
    print("   🎄 Seasonal pattern: High in Q4, Low in Q2")
    print("\nData preview:")
    print(df[['month', 'date', 'sales']].head(8))
    
    return df

def visualize_data_pattern(df):
    """
    Show the data pattern clearly so the cheating becomes obvious
    """
    print("\n📊 STEP 2: Visualizing the Data Pattern")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Full time series
    plt.subplot(2, 2, 1)
    plt.plot(df['month'], df['sales'], 'o-', linewidth=2, markersize=6, color='blue')
    plt.plot(df['month'], df['trend'], '--', linewidth=2, color='red', alpha=0.7, label='Trend')
    plt.title('Sales Over Time\n(Clear upward trend + seasonal pattern)')
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Seasonal pattern by quarter
    df['quarter'] = ((df['month'] - 1) // 3) % 4 + 1
    quarter_avg = df.groupby('quarter')['sales'].mean()
    
    plt.subplot(2, 2, 2)
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    colors = ['lightblue', 'red', 'orange', 'green']
    bars = plt.bar(quarters, quarter_avg.values, color=colors, alpha=0.7)
    plt.title('Average Sales by Quarter\n(Q4 = Holiday Peak, Q2 = Slow Season)')
    plt.ylabel('Average Sales ($)')
    
    # Add value labels
    for bar, val in zip(bars, quarter_avg.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'${val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Year-over-year growth
    year1_avg = df[df['month'] <= 12]['sales'].mean()
    year2_avg = df[df['month'] > 12]['sales'].mean()
    
    plt.subplot(2, 2, 3)
    years = ['Year 1', 'Year 2']
    yearly_avg = [year1_avg, year2_avg]
    bars = plt.bar(years, yearly_avg, color=['lightcoral', 'lightgreen'], alpha=0.7)
    plt.title('Year-over-Year Growth\n(Business is clearly growing!)')
    plt.ylabel('Average Sales ($)')
    
    for bar, val in zip(bars, yearly_avg):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'${val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    growth = ((year2_avg - year1_avg) / year1_avg) * 100
    plt.text(0.5, max(yearly_avg) * 0.8, f'Growth: +{growth:.1f}%', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 4: Monthly growth rate
    plt.subplot(2, 2, 4)
    monthly_growth = df['sales'].pct_change() * 100
    plt.plot(df['month'][1:], monthly_growth[1:], 'o-', linewidth=2, markersize=4)
    plt.title('Month-over-Month Growth Rate\n(Volatile but trending up)')
    plt.xlabel('Month')
    plt.ylabel('Growth Rate (%)')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Key patterns visible:")
    print(f"   📈 Overall growth: +{growth:.1f}% year-over-year")
    print("   🔄 Seasonal cycles: Q4 high, Q2 low")
    print("   📊 This makes it easy to spot when we're cheating!")

def prepare_features(df):
    """
    Create simple features for prediction
    """
    print("\n🔧 STEP 3: Creating Features for Prediction")
    
    df_features = df.copy()
    
    # Simple features that would be available in real-time
    df_features['month_num'] = df_features['month']
    df_features['quarter'] = ((df_features['month'] - 1) // 3) % 4 + 1
    df_features['is_q4'] = (df_features['quarter'] == 4).astype(int)
    df_features['is_q2'] = (df_features['quarter'] == 2).astype(int)
    
    print("Features created:")
    print("   📅 month_num: Sequential month number")
    print("   📈 quarter: Quarter of the year (1-4)")
    print("   🎄 is_q4: Holiday season indicator")
    print("   📉 is_q2: Slow season indicator")
    
    return df_features

def wrong_cross_validation(df_features):
    """
    Demonstrate the WRONG way: Random K-Fold Cross-Validation
    """
    print("\n❌ STEP 4: The WRONG Way - Random Cross-Validation")
    print("=" * 50)
    
    # Prepare data
    X = df_features[['month_num', 'quarter', 'is_q4', 'is_q2']].values
    y = df_features['sales'].values
    
    # Wrong approach: Random KFold (shuffles the data)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("🔀 Random K-Fold Cross-Validation:")
    print("   - Randomly shuffles data into 5 folds")
    print("   - Uses future data to predict the past")
    print("   - This is CHEATING for time series!")
    
    # Show what the splits look like
    print("\n📋 What the random splits look like:")
    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        train_months = sorted(df_features.iloc[train_idx]['month'].tolist())
        test_months = sorted(df_features.iloc[test_idx]['month'].tolist())
        
        print(f"   Fold {fold_num}:")
        print(f"     Train: Months {train_months}")
        print(f"     Test:  Months {test_months}")
        
        # Show the cheating
        if max(test_months) < max(train_months):
            print(f"     ⚠️  CHEATING: Using Month {max(train_months)} to predict Month {min(test_months)}!")
    
    # Calculate scores
    model = LinearRegression()
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    wrong_mae = -cv_scores.mean()
    wrong_std = cv_scores.std()
    
    print(f"\n📊 Random CV Results:")
    print(f"   Average MAE: ${wrong_mae:.0f} ± ${wrong_std:.0f}")
    print(f"   Looks amazing! But it's cheating...")
    
    return wrong_mae, wrong_std

def correct_cross_validation(df_features):
    """
    Demonstrate the CORRECT way: Time Series Split (Walk-Forward)
    """
    print("\n✅ STEP 5: The CORRECT Way - Time Series Cross-Validation")
    print("=" * 50)
    
    # Prepare data
    X = df_features[['month_num', 'quarter', 'is_q4', 'is_q2']].values
    y = df_features['sales'].values
    
    # Correct approach: TimeSeriesSplit (respects chronological order)
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("⏰ Time Series Cross-Validation (Walk-Forward):")
    print("   - Respects chronological order")
    print("   - Only uses past data to predict future")
    print("   - This simulates real-world deployment!")
    
    # Show what the splits look like
    print("\n📋 What the time series splits look like:")
    fold_maes = []
    
    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        train_months = df_features.iloc[train_idx]['month'].tolist()
        test_months = df_features.iloc[test_idx]['month'].tolist()
        
        print(f"   Fold {fold_num}:")
        print(f"     Train: Months {min(train_months)}-{max(train_months)} ({len(train_months)} months)")
        print(f"     Test:  Months {min(test_months)}-{max(test_months)} ({len(test_months)} months)")
        print(f"     ✅ Correct: Only using past to predict future")
        
        # Train and evaluate for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fold_mae = mean_absolute_error(y_test, pred)
        fold_maes.append(fold_mae)
        
        print(f"     MAE: ${fold_mae:.0f}")
    
    correct_mae = np.mean(fold_maes)
    correct_std = np.std(fold_maes)
    
    print(f"\n📊 Time Series CV Results:")
    print(f"   Average MAE: ${correct_mae:.0f} ± ${correct_std:.0f}")
    print(f"   This reflects real-world performance!")
    
    return correct_mae, correct_std, fold_maes

def visualize_cv_comparison(df_features, wrong_mae, correct_mae, fold_maes):
    """
    Visualize the difference between wrong and correct cross-validation
    """
    print("\n📈 STEP 6: Visualizing the Difference")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data timeline with split visualization
    plt.subplot(2, 2, 1)
    
    # Show time series splits visually
    X = df_features[['month_num', 'quarter', 'is_q4', 'is_q2']].values
    tscv = TimeSeriesSplit(n_splits=5)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X)):
        train_months = df_features.iloc[train_idx]['month']
        test_months = df_features.iloc[test_idx]['month']
        
        # Plot training data
        plt.scatter(train_months, [fold_num] * len(train_months), 
                   c=colors[fold_num], alpha=0.6, s=30, marker='s', label=f'Fold {fold_num+1} Train' if fold_num == 0 else "")
        
        # Plot test data
        plt.scatter(test_months, [fold_num] * len(test_months), 
                   c=colors[fold_num], alpha=1.0, s=60, marker='o', label=f'Fold {fold_num+1} Test' if fold_num == 0 else "")
    
    plt.title('Time Series Cross-Validation Splits\n(Squares=Train, Circles=Test)')
    plt.xlabel('Month')
    plt.ylabel('Fold Number')
    plt.yticks(range(5), [f'Fold {i+1}' for i in range(5)])
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Performance comparison
    plt.subplot(2, 2, 2)
    methods = ['Random CV\n(WRONG)', 'Time Series CV\n(CORRECT)']
    errors = [wrong_mae, correct_mae]
    colors = ['red', 'green']
    
    bars = plt.bar(methods, errors, color=colors, alpha=0.7)
    plt.title('Cross-Validation Results Comparison\n(Lower MAE = Better)')
    plt.ylabel('Mean Absolute Error ($)')
    
    for bar, error in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'${error:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Add reality check annotation
    diff_pct = ((correct_mae - wrong_mae) / wrong_mae) * 100
    plt.text(0.5, max(errors) * 0.7, f'Reality Check:\n+{diff_pct:.1f}% higher error\n(More realistic!)', 
             ha='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 3: Fold-by-fold performance
    plt.subplot(2, 2, 3)
    fold_numbers = range(1, 6)
    plt.plot(fold_numbers, fold_maes, 'o-', linewidth=2, markersize=8, color='green')
    plt.title('Performance by Fold\n(Shows model adaptation over time)')
    plt.xlabel('Fold Number (Time →)')
    plt.ylabel('MAE ($)')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(fold_numbers, fold_maes, 1)
    p = np.poly1d(z)
    plt.plot(fold_numbers, p(fold_numbers), "--", alpha=0.8, color='red')
    
    if z[0] > 0:
        trend_text = "↗️ Getting harder"
    else:
        trend_text = "↘️ Getting easier"
    plt.text(3, max(fold_maes) * 0.9, trend_text, ha='center', fontsize=10, fontweight='bold')
    
    # Plot 4: Real-world simulation
    plt.subplot(2, 2, 4)
    
    # Simulate what happens in production
    months = df_features['month'].values
    sales = df_features['sales'].values
    
    plt.plot(months, sales, 'o-', linewidth=2, markersize=4, alpha=0.7, label='Actual Sales')
    
    # Show prediction points from last fold
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]  # Last fold
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = df_features['sales'].iloc[train_idx], df_features['sales'].iloc[test_idx]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    test_months = df_features['month'].iloc[test_idx]
    plt.plot(test_months, pred, 's-', linewidth=2, markersize=8, color='red', label='Predictions')
    
    plt.axvline(x=max(train_idx)+1, color='black', linestyle='--', alpha=0.7, label='Prediction Point')
    plt.title('Real-World Simulation\n(Last Fold: Predict Future from Past)')
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_data_leakage():
    """
    Show a concrete example of data leakage with random CV
    """
    print("\n🚨 STEP 7: Concrete Example of Data Leakage")
    print("=" * 50)
    
    print("📋 Scenario: It's January 2023, and you want to predict February 2023 sales")
    print("\n❌ What Random CV might do:")
    print("   Training data: Jan 2023, Mar 2023, May 2023, Jul 2023, Sep 2023...")
    print("   Test data: Feb 2023")
    print("   🚨 PROBLEM: Using March-September 2023 data to predict February 2023!")
    print("   💭 In real life, you don't have March data when predicting February!")
    
    print("\n✅ What Time Series CV does:")
    print("   Training data: Jan 2022, Feb 2022, ..., Dec 2022, Jan 2023")
    print("   Test data: Feb 2023")
    print("   ✅ CORRECT: Only using data that existed before February 2023!")
    
    print("\n🎯 Why this matters:")
    print("   - Random CV gives false confidence (overly optimistic results)")
    print("   - Time Series CV gives realistic expectations")
    print("   - In production, realistic expectations = better business decisions")

def business_impact_analysis(wrong_mae, correct_mae):
    """
    Analyze the business impact of using wrong vs correct validation
    """
    print("\n💼 STEP 8: Business Impact Analysis")
    print("=" * 50)
    
    # Calculate the difference
    difference = correct_mae - wrong_mae
    pct_increase = (difference / wrong_mae) * 100
    
    print(f"📊 The Numbers:")
    print(f"   Random CV MAE: ${wrong_mae:.0f}")
    print(f"   Correct CV MAE: ${correct_mae:.0f}")
    print(f"   Difference: ${difference:.0f} ({pct_increase:.1f}% higher)")
    
    print(f"\n💰 Business Translation:")
    
    if pct_increase > 20:
        risk_level = "🔴 HIGH RISK"
        impact = "SEVERE"
    elif pct_increase > 10:
        risk_level = "🟡 MEDIUM RISK"
        impact = "SIGNIFICANT"
    else:
        risk_level = "🟢 LOW RISK"
        impact = "MINOR"
    
    print(f"   Risk Level: {risk_level}")
    print(f"   Impact: {impact}")
    
    # Business scenarios
    print(f"\n🏢 Real-World Scenarios:")
    print(f"   📦 Inventory Planning: Off by ${difference:.0f} per month")
    print(f"   💵 Budget Forecasting: ${difference * 12:.0f} annual error")
    print(f"   👥 Staffing Decisions: Wrong headcount planning")
    print(f"   📈 Investor Reports: Misleading growth projections")
    
    print(f"\n🎯 Key Insight:")
    if pct_increase > 15:
        print(f"   Using wrong validation could lead to major business mistakes!")
        print(f"   Your model performs {pct_increase:.1f}% worse than you think.")
    else:
        print(f"   While the difference is smaller, correct validation still matters.")
        print(f"   Always simulate real-world conditions for honest assessment.")

def main():
    """
    Run the complete Time Series Cross-Validation demonstration
    """
    print("🚀 TIME SERIES CROSS-VALIDATION DEMO")
    print("=" * 60)
    print("Learn why walk-forward validation is the ONLY correct way")
    print("to evaluate time series models!")
    print("=" * 60)
    
    # Step 1: Create realistic business data
    df = create_trending_sales_data()
    
    # Step 2: Visualize the patterns
    visualize_data_pattern(df)
    
    # Step 3: Prepare features
    df_features = prepare_features(df)
    
    # Step 4: Show the wrong way
    wrong_mae, wrong_std = wrong_cross_validation(df_features)
    
    # Step 5: Show the correct way
    correct_mae, correct_std, fold_maes = correct_cross_validation(df_features)
    
    # Step 6: Visual comparison
    visualize_cv_comparison(df_features, wrong_mae, correct_mae, fold_maes)
    
    # Step 7: Data leakage explanation
    demonstrate_data_leakage()
    
    # Step 8: Business impact
    business_impact_analysis(wrong_mae, correct_mae)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. ⏰ Time series data has a natural order - NEVER shuffle it!")
    print("2. 🔀 Random cross-validation = CHEATING (uses future to predict past)")
    print("3. ➡️  Time series cross-validation = HONEST (walk-forward approach)")
    print("4. 📊 Correct validation gives realistic performance estimates")
    print("5. 💼 Realistic estimates = better business decisions")
    print("6. 🎯 Always simulate real-world deployment conditions!")
    
    difference_pct = ((correct_mae - wrong_mae) / wrong_mae) * 100
    print(f"\n💡 Bottom Line:")
    print(f"   Your model is likely {difference_pct:.1f}% worse than random CV suggests.")
    print(f"   Time Series CV tells you the truth - use it!")
    print("=" * 60)

if __name__ == "__main__":
    main() 