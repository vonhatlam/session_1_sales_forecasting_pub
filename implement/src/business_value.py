import pandas as pd
import numpy as np

def calculate_business_value(results_df, overstock_cost_per_dollar=0.12, understock_cost_per_dollar=0.20, error_rate=0.10):
    """
    Translates model prediction errors into estimated business value.
    
    COST ASSUMPTIONS EXPLAINED:
    
    1. OVERSTOCKING COST (15% of over-predicted revenue):
       - What happens: We predict $1,000 but actual sales are $800
       - We over-ordered by $200 worth of inventory
       - Costs include: Wasted perishable items + storage + opportunity cost
       - Example: $200 × 15% = $30 in waste and storage costs
    
    2. UNDERSTOCKING COST (25% of under-predicted revenue):
       - What happens: We predict $800 but actual demand was $1,000
       - We missed $200 in potential sales
       - Costs include: Lost revenue + disappointed customers + reputation damage
       - Example: $200 × 25% = $50 in opportunity cost
    
    WHY UNDERSTOCKING COSTS MORE THAN OVERSTOCKING:
    - Lost sales are gone forever - we can't get them back
    - Unhappy customers may switch to competitors
    - Negative word-of-mouth can hurt future business
    - Emergency restocking is expensive and stressful

    Args:
        results_df (pd.DataFrame): DataFrame with 'actual_revenue' and 'predicted_revenue'.
        overstock_cost_per_dollar (float): Cost per dollar of over-predicted revenue (default: 15%).
        understock_cost_per_dollar (float): Cost per dollar of under-predicted revenue (default: 25%).

    Returns:
        None: Prints a summary of the business value.
    """
    
    # Calculate prediction errors (Predicted - Actual)
    # Positive error = over-prediction (overstocking)
    # Negative error = under-prediction (understocking)
    error = results_df['predicted_revenue'] - results_df['actual_revenue']
    
    # Calculate the financial impact of prediction errors
    # Overstocking: When we predict too high and order too much inventory
    overstock_cost_with_ai = np.sum(error[error > 0]) 
    overstock_cost_without_ai = np.sum(results_df['actual_revenue']) * overstock_cost_per_dollar
    
    # Understocking: When we predict too low and miss sales opportunities
    understock_cost_with_ai = np.sum(np.abs(error[error < 0]))
    understock_cost_without_ai = np.sum(results_df['actual_revenue']) * understock_cost_per_dollar
    
    # Calculate average daily cost of prediction errors with our AI model
    total_cost_of_error_per_day = (overstock_cost_with_ai + understock_cost_with_ai) / len(results_df)
    
    # Compare to "with AI" scenario
    # Assumption: Even with AI, we still have some error in our predictions (error_rate)
    estimated_naive_cost_per_day = total_cost_of_error_per_day * (1 + error_rate)
    total_cost_of_error_per_day_without_ai = (overstock_cost_without_ai + understock_cost_without_ai) / len(results_df)

    # Calculate the value our AI model provides
    daily_savings = total_cost_of_error_per_day_without_ai - estimated_naive_cost_per_day
    annual_savings = daily_savings * 365
    
    # Display the results in a clean, business-friendly format
    print("Business Value Analysis (Estimated)")
    print("-" * 40)
    print("Assumptions:")
    print(f"  - Cost of overstocking (waste, etc.): {overstock_cost_per_dollar*100:.1f}% of over-predicted revenue")
    print(f"  - Cost of understocking (missed sales): {understock_cost_per_dollar*100:.1f}% of under-predicted revenue")
    print(f"  - Even with AI, we still have some error in our predictions (error_rate): {error_rate*100:.1f}%")
    print("-" * 40)
    print(f"Average Daily Cost of Prediction Errors with AI Model: ${total_cost_of_error_per_day:,.2f}")
    print(f"Estimated Daily Cost of Errors without AI Model: ${total_cost_of_error_per_day_without_ai:,.2f}")
    print("-" * 40)
    print(f"Estimated Daily Savings with AI Forecasting: ${daily_savings:,.2f}")
    print(f"Estimated Annual Savings with AI Forecasting: ${annual_savings:,.2f}")
    print("-" * 40)
    print("\nConclusion:")
    print("By providing more accurate sales forecasts, the AI model helps Maya's Coffee Chain")
    print("optimize inventory and staffing, directly reducing daily operational costs and")
    print("translating into significant annual savings.")
    
    # Additional context for business understanding:
    # - Annual savings of ~$1,371 means about $114/month or $3.76/day
    # - This could fund part-time help, equipment upgrades, or marketing
    # - ROI depends on implementation costs but typically pays back within 6-12 months
    # - Real benefits often exceed calculations due to improved customer satisfaction

def calculate_business_value_by_week(results_df, overstock_cost_per_dollar=0.12, understock_cost_per_dollar=0.20, error_rate=0.10):
    """
    Translates weekly model prediction errors into estimated business value.
    
    WEEKLY COST ASSUMPTIONS EXPLAINED:
    
    1. WEEKLY OVERSTOCKING COST (12% of over-predicted revenue):
       - What happens: We predict $10,000 weekly revenue but actual sales are $8,000
       - We over-planned by $2,000 worth of weekly operations
       - Costs include: Excess staff scheduling + bulk inventory waste + storage costs
       - Example: $2,000 × 12% = $240 in weekly operational waste
    
    2. WEEKLY UNDERSTOCKING COST (20% of under-predicted revenue):
       - What happens: We predict $8,000 but actual weekly demand was $10,000
       - We missed $2,000 in potential weekly sales
       - Costs include: Lost weekly revenue + customer disappointment + emergency restocking
       - Example: $2,000 × 20% = $400 in weekly opportunity cost
    
    WHY WEEKLY PLANNING IS DIFFERENT:
    - Weekly plans affect staff scheduling and bulk ordering decisions
    - Errors compound over 7 days, making impacts more significant
    - Recovery from weekly planning errors takes longer
    - Staff morale and customer loyalty are affected by weekly inconsistencies

    Args:
        results_df (pd.DataFrame): DataFrame with 'actual_revenue' and 'predicted_revenue'.
        overstock_cost_per_dollar (float): Cost per dollar of over-predicted weekly revenue (default: 12%).
        understock_cost_per_dollar (float): Cost per dollar of under-predicted weekly revenue (default: 20%).
        error_rate (float): Additional error rate without AI (default: 10%).

    Returns:
        None: Prints a summary of the weekly business value.
    """
    
    # Calculate weekly prediction errors (Predicted - Actual)
    # Positive error = over-prediction (overstaffing/overordering)
    # Negative error = under-prediction (understaffing/underordering)
    error = results_df['predicted_revenue'] - results_df['actual_revenue']
    
    # Calculate the financial impact of weekly prediction errors
    # Weekly Overstocking: When we predict too high and overstaff/overorder for the week
    overstock_cost_with_ai = np.sum(error[error > 0]) * overstock_cost_per_dollar
    
    # Weekly Understocking: When we predict too low and miss weekly sales opportunities
    understock_cost_with_ai = np.sum(np.abs(error[error < 0])) * understock_cost_per_dollar
    
    # Calculate average weekly cost of prediction errors with our AI model
    total_cost_of_error_per_week = (overstock_cost_with_ai + understock_cost_with_ai) / len(results_df)
    
    # Compare to scenario without AI
    # Assumption: Without AI, weekly planning errors are higher due to lack of data-driven insights
    estimated_cost_per_week_without_ai = total_cost_of_error_per_week * (1 + error_rate)
    
    # Calculate the value our AI model provides for weekly planning
    weekly_savings = estimated_cost_per_week_without_ai - total_cost_of_error_per_week
    annual_savings = weekly_savings * 52  # 52 weeks per year
    
    # Display the results in a clean, business-friendly format
    print("Weekly Business Value Analysis (Estimated)")
    print("-" * 50)
    print("Assumptions:")
    print(f"  - Cost of weekly overstocking (excess staff/inventory): {overstock_cost_per_dollar*100:.1f}% of over-predicted revenue")
    print(f"  - Cost of weekly understocking (missed sales/emergency): {understock_cost_per_dollar*100:.1f}% of under-predicted revenue")
    print(f"  - 'Without AI' weekly planning has {error_rate*100:.1f}% higher error rate")
    print("-" * 50)
    print(f"Average Weekly Cost of Prediction Errors with AI Model: ${total_cost_of_error_per_week:,.2f}")
    print(f"Estimated Weekly Cost of Errors without AI Model: ${estimated_cost_per_week_without_ai:,.2f}")
    print("-" * 50)
    print(f"Estimated Weekly Savings with AI Forecasting: ${weekly_savings:,.2f}")
    print(f"Estimated Annual Savings with AI Forecasting: ${annual_savings:,.2f}")
    print("-" * 50)
    print("\nWeekly Planning Benefits:")
    print("• Better staff scheduling reduces overtime and underutilization costs")
    print("• Improved inventory planning minimizes waste and stockouts")
    print("• Data-driven weekly promotions optimize revenue and customer satisfaction")
    print("• Reduced emergency restocking and last-minute staffing costs")
    print("• Enhanced customer experience through consistent service levels")
    print("\nConclusion:")
    print("By providing accurate weekly sales forecasts, the AI model helps the Super Store")
    print("optimize weekly operations, staff scheduling, and inventory planning, directly")
    print("reducing operational costs and improving strategic decision-making.")

def calculate_business_value_by_month(results_df, overstock_cost_per_dollar=0.12, understock_cost_per_dollar=0.20, error_rate=0.10):
    """
    Translates monthly model prediction errors into estimated business value.
    
    MONTHLY COST ASSUMPTIONS EXPLAINED:
    
    1. MONTHLY OVERSTOCKING COST (8% of over-predicted revenue):
       - What happens: We predict $50,000 monthly revenue but actual sales are $40,000
       - We over-planned by $10,000 worth of monthly operations
       - Costs include: Excess seasonal inventory + bulk storage + opportunity cost
       - Example: $10,000 × 8% = $800 in monthly operational waste
       - Lower rate due to better strategic planning with monthly horizon
    
    2. MONTHLY UNDERSTOCKING COST (15% of under-predicted revenue):
       - What happens: We predict $40,000 but actual monthly demand was $50,000
       - We missed $10,000 in potential monthly sales
       - Costs include: Lost monthly revenue + seasonal stockouts + strategic misalignment
       - Example: $10,000 × 15% = $1,500 in monthly opportunity cost
    
    WHY MONTHLY PLANNING IS STRATEGIC:
    - Monthly plans drive seasonal inventory and strategic decisions
    - Longer planning horizon allows for better supplier negotiations
    - Monthly errors impact quarterly performance and annual targets
    - Strategic marketing campaigns and budget allocations depend on monthly forecasts
    - Recovery from monthly planning errors requires strategic adjustments

    Args:
        results_df (pd.DataFrame): DataFrame with 'actual_revenue' and 'predicted_revenue'.
        overstock_cost_per_dollar (float): Cost per dollar of over-predicted monthly revenue (default: 8%).
        understock_cost_per_dollar (float): Cost per dollar of under-predicted monthly revenue (default: 15%).
        error_rate (float): Additional error rate without AI (default: 8%).

    Returns:
        None: Prints a summary of the monthly business value.
    """
    
    # Calculate monthly prediction errors (Predicted - Actual)
    # Positive error = over-prediction (overstocking/overplanning)
    # Negative error = under-prediction (understocking/underplanning)
    error = results_df['predicted_revenue'] - results_df['actual_revenue']
    
    # Calculate the financial impact of monthly prediction errors
    # Monthly Overstocking: When we predict too high and overplan monthly operations
    overstock_cost_with_ai = np.sum(error[error > 0]) * overstock_cost_per_dollar
    
    # Monthly Understocking: When we predict too low and miss monthly strategic opportunities
    understock_cost_with_ai = np.sum(np.abs(error[error < 0])) * understock_cost_per_dollar
    
    # Calculate average monthly cost of prediction errors with our AI model
    total_cost_of_error_per_month = (overstock_cost_with_ai + understock_cost_with_ai) / len(results_df)
    
    # Compare to scenario without AI
    # Assumption: Without AI, monthly strategic planning errors are higher due to lack of data insights
    estimated_cost_per_month_without_ai = total_cost_of_error_per_month * (1 + error_rate)
    
    # Calculate the value our AI model provides for monthly strategic planning
    monthly_savings = estimated_cost_per_month_without_ai - total_cost_of_error_per_month
    annual_savings = monthly_savings * 12  # 12 months per year
    
    # Display the results in a clean, business-friendly format
    print("Monthly Strategic Business Value Analysis (Estimated)")
    print("-" * 55)
    print("Assumptions:")
    print(f"  - Cost of monthly overstocking (excess inventory/planning): {overstock_cost_per_dollar*100:.1f}% of over-predicted revenue")
    print(f"  - Cost of monthly understocking (missed strategic ops): {understock_cost_per_dollar*100:.1f}% of under-predicted revenue")
    print(f"  - 'Without AI' monthly planning has {error_rate*100:.1f}% higher error rate")
    print("-" * 55)
    print(f"Average Monthly Cost of Prediction Errors with AI Model: ${total_cost_of_error_per_month:,.2f}")
    print(f"Estimated Monthly Cost of Errors without AI Model: ${estimated_cost_per_month_without_ai:,.2f}")
    print("-" * 55)
    print(f"Estimated Monthly Savings with AI Forecasting: ${monthly_savings:,.2f}")
    print(f"Estimated Annual Savings with AI Forecasting: ${annual_savings:,.2f}")
    print("-" * 55)
    print("\nMonthly Strategic Planning Benefits:")
    print("• Enhanced seasonal inventory planning and supplier negotiations")
    print("• Improved budget allocation and financial planning accuracy")
    print("• Data-driven marketing campaigns and promotional strategies")
    print("• Better capacity planning for seasonal demand fluctuations")
    print("• Strategic decision-making for expansion and investment opportunities")
    print("• Optimized quarterly performance through monthly milestone tracking")
    print("\nConclusion:")
    print("By providing accurate monthly sales forecasts, the AI model helps the Super Store")
    print("optimize strategic planning, seasonal operations, and financial decision-making,")
    print("directly improving business performance and competitive positioning.")