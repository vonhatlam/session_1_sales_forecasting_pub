# %% [markdown]
# # Maya's Coffee Chain: A Story of AI-Powered Sales Forecasting
#
# This notebook tells the story of how Maya's Coffee Chain can use AI to predict daily sales. We will compare two models, evaluate them with a robust method, and translate the results into real-world business value.

# %% [markdown]
# ## Step 1 & 2: Data Generation and Exploration
# First, we generate and explore three years of realistic sales data, which includes growth, weekly cycles, and yearly seasonality.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Import our custom modules
from src.data_generator import generate_sales_data
from src.feature_engineering import create_features
from src.model_trainer import train_linear_regression, train_random_forest, train_final_model, save_model
from src.evaluation import evaluate_model, perform_time_series_cv, plot_predictions
from src.business_value import calculate_business_value

# Define file paths
DATA_DIR = 'data/maya_coffee_sales'
MODEL_DIR = 'models'
DATA_FILE = os.path.join(DATA_DIR, 'maya_coffee_sales.csv')
MODEL_FILE = os.path.join(MODEL_DIR, 'maya_coffee_forecaster.pkl')

# Generate and load data
# generate_sales_data(DATA_FILE)
df = pd.read_csv(DATA_FILE, parse_dates=['date'])
df.set_index('date', inplace=True)

# Plot the raw data
plt.style.use('seaborn-v0_8-whitegrid')
df['daily_revenue'].plot(figsize=(15, 6), title="Maya's Coffee Chain - Daily Revenue Over 3 Years")
plt.ylabel('Daily Revenue ($)')
plt.show()

# %% [markdown]
# ## Step 3: Feature Engineering - Giving the AI Memory and Context
#
# An AI model doesn't understand a date like `2023-10-26` on its own. We need to translate time into a language it can understand. This is called **feature engineering**.
#
# Our `create_features` function does this for us. It creates:
#
# - **Cyclical Features**: Represents time in a circle (like a clock), so the model knows December is close to January.
# - **Lag Features**: Gives the model explicit memory. For example, what were the sales yesterday? Or on the same day last week?
# - **Rolling Averages**: Smooths out noise and shows the underlying trend (e.g., the average sales over the last 7 days).
#
# Let's apply this function and see our new, feature-rich dataset.

# %%
featured_df = create_features(df.reset_index())
print("Columns in our new dataset:")
print(featured_df.columns)
print("\nFirst 5 rows of the new data:")
print(featured_df.head())

# Save the featured dataframe to a csv file
FEATURED_DATA_FILE = os.path.join(DATA_DIR, 'maya_coffee_sales_featured.csv')
featured_df.to_csv(FEATURED_DATA_FILE, index=False)
print(f"\nFeatured data saved to {FEATURED_DATA_FILE}")

# %% [markdown]
# ## Step 4: Model Showdown - A Simple Train-Test Split
# We'll start by doing a quick evaluation of two models using a simple split: training on all data except the last 90 days, which are used for testing.

# %% [markdown]
# ### Model 1: Linear Regression (The Baseline)
# A simple, fast model that finds linear relationships in the data.

# %%
lr_model, X_test_lr, y_test_lr = train_linear_regression(featured_df)
lr_results = evaluate_model(lr_model, X_test_lr, y_test_lr)
plot_predictions(lr_results, title="Linear Regression Model - Actual vs. Predicted Daily Revenue (Test Set)")

# Save Linear Regression results
LR_RESULTS_FILE = os.path.join(DATA_DIR, 'linear_regression_predictions.csv')
lr_results.to_csv(LR_RESULTS_FILE, index=True)
print(f"Linear Regression predictions saved to {LR_RESULTS_FILE}")

# %% [markdown]
# ### Model 2: Random Forest (The Challenger)
# A more powerful model that can capture complex, non-linear patterns by building multiple decision trees.

# %%
rf_model, X_test_rf, y_test_rf = train_random_forest(featured_df)
rf_results = evaluate_model(rf_model, X_test_rf, y_test_rf)
plot_predictions(rf_results, title="Random Forest Model - Actual vs. Predicted Daily Revenue (Test Set)")

# Save Random Forest results
RF_RESULTS_FILE = os.path.join(DATA_DIR, 'random_forest_predictions.csv')
rf_results.to_csv(RF_RESULTS_FILE, index=True)
print(f"Random Forest predictions saved to {RF_RESULTS_FILE}")

# %% [markdown]
# The Random Forest appears to perform better, but a single test period can be misleading.

# %% [markdown]
# ## Step 5: The Ultimate Test - Time-Series Cross-Validation
# This is a more robust method that mimics reality. It splits the data into multiple, sequential train-and-test "folds" to get a more reliable performance estimate.

# %% [markdown]
# #### Cross-Validation: Linear Regression

# %%
print("--- Running CV for Linear Regression ---")
lr_cv_model = LinearRegression()
lr_cv_results = perform_time_series_cv(lr_cv_model, featured_df)

# Save Linear Regression CV results
LR_CV_RESULTS_FILE = os.path.join(DATA_DIR, 'linear_regression_cv_predictions.csv')
lr_cv_results.to_csv(LR_CV_RESULTS_FILE, index=True)
print(f"Linear Regression CV predictions saved to {LR_CV_RESULTS_FILE}")

# %% [markdown]
# #### Cross-Validation: Random Forest

# %%
print("\n--- Running CV for Random Forest ---")
rf_cv_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_cv_results = perform_time_series_cv(rf_cv_model, featured_df)

# Save Random Forest CV results
RF_CV_RESULTS_FILE = os.path.join(DATA_DIR, 'random_forest_cv_predictions.csv')
rf_cv_results.to_csv(RF_CV_RESULTS_FILE, index=True)
print(f"Random Forest CV predictions saved to {RF_CV_RESULTS_FILE}")

# %% [markdown]
# ## Step 6: Crowning the Champion & Training the Final Model
# The cross-validation results confirm that the **Random Forest** model is significantly more accurate. Now, we train this winning model on the *entire* dataset to make it as powerful as possible for future predictions. This is the model we would save and deploy.

# %%
final_model = train_final_model(featured_df, model_type='lr') # model_type can be 'rf' (Random Forest) or 'lr' (Linear Regression)
save_model(final_model, MODEL_FILE)

# %% [markdown]
# ## Step 7: The Payoff - Business Value
# Finally, we translate the robust performance of our champion model (the Random Forest) into tangible business value. The calculation is based on the reliable error rates from our cross-validation.

# %%
# Add 'error' column for business value calculation
rf_cv_results['error'] = rf_cv_results['actual_revenue'] - rf_cv_results['predicted_revenue']
calculate_business_value(rf_cv_results)

# %% [markdown]
# ## Conclusion: From Data to Dollars
# We have successfully compared two models, used a robust validation technique to choose the clear winner, and trained a final, production-ready model. Most importantly, we've shown that the Random Forest model can provide significant annual savings for Maya's Coffee Chain by enabling smarter, data-driven inventory and staffing decisions. 