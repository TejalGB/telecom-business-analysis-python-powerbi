# -*- coding: utf-8 -*-
"""
==============================================================================
TELECOM STRATEGIC BUSINESS PERFORMANCE & FORECASTING ANALYSIS
==============================================================================
Environment: Spyder / Python 3.x
Description: End-to-end data generation, ETL, KPI analysis, and linear regression
             forecasting for strategic telecom business insights.
==============================================================================
"""

# %% [1] Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("=" * 70)
print("Starting Telecom Strategic Analysis & Forecasting Pipeline...")
print("=" * 70)

# %% [2] Enterprise-Level Data Simulation
np.random.seed(42)

years = list(range(2016, 2023))
companies = ["Telecom_A", "Telecom_B", "Telecom_C"]
regions = ["North", "South", "East", "West"]
service_types = ["Prepaid", "Postpaid"]

raw_records = []

for company in companies:
    # Baseline metrics per company
    base_revenue = np.random.randint(22000, 32000)
    base_subscribers = np.random.uniform(12.0, 18.0)
    market_share = np.random.uniform(25.0, 38.0)
    
    current_rev = base_revenue
    current_subs = base_subscribers
    
    for year in years:
        # Yearly organic changes
        current_rev += np.random.randint(1800, 3800)
        current_subs += np.random.uniform(0.6, 1.8)
        churn_rate = np.random.uniform(6.0, 14.0)
        
        # Operational costs & profitability
        operating_cost = current_rev * np.random.uniform(0.58, 0.72)
        profit = current_rev - operating_cost
        ebitda_margin = (profit / current_rev) * 100
        
        # ARPU (Average Revenue Per User in Millions)
        arpu = current_rev / (current_subs * 1_000_000)
        
        region = np.random.choice(regions)
        service_type = np.random.choice(service_types)
        
        raw_records.append([
            year,
            company,
            region,
            service_type,
            round(current_rev, 2),
            round(operating_cost, 2),
            round(profit, 2),
            round(current_subs, 2),
            round(churn_rate, 2),
            round(arpu, 6),
            round(market_share, 2),
            round(ebitda_margin, 2)
        ])

columns = [
    "Year", "Company", "Region", "Service_Type", "Revenue",
    "Operating_Cost", "Profit", "Subscribers_Millions",
    "Churn_Rate", "ARPU", "Market_Share_%", "EBITDA_Margin_%"
]

df = pd.DataFrame(raw_records, columns=columns)
df.to_csv("telecom_raw_data.csv", index=False)
print("\n[OK] Raw dataset simulated and saved to 'telecom_raw_data.csv'.")
print(df.head())

# %% [3] ETL & Feature Engineering (Growth Rates)
# Calculate Year-over-Year (YoY) metrics
df["Revenue_Growth_%"] = df.groupby("Company")["Revenue"].pct_change() * 100
df["Subscriber_Growth_%"] = df.groupby("Company")["Subscribers_Millions"].pct_change() * 100

# Fill baseline starting year NaNs with 0.0
df["Revenue_Growth_%"] = df["Revenue_Growth_%"].fillna(0.0).round(2)
df["Subscriber_Growth_%"] = df["Subscriber_Growth_%"].fillna(0.0).round(2)

# Save cleaned & engineered dataset
df.to_csv("telecom_cleaned_data.csv", index=False)
df.to_csv("telecom_data.csv", index=False)
print("\n[OK] ETL completed: Growth rates engineered and saved to 'telecom_data.csv'.")

# %% [4] Business Summary & Aggregations
print("\n" + "=" * 50)
print("EXECUTIVE BUSINESS SUMMARY")
print("=" * 50)

# Revenue by Company across Years
revenue_pivot = pd.pivot_table(df, values="Revenue", index="Year", columns="Company")
print("\n--- Revenue Trajectory ($ in Millions) ---")
print(revenue_pivot)

# Profit by Region
regional_profit = df.groupby("Region")["Profit"].sum().round(2)
print("\n--- Cumulative Profit by Region ---")
print(regional_profit)

# Average EBITDA Margin & ARPU by Company
company_kpis = df.groupby("Company").agg({
    "EBITDA_Margin_%": "mean",
    "ARPU": "mean",
    "Churn_Rate": "mean"
}).round(2)
print("\n--- Key Performance Indicators by Operator ---")
print(company_kpis)

# %% [5] Exploratory Visualizations
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")

# Plot 1: Revenue Trends
plt.figure(figsize=(9, 5))
for company in companies:
    subset = df[df["Company"] == company]
    plt.plot(subset["Year"], subset["Revenue"], marker='o', linewidth=2.2, label=company)

plt.title("Telecom Revenue Trend (2016 - 2022)", fontsize=13, fontweight='bold')
plt.xlabel("Year", fontsize=11)
plt.ylabel("Revenue ($ Millions)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("Revenuetrend.png", dpi=300)
plt.show()

# Plot 2: Profit by Region
plt.figure(figsize=(8, 4.5))
regional_profit.plot(kind="bar", color="#2b5c8f", edgecolor="black", alpha=0.85)
plt.title("Cumulative Profit by Region", fontsize=13, fontweight='bold')
plt.xlabel("Region", fontsize=11)
plt.ylabel("Total Profit ($ Millions)", fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("ProfitbyRegion.png", dpi=300)
plt.show()

# %% [6] 5-Year Revenue Forecasting with Scikit-Learn Linear Regression
print("\n" + "=" * 50)
print("5-YEAR REVENUE FORECASTING (2023 - 2027)")
print("=" * 50)

future_years = np.array(range(2023, 2028)).reshape(-1, 1)
forecast_records = []

for company in companies:
    company_data = df[df["Company"] == company]
    X = company_data[["Year"]].values
    y = company_data["Revenue"].values
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Evaluate model fit (R^2 Score)
    fitted_values = model.predict(X)
    r2 = r2_score(y, fitted_values) * 100
    
    # Predict future revenue
    future_preds = model.predict(future_years)
    
    print(f"Company: {company:<10} | Model Fit R^2 Score: {r2:.2f}% | Annual Growth Slope: +${model.coef_[0]:,.2f}M/year")
    
    for yr, pred in zip(range(2023, 2028), future_preds):
        forecast_records.append([yr, company, round(pred, 2)])

forecast_df = pd.DataFrame(forecast_records, columns=["Year", "Company", "Forecasted_Revenue"])
forecast_df.to_csv("telecom_forecast_data.csv", index=False)
print("\n[OK] 5-Year forecast data saved to 'telecom_forecast_data.csv'.")
print(forecast_df.head(10))

print("\n" + "=" * 70)
print("Project pipeline executed successfully! All files and charts exported.")
print("=" * 70)