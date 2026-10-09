# -*- coding: utf-8 -*-
"""
==============================================================================
TELECOM STRATEGIC BUSINESS PERFORMANCE & FORECASTING ANALYSIS
==============================================================================
Environment: Python 3.x
Methodology: Multi-dimensional synthetic scenario simulation, ETL, KPI analysis,
             and time-based holdout validated revenue trend forecasting.
==============================================================================
"""

# %% [1] Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 75)
print("Starting Telecom Strategic Performance & Forecasting Pipeline...")
print("=" * 75)

# %% [2] Multi-Dimensional Synthetic Scenario Simulation
# True Grain: Company x Year x Region x Service_Type (3 x 7 x 4 x 2 = 168 rows)
np.random.seed(42)

years = list(range(2016, 2023))
companies = ["Telecom_A", "Telecom_B", "Telecom_C"]
regions = ["North", "South", "East", "West"]
service_types = ["Prepaid", "Postpaid"]

# Baseline annual subscriber allocation (Millions) and growth rates by company
company_profiles = {
    "Telecom_A": {"base_subs": 14.5, "growth_rate": 0.065, "postpaid_share": 0.45, "market_share": 38.5},
    "Telecom_B": {"base_subs": 18.0, "growth_rate": 0.050, "postpaid_share": 0.35, "market_share": 34.0},
    "Telecom_C": {"base_subs": 12.0, "growth_rate": 0.080, "postpaid_share": 0.30, "market_share": 27.5},
}

# Regional subscriber distribution weights (North, South, East, West)
region_weights = {"North": 0.28, "South": 0.24, "East": 0.22, "West": 0.26}

records = []

for company, profile in company_profiles.items():
    total_subs = profile["base_subs"]
    
    for year_idx, year in enumerate(years):
        # Organic annual subscriber growth with slight random variation
        if year_idx > 0:
            growth_noise = np.random.uniform(-0.01, 0.02)
            total_subs *= (1.0 + profile["growth_rate"] + growth_noise)
        
        for region, r_weight in region_weights.items():
            regional_subs = total_subs * r_weight * np.random.uniform(0.95, 1.05)
            
            for service in service_types:
                # Service mix breakdown
                if service == "Postpaid":
                    subs = regional_subs * profile["postpaid_share"]
                    # Postpaid: Higher ARPU ($28 - $42/month), Lower Churn (1.5% - 3.5%)
                    monthly_arpu = np.random.uniform(28.0, 42.0)
                    churn_rate = np.random.uniform(1.5, 3.5)
                else:  # Prepaid
                    subs = regional_subs * (1.0 - profile["postpaid_share"])
                    # Prepaid: Lower ARPU ($9 - $18/month), Higher Churn (5.0% - 9.5%)
                    monthly_arpu = np.random.uniform(9.0, 18.0)
                    churn_rate = np.random.uniform(5.0, 9.5)
                
                # Annual Revenue ($ Millions) = Subscribers (Millions) * Monthly ARPU ($) * 12 months
                annual_revenue = subs * monthly_arpu * 12.0
                
                # Operating Cost (60% to 74% of revenue depending on scale and service)
                cost_ratio = np.random.uniform(0.60, 0.72) if service == "Postpaid" else np.random.uniform(0.64, 0.75)
                operating_cost = annual_revenue * cost_ratio
                profit = annual_revenue - operating_cost
                ebitda_margin = (profit / annual_revenue) * 100.0
                
                # ARPU Metrics
                # Annual ARPU ($/subscriber/year) = Revenue ($M) / Subscribers (M)
                arpu_annual = annual_revenue / subs
                arpu_monthly = arpu_annual / 12.0
                
                records.append([
                    year,
                    company,
                    region,
                    service,
                    round(annual_revenue, 2),
                    round(operating_cost, 2),
                    round(profit, 2),
                    round(subs, 3),
                    round(churn_rate, 2),
                    round(arpu_monthly, 2),   # Primary ARPU for Power BI
                    round(profile["market_share"], 2),
                    round(ebitda_margin, 2),
                    round(arpu_monthly, 2),
                    round(arpu_annual, 2)
                ])

columns = [
    "Year", "Company", "Region", "Service_Type", "Revenue",
    "Operating_Cost", "Profit", "Subscribers_Millions",
    "Churn_Rate", "ARPU", "Market_Share_%", "EBITDA_Margin_%",
    "Monthly_ARPU", "Annual_ARPU"
]

df = pd.DataFrame(records, columns=columns)
df.to_csv("telecom_raw_data.csv", index=False)
print(f"\n[OK] Raw multi-dimensional dataset created with {len(df)} records.")
print(f"     Grain: Company x Year x Region x Service Type (Saved to 'telecom_raw_data.csv').")

# %% [3] ETL & Feature Engineering (Aggregations & Growth Rates)
# Calculate Year-over-Year Growth at the Company-Year aggregate level
company_yearly = df.groupby(["Company", "Year"]).agg({
    "Revenue": "sum",
    "Profit": "sum",
    "Subscribers_Millions": "sum"
}).reset_index()

company_yearly["Revenue_Growth_%"] = company_yearly.groupby("Company")["Revenue"].pct_change() * 100.0
company_yearly["Subscriber_Growth_%"] = company_yearly.groupby("Company")["Subscribers_Millions"].pct_change() * 100.0
company_yearly["Revenue_Growth_%"] = company_yearly["Revenue_Growth_%"].fillna(0.0).round(2)
company_yearly["Subscriber_Growth_%"] = company_yearly["Subscriber_Growth_%"].fillna(0.0).round(2)

# Merge growth rates back into the granular dataset
df = df.merge(
    company_yearly[["Company", "Year", "Revenue_Growth_%", "Subscriber_Growth_%"]],
    on=["Company", "Year"],
    how="left"
)

# Reorder columns to exactly match Power BI expected schema
final_columns = [
    "Year", "Company", "Region", "Service_Type", "Revenue",
    "Operating_Cost", "Profit", "Subscribers_Millions",
    "Churn_Rate", "ARPU", "Market_Share_%", "EBITDA_Margin_%",
    "Revenue_Growth_%", "Subscriber_Growth_%", "Monthly_ARPU", "Annual_ARPU"
]
df = df[final_columns]

# Save primary cleaned dataset consumed by Power BI
df.to_csv("telecom_cleaned_data.csv", index=False)
df.to_csv("telecom_data.csv", index=False)
print(f"[OK] ETL Completed: Engineered growth rates and saved 'telecom_data.csv' ({len(df)} rows).")

# %% [4] Business Summary & Exploratory Aggregations
print("\n" + "=" * 55)
print("EXECUTIVE BUSINESS SUMMARY (Aggregated Across Segments)")
print("=" * 55)

# Annual Company Totals
print("\n--- Annual Total Revenue ($ Millions) by Operator ---")
print(pd.pivot_table(df, values="Revenue", index="Year", columns="Company", aggfunc="sum").round(2))

# Regional Profitability Breakdown
regional_summary = df.groupby("Region").agg({
    "Revenue": "sum",
    "Profit": "sum",
    "Subscribers_Millions": "mean"
}).round(2)
regional_summary["Profit_Margin_%"] = ((regional_summary["Profit"] / regional_summary["Revenue"]) * 100).round(2)
print("\n--- Regional Financial Performance ---")
print(regional_summary)

# Service Tier Dynamics (Prepaid vs. Postpaid)
service_summary = df.groupby("Service_Type").agg({
    "Revenue": "sum",
    "Profit": "sum",
    "Monthly_ARPU": "mean",
    "Churn_Rate": "mean"
}).round(2)
print("\n--- Service Tier Economics (Prepaid vs. Postpaid) ---")
print(service_summary)

# %% [5] Exploratory Visualizations
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")

# Plot 1: Total Annual Revenue Trend by Company
yearly_revenue = df.groupby(["Year", "Company"])["Revenue"].sum().unstack()
plt.figure(figsize=(9, 5))
for company in companies:
    plt.plot(yearly_revenue.index, yearly_revenue[company], marker='o', linewidth=2.2, label=company)

plt.title("Total Annual Revenue Trend by Operator (2016 - 2022)", fontsize=13, fontweight='bold')
plt.xlabel("Year", fontsize=11)
plt.ylabel("Total Revenue ($ Millions)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("Revenuetrend.png", dpi=300)
plt.show()

# Plot 2: Total Profit by Region
plt.figure(figsize=(8, 4.5))
regional_profit = df.groupby("Region")["Profit"].sum()
regional_profit.plot(kind="bar", color="#2b5c8f", edgecolor="black", alpha=0.85)
plt.title("Cumulative Profit by Region (2016 - 2022)", fontsize=13, fontweight='bold')
plt.xlabel("Region", fontsize=11)
plt.ylabel("Total Profit ($ Millions)", fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("ProfitbyRegion.png", dpi=300)
plt.show()

# %% [6] Time-Based Holdout Validation & 5-Year Trend Forecasting
print("\n" + "=" * 65)
print("TIME-BASED FORECAST VALIDATION & 5-YEAR PROJECTION")
print("=" * 65)

train_years = list(range(2016, 2021))
test_years = [2021, 2022]
future_years = np.array(range(2023, 2028)).reshape(-1, 1)

forecast_records = []

print("\n--- Out-of-Sample Validation Results (Test Holdout: 2021-2022) ---")

for company in companies:
    comp_totals = df[df["Company"] == company].groupby("Year")["Revenue"].sum().reset_index()
    
    # Train / Test split
    train_data = comp_totals[comp_totals["Year"].isin(train_years)]
    test_data = comp_totals[comp_totals["Year"].isin(test_years)]
    
    X_train = train_data[["Year"]].values
    y_train = train_data["Revenue"].values
    X_test = test_data[["Year"]].values
    y_test = test_data["Revenue"].values
    
    # Fit model on training set
    val_model = LinearRegression()
    val_model.fit(X_train, y_train)
    test_preds = val_model.predict(X_test)
    
    # Calculate Out-of-Sample Metrics
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100.0
    
    print(f"Operator: {company:<10} | Test MAE: ${test_mae:,.2f}M | Test RMSE: ${test_rmse:,.2f}M | Test MAPE: {test_mape:.2f}%")
    
    # Full Historical Fit for Future 5-Year Trend Projection (2023-2027)
    X_full = comp_totals[["Year"]].values
    y_full = comp_totals["Revenue"].values
    
    full_model = LinearRegression()
    full_model.fit(X_full, y_full)
    future_preds = full_model.predict(future_years)
    
    for yr, pred in zip(range(2023, 2028), future_preds):
        forecast_records.append([
            yr,
            company,
            round(pred, 2),
            round(test_mae, 2),
            round(test_mape, 2)
        ])

forecast_df = pd.DataFrame(forecast_records, columns=[
    "Year", "Company", "Forecasted_Revenue", "Holdout_MAE", "Holdout_MAPE_%"
])
forecast_df.to_csv("telecom_forecast_data.csv", index=False)
print("\n[OK] 5-Year revenue forecast with holdout validation metrics saved to 'telecom_forecast_data.csv'.")

print("\n" + "=" * 75)
print("Pipeline execution complete! Multi-grain datasets and validated forecasts ready.")
print("=" * 75)