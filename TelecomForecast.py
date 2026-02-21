# -*- coding: utf-8 -*-

# ==========================================================
# TELECOM STRATEGIC BUSINESS PERFORMANCE PROJECT
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("\nStarting Telecom Strategic Analysis Project...\n")

# ----------------------------------------------------------
# 1️⃣ DATA CREATION (Enterprise-Level Simulation)
# ----------------------------------------------------------

np.random.seed(42)

years = list(range(2016, 2023))
companies = ["Telecom_A", "Telecom_B", "Telecom_C"]
regions = ["North", "South", "East", "West"]
service_types = ["Prepaid", "Postpaid"]

data = []

for company in companies:
    
    revenue = np.random.randint(20000, 30000)
    subscribers = np.random.randint(10, 20)
    market_share = np.random.uniform(20, 40)
    
    for year in years:
        
        revenue += np.random.randint(1500, 4000)
        subscribers += np.random.uniform(0.5, 2.0)
        churn = np.random.uniform(5, 15)
        
        operating_cost = revenue * np.random.uniform(0.55, 0.75)
        profit = revenue - operating_cost
        ebitda_margin = (profit / revenue) * 100
        arpu = revenue / (subscribers * 1_000_000)
        
        region = np.random.choice(regions)
        service_type = np.random.choice(service_types)
        
        data.append([
            year,
            company,
            region,
            service_type,
            round(revenue,2),
            round(operating_cost,2),
            round(profit,2),
            round(subscribers,2),
            round(churn,2),
            round(arpu,6),
            round(market_share,2),
            round(ebitda_margin,2)
        ])

df = pd.DataFrame(data, columns=[
    "Year",
    "Company",
    "Region",
    "Service_Type",
    "Revenue",
    "Operating_Cost",
    "Profit",
    "Subscribers_Millions",
    "Churn_Rate",
    "ARPU",
    "Market_Share_%",
    "EBITDA_Margin_%"
])

df.to_csv("telecom_raw_data.csv", index=False)

print("Enterprise-Level Dataset Created.\n")
print(df.head())

# ----------------------------------------------------------
# 2️⃣ ETL PROCESS
# ----------------------------------------------------------

df.dropna(inplace=True)

df["Revenue_Growth_%"] = df.groupby("Company")["Revenue"].pct_change() * 100
df["Subscriber_Growth_%"] = df.groupby("Company")["Subscribers_Millions"].pct_change() * 100

df.fillna(0, inplace=True)

df.to_csv("telecom_cleaned_data.csv", index=False)

print("\nETL Completed.\n")

# ----------------------------------------------------------
# 3️⃣ OLAP ANALYSIS
# ----------------------------------------------------------

print("Revenue by Year & Company:\n")
print(pd.pivot_table(df, values="Revenue", index="Year", columns="Company"))

print("\nProfit by Region:\n")
print(df.groupby("Region")["Profit"].sum())

print("\nAverage EBITDA Margin by Company:\n")
print(df.groupby("Company")["EBITDA_Margin_%"].mean())

# ----------------------------------------------------------
# 4️⃣ VISUALIZATION
# ----------------------------------------------------------

plt.figure()
for company in companies:
    company_data = df[df["Company"] == company]
    plt.plot(company_data["Year"], company_data["Revenue"])

plt.title("Revenue Trend (2016-2022)")
plt.xlabel("Year")
plt.ylabel("Revenue")
plt.show()

plt.figure()
df.groupby("Region")["Profit"].sum().plot(kind="bar")
plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Total Profit")
plt.show()

# ----------------------------------------------------------
# 5️⃣ 5-YEAR FORECAST
# ----------------------------------------------------------

future_years = list(range(2023, 2028))
forecast_data = []

for company in companies:
    
    company_data = df[df["Company"] == company]
    
    X = company_data[["Year"]]
    y = company_data["Revenue"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_df = pd.DataFrame(future_years, columns=["Year"])
    predictions = model.predict(future_df)
    
    for year, revenue in zip(future_years, predictions):
        forecast_data.append([year, company, round(revenue,2)])

forecast_df = pd.DataFrame(forecast_data, columns=[
    "Year",
    "Company",
    "Forecasted_Revenue"
])

forecast_df.to_csv("telecom_forecast_data.csv", index=False)

print("\n5-Year Forecast Generated.\n")
print(forecast_df.head())

print("\nProject Completed Successfully!")

df.to_csv("telecom_data.csv", index=False)


