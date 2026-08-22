# Task 3 - Data Visualization
# YUVA Data Analytics Internship
# Dataset: Sample Superstore

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# 1. Load the cleaned dataset
# ---------------------------------------------------

df = pd.read_csv("cleaned_superstore.csv")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ---------------------------------------------------
# 2. Convert Order Date to datetime
# ---------------------------------------------------

df["Order Date"] = pd.to_datetime(df["Order Date"])

# ===================================================
# 3. Bar Chart - Sales by Category
# ===================================================

category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ===================================================
# 4. Bar Chart - Profit by Category
# ===================================================

category_profit = df.groupby("Category")["Profit"].sum()

plt.figure(figsize=(8, 5))
category_profit.plot(kind="bar")

plt.title("Total Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ===================================================
# 5. Line Chart - Monthly Sales Trend
# ===================================================

df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===================================================
# 6. Scatter Plot - Discount vs Profit
# ===================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Discount",
    y="Profit",
    alpha=0.5
)

plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()

# ===================================================
# 7. Heatmap - Correlation Analysis
# ===================================================

corr = df[["Sales", "Quantity", "Discount", "Profit"]].corr()

plt.figure(figsize=(7, 5))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ===================================================
# 8. Bar Chart - Profit by Region
# ===================================================

region_profit = (
    df.groupby("Region")["Profit"]
    .sum()
    .sort_values()
)

plt.figure(figsize=(8, 5))

region_profit.plot(kind="barh")

plt.title("Total Profit by Region")
plt.xlabel("Profit")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

# ===================================================
# 9. Display Basic Findings
# ===================================================

print("\n----- Visualization Findings -----")

print("\nCategory with highest sales:")
print(
    category_sales.idxmax(),
    "->",
    round(category_sales.max(), 2)
)

print("\nCategory with highest profit:")
print(
    category_profit.idxmax(),
    "->",
    round(category_profit.max(), 2)
)

print("\nRegion with highest profit:")
print(
    region_profit.idxmax(),
    "->",
    round(region_profit.max(), 2)
)

print("\nRegion with lowest profit:")
print(
    region_profit.idxmin(),
    "->",
    round(region_profit.min(), 2)
)

print("\nData visualization completed successfully!")