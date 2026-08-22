# YUVA Internship - Superstore EDA
# Save this file in the same folder as "Sample - Superstore.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Sample - Superstore.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types and non-null counts:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# Numerical summary
print("\nSummary statistics:")
print(df[["Sales", "Quantity", "Discount", "Profit"]].describe())

# Category analysis
print("\nSales by Category:")
print(df.groupby("Category")["Sales"].sum().sort_values(ascending=False))

print("\nProfit by Category:")
print(df.groupby("Category")["Profit"].sum().sort_values(ascending=False))

# Region analysis
print("\nRegion performance:")
print(df.groupby("Region")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False))

# Time features
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

monthly = df.groupby("Month")[["Sales", "Profit"]].sum()

# Visualizations
sns.histplot(df["Sales"], bins=40, kde=True)
plt.title("Sales Distribution")
plt.tight_layout()
plt.show()

sns.histplot(df["Profit"], bins=40, kde=True)
plt.title("Profit Distribution")
plt.tight_layout()
plt.show()

sns.boxplot(data=df[["Sales", "Profit"]])
plt.title("Sales and Profit Box Plot")
plt.tight_layout()
plt.show()

sns.barplot(data=df, x="Category", y="Sales", estimator=sum, errorbar=None)
plt.title("Total Sales by Category")
plt.tight_layout()
plt.show()

sns.barplot(data=df, x="Category", y="Profit", estimator=sum, errorbar=None)
plt.title("Total Profit by Category")
plt.tight_layout()
plt.show()

monthly["Sales"].plot(figsize=(10, 5))
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

sns.scatterplot(data=df, x="Discount", y="Profit", alpha=0.5)
plt.title("Discount vs Profit")
plt.tight_layout()
plt.show()

print("\nCorrelation matrix:")
print(df[["Sales", "Quantity", "Discount", "Profit"]].corr())
