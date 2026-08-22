# YUVA Internship - Task 2
# Data Cleaning and Pre-processing
# Dataset: Sample Superstore

import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv("Sample-Superstore.csv")
print("Original shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Check data quality
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 3. Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# 4. Convert date columns
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

# 5. Clean text columns
text_columns = ["Ship Mode", "Segment", "Country", "City", "State", "Region", "Category", "Sub-Category"]
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# 6. Convert numerical columns
numeric_columns = ["Sales", "Quantity", "Discount", "Profit"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 7. Fill missing numerical values with the median when needed
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Postal Code is kept missing if it has no value; no artificial location is created.

# 8. Check invalid shipping dates
invalid_dates = df[df["Ship Date"] < df["Order Date"]]
print("\nInvalid shipping dates:", len(invalid_dates))

# 9. Detect outliers using the IQR method
def find_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data[column] < lower) | (data[column] > upper)]

sales_outliers = find_outliers(df, "Sales")
profit_outliers = find_outliers(df, "Profit")
print("Sales outliers:", len(sales_outliers))
print("Profit outliers:", len(profit_outliers))

# 10. Feature engineering
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Quarter"] = df["Order Date"].dt.quarter
df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Profit Margin"] = np.where(df["Sales"] != 0, (df["Profit"] / df["Sales"]) * 100, 0)

# 11. Final validation
print("\nFinal shape:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum())
print("\nRemaining duplicate rows:", df.duplicated().sum())

# 12. Save cleaned dataset
df.to_csv("Superstore_Cleaned.csv", index=False)
print("\nCleaning completed successfully!")
print("Cleaned file saved as: Superstore_Cleaned.csv")
