import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# AI-POWERED CUSTOMER CHURN & RFM SEGMENTATION
# EXPLORATORY DATA ANALYSIS
# ============================================================

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_shopping_data_clean.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "images"
    / "eda"
)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Convert purchase date
df["purchase_date"] = pd.to_datetime(df["purchase_date"])

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset Rows: {df.shape[0]:,}")
print(f"Dataset Columns: {df.shape[1]}")

# ============================================================
# 1. CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(df["churn"].value_counts())

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = Retained, 1 = Churned)")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "churn_distribution.png",
    dpi=300
)

plt.close()

# ============================================================
# 2. REVENUE BY PRODUCT CATEGORY
# ============================================================

category_revenue = (
    df.groupby("product_category")["total_purchase_amount"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("REVENUE BY PRODUCT CATEGORY")
print("=" * 60)

print(category_revenue)

plt.figure(figsize=(10, 6))

category_revenue.plot(
    kind="bar"
)

plt.title("Total Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "revenue_by_category.png",
    dpi=300
)

plt.close()

# ============================================================
# 3. CHURN RATE BY PRODUCT CATEGORY
# ============================================================

category_churn = (
    df.groupby("product_category")["churn"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\n" + "=" * 60)
print("CHURN RATE BY PRODUCT CATEGORY")
print("=" * 60)

print(category_churn.round(2))

plt.figure(figsize=(10, 6))

category_churn.plot(
    kind="bar"
)

plt.title("Churn Rate by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "churn_by_category.png",
    dpi=300
)

plt.close()

# ============================================================
# 4. AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["customer_age"],
    bins=20,
    kde=True
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_distribution.png",
    dpi=300
)

plt.close()

# ============================================================
# 5. PURCHASE AMOUNT VS CHURN
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="churn",
    y="total_purchase_amount"
)

plt.title("Purchase Amount by Churn Status")
plt.xlabel("Churn (0 = Retained, 1 = Churned)")
plt.ylabel("Total Purchase Amount")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "purchase_amount_vs_churn.png",
    dpi=300
)

plt.close()

# ============================================================
# 6. CORRELATION MATRIX
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
)

correlation_matrix = numeric_columns.corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(correlation_matrix.round(2))

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# EDA COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nVisualizations saved to:")
print(OUTPUT_DIR)