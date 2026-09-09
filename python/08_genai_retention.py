import os
import time
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from google import genai

# ============================================================
# AI-POWERED CUSTOMER CHURN & RFM SEGMENTATION
# GENAI RETENTION RECOMMENDATION ENGINE
# ============================================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check your .env file."
    )

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# ============================================================
# FILE PATHS
# ============================================================

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_intelligence.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ai_retention_recommendations.csv"
)

# ============================================================
# LOAD CUSTOMER DATA
# ============================================================

print("=" * 60)
print("GENAI RETENTION RECOMMENDATION ENGINE")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print(f"\nTotal Customers: {len(df):,}")

# ============================================================
# IDENTIFY RECENCY COLUMN
# ============================================================

if "recency_rfm" in df.columns:
    RECENCY_COLUMN = "recency_rfm"

elif "recency" in df.columns:
    RECENCY_COLUMN = "recency"

elif "recency_churn" in df.columns:
    RECENCY_COLUMN = "recency_churn"

else:
    raise ValueError(
        "No recency column found in customer_intelligence.csv"
    )

print(f"Using Recency Column: {RECENCY_COLUMN}")

# ============================================================
# SELECT HIGH-RISK CUSTOMERS
# ============================================================

high_risk_customers = df[
    df["churn_risk"] == "High Risk"
].copy()

print(
    f"High-Risk Customers Found: "
    f"{len(high_risk_customers):,}"
)

# ============================================================
# LIMIT API REQUESTS
# ============================================================

SAMPLE_SIZE = 10

customers_to_analyze = high_risk_customers.head(
    SAMPLE_SIZE
)

print(
    f"\nGenerating AI recommendations for "
    f"{len(customers_to_analyze)} customers..."
)

# ============================================================
# GENERATE AI RECOMMENDATIONS
# ============================================================

recommendations = []

for _, customer in customers_to_analyze.iterrows():

    customer_recency = customer[RECENCY_COLUMN]

    prompt = f"""
You are a customer retention analyst for an e-commerce company.

Analyze the following customer profile and create a concise,
practical retention strategy.

CUSTOMER PROFILE

Customer ID: {customer['customer_id']}
Days Since Last Purchase: {customer_recency}
Purchase Frequency: {customer['frequency']}
Total Customer Spending: ₹{customer['monetary']:.2f}
Churn Probability: {customer['churn_probability']:.2%}
Churn Risk Level: {customer['churn_risk']}
Customer Segment: Cluster {customer['cluster']}

Provide your response using exactly these sections:

Risk Assessment:
Explain briefly why this customer is at risk.

Recommended Action:
Provide one specific action the business should take.

Retention Offer:
Suggest a realistic personalized offer.

Priority:
High, Medium, or Low.
"""

    try:

        # Gemini Interactions API
        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        recommendation = interaction.output_text

        recommendations.append({

            "customer_id": customer["customer_id"],

            "recency": customer_recency,

            "frequency": customer["frequency"],

            "monetary": customer["monetary"],

            "churn_probability":
                customer["churn_probability"],

            "churn_risk":
                customer["churn_risk"],

            "cluster":
                customer["cluster"],

            "ai_recommendation":
                recommendation
        })

        print(
            f"✓ Recommendation generated for "
            f"Customer {customer['customer_id']}"
        )

        # Prevent rapid API requests
        time.sleep(1)

    except Exception as e:

        print(
            f"✗ Error processing Customer "
            f"{customer['customer_id']}: {e}"
        )

# ============================================================
# SAVE RECOMMENDATIONS
# ============================================================

recommendations_df = pd.DataFrame(recommendations)

recommendations_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("GENAI RETENTION ANALYSIS COMPLETED")
print("=" * 60)

print(f"\nRecommendations saved to:\n{OUTPUT_PATH}")

print(
    f"\nTotal AI Recommendations Generated: "
    f"{len(recommendations_df)}"
)