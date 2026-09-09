---

# AI-Powered Customer Churn & RFM Segmentation

An end-to-end customer analytics solution combining Data Analytics, Machine Learning, Generative AI, and Business Intelligence to analyze customer behavior, segment users, predict churn risk, and generate automated AI retention recommendations.

---

## Executive Overview

Customer retention is a critical challenge for modern businesses operating in highly competitive markets. Identifying disengaging customers before they churn enables organizations to take proactive measures and mitigate revenue loss.

This project transforms 49,673 raw transaction records into actionable business intelligence. The workflow covers data engineering, exploratory data analysis, RFM scoring, K-Means customer segmentation, Random Forest predictive modeling, Google Gemini AI integration, and a multi-page Power BI dashboard.

---

## Analytical Workflow

```text
  ┌──────────────────────────┐
  │ Customer Transaction Data│ (49,673 Records, Demographics & Purchases)
  └─────────────┬────────────┘
                │
                ▼
  ┌──────────────────────────┐
  │ Data Quality & Cleaning  │ (Deduplication, Imputation, & Feature Formatting)
  └─────────────┬────────────┘
                │
                ▼
  ┌──────────────────────────┐
  │ EDA & RFM Analysis       │ (Recency, Frequency, & Monetary Scoring)
  └─────────────┬────────────┘
                │
                ▼
  ┌──────────────────────────┐
  │ K-Means Segmentation     │ (Behavioral Cluster Grouping)
  └─────────────┬────────────┘
                │
                ▼
  ┌──────────────────────────┐
  │ Random Forest Prediction │ (Predictive Churn Risk Probability)
  └─────────────┬────────────┘
                │
                ▼
  ┌──────────────────────────┐
  │ GenAI Retention Engine   │ (Google Gemini Personalized Strategy Prompting)
  └─────────────┬────────────┘
                │
                ▼
  ┌──────────────────────────┐
  │ Power BI Dashboard       │ (Interactive Executive Reporting)
  └──────────────────────────┘

```

---

## Core Analytics & AI Architecture

| Pipeline Stage | Operational Focus | Primary Output |
| --- | --- | --- |
| **RFM Analysis** | Evaluates Recency, Frequency, and Monetary values across customer profiles. | Quantitative scores identifying user engagement levels. |
| **K-Means Clustering** | Unsupervised ML grouping customers into distinct behavioral profiles. | Data-driven customer segments for targeted strategy. |
| **Random Forest ML** | Supervised classification model estimating churn probability. | Risk category assignments (High, Medium, Low). |
| **Google Gemini GenAI** | Natural language strategy engine processing customer metadata. | Automated, personalized retention playbook per high-risk user. |
| **Power BI Business Intelligence** | Multi-page executive dashboard visualizing revenue, risk, and segments. | Interactive decision-support portal for stakeholders. |

---

## Key Business Insights & Strategic Impact

Combining RFM metrics with unsupervised K-Means clustering revealed that visual spend indicators alone do not reflect long-term retention. Segmenting customers allows organizations to move away from generic marketing and apply precise retention strategies.

Integrating the Random Forest classifier shifts retention operations from reactive recovery to proactive intervention. Flagging high-risk accounts early allows marketing and success teams to allocate retention budgets effectively before customer churn occurs.

The integration of Google Gemini AI automates strategy generation. By feeding customer segments, spending history, and churn risk metrics into the generative model, the platform instantly outputs tailored intervention campaigns for front-line teams.

---

## Strategic Business Recommendations

| Target Customer Group | Primary Risk Factor | Recommended Strategic Action |
| --- | --- | --- |
| **High-Risk Segment** | Elevated Recency & Low Frequency | Immediate deployment of AI-generated personalized retention incentives. |
| **High-Value Champions** | High Monetary Value & High Frequency | Enrollment in exclusive loyalty tiers, VIP perks, and early feature access. |
| **Inactive / Lapsed Segment** | Long Recency Gap & Moderate Spend | Target with re-engagement workflows and category-specific product recommendations. |
| **General Audience** | Variable Spending & Broad Demographic | Apply segmented outreach campaigns instead of broad one-size-fits-all promotions. |

---

## Repository Structure

```text
AI-Customer-Churn-RFM-Segmentation/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── python/
│   ├── 01_data_quality_check.py
│   ├── 02_data_cleaning.py
│   ├── 03_eda.py
│   ├── 04_rfm_analysis.py
│   ├── 05_customer_segmentation.py
│   ├── 06_churn_prediction.py
│   ├── 07_customer_intelligence.py
│   └── 08_genai_retention.py
│
├── powerbi/
│   └── Customer_Churn_Dashboard.pbix
│
├── reports/
│   └── business_insights.md
│
├── presentation/
│   └── Customer_Churn_Presentation.pptx
│
├── images/
├── requirements.txt
└── README.md

```

---

## Tech Stack

| Domain | Tools & Libraries |
| --- | --- |
| **Language & Environment** | Python 3.x |
| **Data Engineering & EDA** | Pandas, NumPy |
| **Machine Learning & Stats** | Scikit-learn (K-Means, Random Forest) |
| **Generative AI** | Google Gemini API |
| **Business Intelligence** | Microsoft Power BI |
| **Version Control** | Git, GitHub |

---

## Getting Started

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/MoinFarooqui/AI-Customer-Churn-RFM-Segmentation.git

```

Navigate to the project directory:

```bash
cd AI-Customer-Churn-RFM-Segmentation

```

Install required Python dependencies:

```bash
pip install -r requirements.txt

```

### Environment Setup

The Generative AI retention module requires a valid Google Gemini API Key. Create a `.env` file in the root project directory and set your credentials:

```env
GEMINI_API_KEY=your_gemini_api_key_here

```

*(Note: The `.env` file is excluded from repository commits via `.gitignore` to protect sensitive API keys.)*

---

## Author & Connect

**Moin Farooqui**

*Data Analytics | Python Backend Development | Machine Learning | GenAI*
