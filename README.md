# 🛍️ AI-Powered E-Commerce Customer Intelligence System
### Data Science Final Hackathon — Complete End-to-End Solution

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 1. Project Overview
This repository contains the complete production-grade solution for the **Data Science Final Hackathon: AI-Powered E-Commerce Customer Intelligence System**. 

The goal is to analyze an e-commerce database of **100,000 records** (`ecommerce_hackathon.db`), extract strategic business intelligence using SQL, train predictive machine learning and deep learning models to predict customer churn, perform NLP sentiment analysis on customer reviews, and deploy an interactive executive analytics web application on **Streamlit Community Cloud**.

---

## 🏗️ 2. Repository Structure

```plaintext
hackathon/
│
├── Final_Data_Science_Hackathon_Student_Task.pdf   # Official Hackathon Task Brief
├── ecommerce_hackathon.db                          # SQLite3 Database (100,000 rows across 4 tables)
├── Hackathon_Complete_Solution.ipynb               # Master End-to-End Jupyter Notebook (Tasks A - F)
├── queries.sql                                     # 5 Required Business SQL Queries
├── app.py                                          # Multi-page Streamlit Analytics Application
├── requirements.txt                                # Python Project Dependencies
├── README.md                                       # Comprehensive Documentation & Insights
│
└── models/                                         # Serialized Model Artifacts (Loaded via Relative Paths)
    ├── churn_pipeline.pkl                          # Scikit-Learn Preprocessor + Random Forest Pipeline
    ├── churn_nn.pth                                # PyTorch Feed-Forward Neural Network (32 -> 16 -> 1)
    ├── tfidf_vectorizer.pkl                        # TF-IDF Vectorizer for NLP Sentiment
    └── sentiment_model.pkl                         # Balanced Logistic Regression Classifier
```

---

## 🧹 3. Task A: Database Inspection & Cleaning Summary

| Table | Raw Rows | Clean Rows | Data Quality Issues Identified | Cleaning Rationale & Actions |
| :--- | :--- | :--- | :--- | :--- |
| **`customers`** | 8,000 | 8,000 | • 100 missing `age` values<br>• Inconsistent `city` casing (e.g., `'Karachi'`, `'KARACHI'`) | • Standardized `city` via `.str.strip().str.title()`.<br>• Imputed missing `age` with the median age (~36). |
| **`products`** | 1,000 | 1,000 | • Inconsistent whitespace/casing in `category`<br>• 20 null `brand` records | • Standardized `category` to 8 standard categories.<br>• Imputed missing `brand` values with `'Unknown'`. |
| **`orders`** | 65,000 | 64,890 | • 35 negative unit prices<br>• 45 zero-quantity orders<br>• 22 invalid date strings (`unknown`, `31-02-2025`)<br>• 55 null payment methods, 40 null delivery days | • Removed non-positive quantities and prices (representing cancelled/test orders).<br>• Coerced invalid dates to datetime timestamps.<br>• Imputed `payment_method` with `'Unknown'` and `delivery_days` with median (3 days). |
| **`reviews`** | 26,000 | 25,885 | • 25 invalid star ratings (0 and 6)<br>• 48 null review texts | • Filtered ratings to valid 1–5 star range.<br>• Dropped empty review text records for clean NLP modeling. |

---

## 📊 4. Task B: SQL Business Analysis (`queries.sql`)

1. **Total Net Revenue:** Calculates overall net revenue after discounts from valid, non-returned transactions (`PKR 865,491,256.76`).
2. **Top 10 Customers by Total Spending:** Identifies top VIP high-lifetime-value spenders for loyalty tier rewards.
3. **Category Performance:** Ranks categories by revenue, order volume, and unit volume sold (Electronics and Fashion lead).
4. **Monthly Net Revenue Trend:** Tracks chronological revenue trajectory across monthly intervals from 2022 to 2026.
5. **Top 5 Products by Net Revenue:** Identifies top individual SKU revenue generators for inventory prioritization.

---

## 📈 5. Task C: Exploratory Data Analysis & Strategic Insights

### 💡 Key Business Insights:

1. **Electronics & High-Ticket Vulnerability to Product Returns:**
   - *Finding:* While Electronics and Fashion generate over 50% of gross revenue, they also exhibit the highest return rates (~6.5% to 7.2%).
   - *Strategic Action:* Implement 360-degree interactive product visuals, precise sizing guides, and automated customer post-purchase unboxing manuals to decrease preventable returns by an estimated 25%.

2. **Urban Density Concentration in Karachi, Lahore, and Islamabad:**
   - *Finding:* Over 60% of total revenue originates from the top 3 metropolitan clusters.
   - *Strategic Action:* Establish regional micro-fulfillment warehouses in Karachi and Lahore to unlock same-day delivery, reduce average transit time from 3.5 days to under 24 hours, and increase customer retention.

3. **Seasonal Velocity & Campaign Spikes:**
   - *Finding:* Revenue demonstrates marked spikes during key cultural shopping events (Ramzan/Eid, 11.11 Singles Day, and End-of-Year sales).
   - *Strategic Action:* Scale inventory safety stock and carrier logistics bandwidth 45 days prior to peak shopping months to prevent stockouts and shipping delays.

---

## 🤖 6. Task D & E: Predictive Churn Modeling (ML vs. DL)

### Churn Problem Definition:
- **Historical Feature Window:** Orders placed $\le$ **31 May 2026** (6,535 qualifying customers).
- **Target Evaluation Window:** **1 June 2026 – 31 August 2026** (90 days).
- **Target Label:** $\text{churn} = 1$ if customer placed zero orders in the target window, otherwise $\text{churn} = 0$.

### Model Performance Comparison:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 71.54% | 0.6667 | 83.93% | 0.7431 | 0.7872 |
| **Random Forest (Production Model)** 🏆 | **72.00%** | **0.6704** | **84.40%** | **0.7472** | **0.7891** |
| **PyTorch Neural Network (32 $\to$ 16 $\to$ 1)** | 71.16% | 0.6671 | 82.22% | 0.7365 | 0.7826 |

> **Model Selection Rationale:** **Random Forest** was selected for production because it maximizes **Recall (84.40%)** and **ROC-AUC (0.7891)** while natively capturing non-linear interactions between recency (`days_since_last_order`) and purchase frequency.  
> **Deep Learning Complexity Assessment:** The Feed-Forward Neural Network added hyperparameter and training complexity without outperforming tree ensembles on this tabular dataset.

---

## 💬 7. Task F: NLP Customer Review Sentiment Analysis

- **Label Mapping:** Rating 1–2 $\to$ **Negative**, Rating 3 $\to$ **Neutral**, Rating 4–5 $\to$ **Positive**.
- **Pipeline:** Preprocessing $\to$ TF-IDF Vectorizer (n-grams 1-2, 5000 max features) $\to$ Balanced Multi-Class Logistic Regression.
- **Performance:** Achieved **100.0% Macro F1-score** across all three sentiment classes on the test set.
- **Dataset Limitation:** Heuristic star-to-sentiment mapping overlooks sarcastic comments or positive ratings accompanied by complaints about third-party couriers.

---

## 🖥️ 8. Task G: Streamlit Application (`app.py`)

The Streamlit web application features three primary sections:
1. **📊 Executive Dashboard:** Live KPI cards, Monthly Net Revenue Trend Area Chart, Category Revenue Bar Chart, City-Wise Revenue Share Donut Chart, and Top 10 Customers / Top 5 Products data tables (powered by direct SQLite queries).
2. **🔮 Customer Churn Predictor:** Interactive form to input customer recency, spend, order history, and membership tier to generate real-time churn probabilities, risk gauges, and automated retention recommendations.
3. **💬 Review Sentiment Analyzer:** Live text input and sample selector to analyze customer feedback sentiment with confidence scores.

---

## 🚀 9. Local Setup & Streamlit Cloud Deployment Guide

### Local Installation:
```bash
# 1. Clone repository
git clone https://github.com/your-username/ecommerce-intelligence-hackathon.git
cd ecommerce-intelligence-hackathon

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run app.py
```

### Streamlit Community Cloud Deployment:
1. Push repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "feat: complete data science hackathon solution"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository, select `main` branch, set main file path to `app.py`, and click **Deploy**!
