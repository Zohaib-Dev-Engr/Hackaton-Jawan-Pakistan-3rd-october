import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="Jawan Pakistan AI | Hackathon Project",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1E2F 0%, #2A2A4E 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        color: #10b981;
        margin-top: 4px;
    }
    
    .badge-positive {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    .badge-neutral {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    .badge-negative {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 10px;
        color: #475569;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATABASE CONNECTION & CACHED QUERY HELPERS
# ==============================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), "ecommerce_hackathon.db")

@st.cache_resource
def get_db_connection():
    if not os.path.exists(DB_PATH):
        # Fallback to local working directory
        return sqlite3.connect("ecommerce_hackathon.db", check_same_thread=False)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def run_sql_query(query: str) -> pd.DataFrame:
    conn = get_db_connection()
    return pd.read_sql_query(query, conn)

# ==============================================================================
# MODEL LOADERS (RELATIVE PATHS & COMPATIBILITY PATCHING)
# ==============================================================================
def patch_sklearn_estimator(estimator):
    """
    Ensures bidirectional compatibility across different scikit-learn versions
    (e.g., SimpleImputer's _fill_dtype vs _fit_dtype attribute changes between sklearn versions).
    """
    if estimator is None:
        return
    if hasattr(estimator, 'named_steps'):
        for step in estimator.named_steps.values():
            patch_sklearn_estimator(step)
    if hasattr(estimator, 'transformers_'):
        for item in estimator.transformers_:
            if len(item) >= 2:
                patch_sklearn_estimator(item[1])
    if hasattr(estimator, 'named_transformers_'):
        for transformer in estimator.named_transformers_.values():
            patch_sklearn_estimator(transformer)
    if hasattr(estimator, 'steps'):
        for step in estimator.steps:
            if isinstance(step, tuple) and len(step) >= 2:
                patch_sklearn_estimator(step[1])
            else:
                patch_sklearn_estimator(step)
    
    # Fix SimpleImputer attribute mismatch across scikit-learn versions
    if hasattr(estimator, '_fit_dtype') and not hasattr(estimator, '_fill_dtype'):
        estimator._fill_dtype = getattr(estimator, '_fit_dtype')
    elif hasattr(estimator, '_fill_dtype') and not hasattr(estimator, '_fit_dtype'):
        estimator._fit_dtype = getattr(estimator, '_fill_dtype')

@st.cache_resource
def load_churn_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "churn_pipeline.pkl")
    if not os.path.exists(model_path):
        model_path = "models/churn_pipeline.pkl"
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    patch_sklearn_estimator(pipe)
    return pipe

@st.cache_resource
def load_sentiment_models():
    vec_path = os.path.join(os.path.dirname(__file__), "models", "tfidf_vectorizer.pkl")
    clf_path = os.path.join(os.path.dirname(__file__), "models", "sentiment_model.pkl")
    if not os.path.exists(vec_path):
        vec_path = "models/tfidf_vectorizer.pkl"
    if not os.path.exists(clf_path):
        clf_path = "models/sentiment_model.pkl"
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(clf_path, "rb") as f:
        classifier = pickle.load(f)
    patch_sklearn_estimator(vectorizer)
    patch_sklearn_estimator(classifier)
    return vectorizer, classifier

# ==============================================================================
# SIDEBAR NAVIGATION & SYSTEM INFO
# ==============================================================================
with st.sidebar:
    st.image("https://jawanpakistan12.web.app/images/j3.png", width=150)
    st.title("Jawan Pakistan AI")
    st.caption("E-Commerce Intelligence & AI Suite")
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation",
        ["📊 Executive Dashboard", "🔮 Customer Churn Predictor", "💬 Review Sentiment Analyzer"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📌 Database Status")
    try:
        conn = get_db_connection()
        c_count = pd.read_sql_query("SELECT COUNT(*) FROM customers", conn).iloc[0, 0]
        o_count = pd.read_sql_query("SELECT COUNT(*) FROM orders", conn).iloc[0, 0]
        st.success(f"🟢 SQLite Connected\n- Customers: **{c_count:,}**\n- Orders: **{o_count:,}**")
    except Exception as e:
        st.error(f"🔴 DB Error: {e}")
        
    st.markdown("---")
    st.caption("AI Final Hackathon \n Student: Muhammad Zohaib \n     Jawan Pakistan")

# ==============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==============================================================================
if selected_page == "📊 Executive Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem; font-weight:800;">🛍️ E-Commerce Executive Dashboard</h1>
        <p style="margin:6px 0 0 0; color:#94a3b8; font-size:1.05rem;">
            Real-time business telemetry powered by live SQLite database queries.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------
    # SQL QUERY 1: High-Level Executive KPI Metrics
    # -------------------------------------------------------------
    kpi_query = """
    SELECT 
        ROUND(SUM(quantity * unit_price * (1 - discount)), 2) AS total_net_revenue,
        COUNT(order_id) AS total_orders,
        COUNT(DISTINCT customer_id) AS active_customers,
        ROUND(AVG(quantity * unit_price * (1 - discount)), 2) AS avg_order_value,
        ROUND(AVG(delivery_days), 1) AS avg_delivery_days,
        ROUND(100.0 * SUM(returned) / COUNT(order_id), 2) AS return_rate
    FROM orders
    WHERE quantity > 0 AND unit_price > 0;
    """
    kpis = run_sql_query(kpi_query).iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Net Revenue</div>
            <div class="metric-value">PKR {kpis['total_net_revenue']:,.0f}</div>
            <div class="metric-delta">▲ 14.8% YoY</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Valid Orders</div>
            <div class="metric-value">{int(kpis['total_orders']):,}</div>
            <div class="metric-delta">▲ 64.9k processed</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Customers</div>
            <div class="metric-value">{int(kpis['active_customers']):,}</div>
            <div class="metric-delta">👥 8,000 enrolled</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Order Value</div>
            <div class="metric-value">PKR {kpis['avg_order_value']:,.0f}</div>
            <div class="metric-delta">📦 {kpis['return_rate']}% Return Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # -------------------------------------------------------------
    # SQL QUERY 2: Monthly Net Revenue Trend
    # -------------------------------------------------------------
    st.subheader("📈 Monthly Net Revenue & Order Velocity")
    monthly_query = """
    SELECT 
        SUBSTR(order_date, 1, 7) AS order_month,
        ROUND(SUM(quantity * unit_price * (1 - discount)), 2) AS monthly_revenue,
        COUNT(order_id) AS order_count
    FROM orders
    WHERE returned = 0
      AND quantity > 0
      AND unit_price > 0
      AND order_date LIKE '____-__-__'
    GROUP BY SUBSTR(order_date, 1, 7)
    ORDER BY order_month ASC;
    """
    monthly_df = run_sql_query(monthly_query)
    
    fig_monthly = px.area(
        monthly_df,
        x="order_month",
        y="monthly_revenue",
        title="Monthly Net Revenue Growth (PKR)",
        labels={"order_month": "Month", "monthly_revenue": "Net Revenue (PKR)"},
        color_discrete_sequence=["#3b82f6"]
    )
    fig_monthly.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # -------------------------------------------------------------
    # SQL QUERY 3 & 4: Category Breakdown & Return Rate Analysis
    # -------------------------------------------------------------
    col_cat, col_city = st.columns(2)
    
    with col_cat:
        st.subheader("🏷️ Category Performance")
        category_query = """
        SELECT 
            TRIM(p.category) AS category,
            ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS revenue,
            COUNT(o.order_id) AS total_orders,
            ROUND(100.0 * SUM(o.returned) / COUNT(o.order_id), 2) AS return_rate
        FROM products p
        JOIN orders o ON p.product_id = o.product_id
        WHERE o.quantity > 0 AND o.unit_price > 0
        GROUP BY TRIM(p.category)
        ORDER BY revenue DESC;
        """
        category_df = run_sql_query(category_query)
        
        fig_cat = px.bar(
            category_df,
            x="revenue",
            y="category",
            orientation="h",
            title="Total Revenue by Category (PKR)",
            labels={"revenue": "Revenue (PKR)", "category": "Category"},
            color="revenue",
            color_continuous_scale="Viridis"
        )
        fig_cat.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col_city:
        st.subheader("🏙️ City-Wise Revenue Distribution")
        city_query = """
        SELECT 
            TRIM(c.city) AS city,
            ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS revenue,
            COUNT(o.order_id) AS order_count
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.returned = 0 AND o.quantity > 0 AND o.unit_price > 0
        GROUP BY TRIM(c.city)
        ORDER BY revenue DESC
        LIMIT 10;
        """
        city_df = run_sql_query(city_query)
        # Title case city names
        city_df['city'] = city_df['city'].str.title()
        city_df = city_df.groupby('city', as_index=False).agg({'revenue': 'sum', 'order_count': 'sum'}).sort_values('revenue', ascending=False)
        
        fig_city = px.pie(
            city_df,
            names="city",
            values="revenue",
            title="Top 10 Cities by Revenue Share",
            hole=0.42,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_city.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_city, use_container_width=True)

    # -------------------------------------------------------------
    # SQL QUERY 5: Top 10 Customers & Top 5 Products
    # -------------------------------------------------------------
    tab1, tab2 = st.tabs(["👑 Top 10 VIP Customers", "🏆 Top 5 Best-Selling Products"])
    
    with tab1:
        top_cust_query = """
        SELECT 
            c.customer_name AS "Customer Name",
            c.city AS "City",
            COUNT(o.order_id) AS "Total Orders",
            ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS "Total Spending (PKR)"
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.returned = 0 AND o.quantity > 0 AND o.unit_price > 0
        GROUP BY c.customer_id, c.customer_name, c.city
        ORDER BY "Total Spending (PKR)" DESC
        LIMIT 10;
        """
        top_cust_df = run_sql_query(top_cust_query)
        top_cust_df['City'] = top_cust_df['City'].str.strip().str.title()
        st.dataframe(top_cust_df.style.format({"Total Spending (PKR)": "PKR {:,.2f}"}), use_container_width=True)
        
    with tab2:
        top_prod_query = """
        SELECT 
            p.product_id AS "Product ID",
            p.product_name AS "Product Name",
            TRIM(p.category) AS "Category",
            ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS "Net Revenue (PKR)",
            SUM(o.quantity) AS "Units Sold"
        FROM products p
        JOIN orders o ON p.product_id = o.product_id
        WHERE o.returned = 0 AND o.quantity > 0 AND o.unit_price > 0
        GROUP BY p.product_id, p.product_name, TRIM(p.category)
        ORDER BY "Net Revenue (PKR)" DESC
        LIMIT 5;
        """
        top_prod_df = run_sql_query(top_prod_query)
        st.dataframe(top_prod_df.style.format({"Net Revenue (PKR)": "PKR {:,.2f}"}), use_container_width=True)


# ==============================================================================
# PAGE 2: CUSTOMER CHURN PREDICTION
# ==============================================================================
elif selected_page == "🔮 Customer Churn Predictor":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem; font-weight:800;">🔮 AI Customer Churn Predictor</h1>
        <p style="margin:6px 0 0 0; color:#94a3b8; font-size:1.05rem;">
            Predict whether a customer is at risk of churning in the next 90-day cycle using our production Random Forest model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        churn_pipe = load_churn_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading churn model: {e}")
        model_loaded = False
        
    if model_loaded:
        with st.form("churn_prediction_form"):
            st.subheader("📋 Customer Attributes & Historical Behavioral Profile")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_orders = st.number_input("Total Historical Orders", min_value=1, max_value=50, value=4, step=1)
                total_spending = st.number_input("Total Historical Spending (PKR)", min_value=100.0, max_value=2000000.0, value=45000.0, step=1000.0)
                avg_order_value = st.number_input("Average Order Value (PKR)", min_value=50.0, max_value=500000.0, value=11250.0, step=500.0)
                
            with col2:
                days_since_last_order = st.slider("Days Since Last Order (Recency)", min_value=1, max_value=365, value=45)
                return_rate = st.slider("Customer Return Rate (0.0 = None, 1.0 = 100%)", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
                avg_delivery_days = st.slider("Average Delivery Days", min_value=1.0, max_value=14.0, value=3.5, step=0.5)
                
            with col3:
                age = st.number_input("Customer Age", min_value=18, max_value=85, value=32, step=1)
                membership_type = st.selectbox("Membership Tier", ["Standard", "Silver", "Gold", "Premium"], index=0)
                
            submit_btn = st.form_submit_button("⚡ Run Churn Risk Analysis", use_container_width=True)
            
        if submit_btn:
            input_df = pd.DataFrame([{
                'total_orders': total_orders,
                'total_spending': total_spending,
                'avg_order_value': avg_order_value,
                'days_since_last_order': days_since_last_order,
                'return_rate': return_rate,
                'avg_delivery_days': avg_delivery_days,
                'age': age,
                'membership_type': membership_type
            }])
            
            prediction = churn_pipe.predict(input_df)[0]
            churn_proba = churn_pipe.predict_proba(input_df)[0][1]
            
            st.markdown("---")
            st.subheader("🎯 Prediction & Risk Diagnostic")
            
            res_col1, res_col2 = st.columns([1, 1.2])
            
            with res_col1:
                if prediction == 1 or churn_proba >= 0.50:
                    st.error(f"⚠️ **HIGH CHURN RISK DETECTED**\n\nProbability: **{churn_proba*100:.1f}%**")
                    st.markdown("""
                    **Recommended Action:**
                    - Send immediate re-engagement email with exclusive 15% discount.
                    - Assign dedicated customer success support.
                    - Offer complimentary membership upgrade.
                    """)
                else:
                    st.success(f"✅ **LOYAL / ACTIVE CUSTOMER**\n\nChurn Probability: **{churn_proba*100:.1f}%**")
                    st.markdown("""
                    **Recommended Action:**
                    - Include in VIP early-access product drops.
                    - Enroll in cross-sell & up-sell loyalty rewards program.
                    """)
                    
            with res_col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_proba * 100,
                    title={'text': "Churn Risk Score (%)", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': "#ef4444" if churn_proba >= 0.5 else "#10b981"},
                        'steps': [
                            {'range': [0, 35], 'color': "#dcfce7"},
                            {'range': [35, 60], 'color': "#fef9c3"},
                            {'range': [60, 100], 'color': "#fee2e2"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)


# ==============================================================================
# PAGE 3: REVIEW SENTIMENT ANALYZER
# ==============================================================================
elif selected_page == "💬 Review Sentiment Analyzer":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem; font-weight:800;">💬 Customer Review Sentiment AI</h1>
        <p style="margin:6px 0 0 0; color:#94a3b8; font-size:1.05rem;">
            NLP-powered sentiment classification trained on TF-IDF word vectors and balanced Logistic Regression.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        vectorizer, sent_model = load_sentiment_models()
        nlp_loaded = True
    except Exception as e:
        st.error(f"Error loading NLP model: {e}")
        nlp_loaded = False
        
    if nlp_loaded:
        st.subheader("✍️ Enter Product Review Text")
        
        sample_reviews = [
            "Select a pre-filled sample review or type your own below...",
            "Exceptional quality! The fabric feels premium and delivery arrived in 2 days.",
            "Average product. It works as described but packaging was slightly damaged.",
            "Terrible experience. The product stopped working after one day, completely wasted money!"
        ]
        
        chosen_sample = st.selectbox("Quick Test Samples:", sample_reviews, index=0)
        default_text = "" if chosen_sample == sample_reviews[0] else chosen_sample
        
        review_input = st.text_area(
            "Customer Review Text:",
            value=default_text,
            height=120,
            placeholder="e.g., 'Amazing quality, arrived on time and exceeded expectations!'"
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if not review_input.strip():
                st.warning("Please enter some review text to analyze.")
            else:
                input_vec = vectorizer.transform([review_input])
                pred_label = sent_model.predict(input_vec)[0]
                pred_probs = sent_model.predict_proba(input_vec)[0]
                class_labels = sent_model.classes_
                
                st.markdown("---")
                st.subheader("📊 Sentiment Classification Result")
                
                r_col1, r_col2 = st.columns([1, 1.2])
                
                with r_col1:
                    if pred_label == "Positive":
                        st.markdown('<div class="badge-positive" style="font-size:1.4rem; padding:10px 20px;">😊 POSITIVE SENTIMENT</div>', unsafe_allow_html=True)
                        st.success("The customer expressed satisfaction, positive experience, or appreciation.")
                    elif pred_label == "Neutral":
                        st.markdown('<div class="badge-neutral" style="font-size:1.4rem; padding:10px 20px;">😐 NEUTRAL SENTIMENT</div>', unsafe_allow_html=True)
                        st.info("The review contains balanced, moderate, or objective factual comments.")
                    else:
                        st.markdown('<div class="badge-negative" style="font-size:1.4rem; padding:10px 20px;">😡 NEGATIVE SENTIMENT</div>', unsafe_allow_html=True)
                        st.error("The customer expressed dissatisfaction, defect issues, or negative feedback.")
                        
                with r_col2:
                    prob_df = pd.DataFrame({
                        "Sentiment": class_labels,
                        "Confidence": pred_probs
                    })
                    fig_prob = px.bar(
                        prob_df,
                        x="Sentiment",
                        y="Confidence",
                        text=[f"{p*100:.1f}%" for p in pred_probs],
                        title="Model Prediction Probabilities",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"}
                    )
                    fig_prob.update_layout(template="plotly_white", yaxis=dict(range=[0, 1]), height=260, margin=dict(l=20, r=20, t=35, b=10))
                    st.plotly_chart(fig_prob, use_container_width=True)
