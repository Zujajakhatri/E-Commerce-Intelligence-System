import sqlite3
from pathlib import Path
import re
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="E-Commerce Intelligence", page_icon="🛒", layout="wide")

# All paths are relative to this app.py file, so the app works from any folder
BASE = Path(__file__).parent
DB_PATH = BASE / "ecommerce_hackathon.db"
CSS_PATH = BASE / "style.css"
HTML_PATH = BASE / "index.html"
BLUE = "#7C6FE6"
PALETTE = ["#7C6FE6", "#F9A8B8", "#A99CF5", "#FCD34D", "#86EFAC", "#C4B5FD"]

# =============== Load CSS & HTML ===============
# Backup templates, used only if index.html is missing
DEFAULT_TEMPLATES = {
    "BRAND": '<div class="brand"><h2>Ecom IQ</h2><p>E-Commerce Intelligence</p></div>',
    "NAV_TITLE": '<div class="nav-title">MENU</div>',
    "HERO": '<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
    "SECTION": '<div class="section">{text}</div>',
    "RESULT": '<div class="result {color}">{text}<small>{sub}</small></div>',
    "TIP": '<div class="tip"><b>{title}</b><br>{text}</div>',
    "FOOTER": '<div class="footer">Built with Streamlit · SQLite · Scikit-learn | Data Science Hackathon 2026</div>',
}

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS file not found: {CSS_PATH}")

TEMPLATES = dict(DEFAULT_TEMPLATES)
if HTML_PATH.exists():
    for part in HTML_PATH.read_text(encoding="utf-8").split("<!-- "):
        if " -->" in part:
            name, body = part.split(" -->", 1)
            TEMPLATES[name.strip()] = body.strip()


def html(name, where=st, **values):
    template = TEMPLATES.get(name, DEFAULT_TEMPLATES[name])
    where.markdown(template.format(**values), unsafe_allow_html=True)


def hero(title, subtitle):
    html("HERO", title=title, subtitle=subtitle)


def section(text):
    html("SECTION", text=text)


def result_card(text, sub, color):
    html("RESULT", text=text, sub=sub, color=color)


def tip(title, text):
    html("TIP", title=title, text=text)


# =============== Data & models ===============
@st.cache_data
def run_query(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


@st.cache_resource
def load_models():
    churn = joblib.load(BASE / "models" / "churn_model.pkl")
    sentiment = joblib.load(BASE / "models" / "sentiment_model.pkl")
    return churn, sentiment


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def style(fig):
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEEEEE")
    return fig


churn_model, sentiment_model = load_models()

html("BRAND", where=st.sidebar)
html("NAV_TITLE", where=st.sidebar)
page = st.sidebar.radio("Menu", ["📊 Dashboard", "🔮 Churn Prediction", "💬 Sentiment Analysis"],
                        label_visibility="collapsed")

VALID = "order_date GLOB '[0-9][0-9][0-9][0-9]-[0-1][0-9]-*' AND substr(order_date, 6, 2) BETWEEN '01' AND '12'"
REV = "SUM(o.quantity * o.unit_price * (1 - o.discount))"

# =============== DASHBOARD ===============
if page == "📊 Dashboard":
    years = run_query(f"SELECT DISTINCT substr(order_date, 1, 4) AS y FROM orders_clean WHERE {VALID} ORDER BY y")["y"].tolist()
    year = st.sidebar.selectbox("Filter by year", ["All"] + years)
    yf = "" if year == "All" else f"AND substr(o.order_date, 1, 4) = '{year}'"

    hero("📊 Business Dashboard", f"Live insights from the SQLite database · Showing: {year}")

    k = run_query(f"""
        SELECT COUNT(*) AS orders, COUNT(DISTINCT o.customer_id) AS customers,
               AVG(o.returned) * 100 AS return_rate
        FROM orders_clean o WHERE {VALID} {yf}
    """).iloc[0]
    net = run_query(f"SELECT {REV} AS r FROM orders_clean o WHERE o.returned = 0 AND {VALID} {yf}")["r"][0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Net Revenue", f"PKR {net / 1e6:,.0f}M")
    c2.metric("📦 Total Orders", f"{int(k['orders']):,}")
    c3.metric("👥 Customers", f"{int(k['customers']):,}")
    c4.metric("🧾 Avg Order", f"PKR {net / k['orders']:,.0f}")
    c5.metric("↩️ Return Rate", f"{k['return_rate']:.1f}%")

    monthly = run_query(f"""
        SELECT substr(o.order_date, 1, 7) AS month, {REV} / 1000000 AS revenue
        FROM orders_clean o WHERE o.returned = 0 AND {VALID} {yf}
        GROUP BY month ORDER BY month
    """)
    section("📈 Monthly Net Revenue (Million PKR)")
    st.plotly_chart(style(px.area(monthly, x="month", y="revenue", color_discrete_sequence=[BLUE])))

    left, right = st.columns(2)
    with left:
        section("🏷️ Revenue by Category (M PKR)")
        cat = run_query(f"""
            SELECT p.category, {REV} / 1000000 AS revenue
            FROM orders_clean o JOIN products_clean p ON o.product_id = p.product_id
            WHERE o.returned = 0 AND {VALID} {yf}
            GROUP BY p.category ORDER BY revenue
        """)
        st.plotly_chart(style(px.bar(cat, x="revenue", y="category", orientation="h",
                                     text_auto=".0f", color_discrete_sequence=[BLUE])))
    with right:
        section("🏙️ Revenue by City (M PKR)")
        city = run_query(f"""
            SELECT c.city, {REV} / 1000000 AS revenue
            FROM orders_clean o JOIN customers_clean c ON o.customer_id = c.customer_id
            WHERE o.returned = 0 AND {VALID} {yf}
            GROUP BY c.city ORDER BY revenue
        """)
        st.plotly_chart(style(px.bar(city, x="revenue", y="city", orientation="h",
                                     text_auto=".0f", color_discrete_sequence=[BLUE])))

    left, right = st.columns(2, gap="large")
    with left:
        section("↩️ Return Rate by Category (%)")
        ret = run_query(f"""
            SELECT p.category, AVG(o.returned) * 100 AS return_rate
            FROM orders_clean o JOIN products_clean p ON o.product_id = p.product_id
            WHERE {VALID} {yf}
            GROUP BY p.category ORDER BY return_rate
        """)
        fig = px.bar(ret, x="return_rate", y="category", orientation="h", text_auto=".1f",
                     color="return_rate", color_continuous_scale=["#E9E5FF", "#7C6FE6"])
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style(fig))
    with right:
        section("💳 Orders by Payment Method")
        pay = run_query(f"""
            SELECT o.payment_method, COUNT(*) AS orders
            FROM orders_clean o WHERE {VALID} {yf}
            GROUP BY o.payment_method
        """)
        st.plotly_chart(style(px.pie(pay, names="payment_method", values="orders", hole=0.55,
                                     color_discrete_sequence=PALETTE)))

    left, right = st.columns(2)
    with left:
        section("🏆 Top 10 Customers")
        top = run_query(f"""
            SELECT c.customer_name AS Customer, c.city AS City, COUNT(o.order_id) AS Orders,
                   ROUND({REV}, 0) AS Spending
            FROM orders_clean o JOIN customers_clean c ON o.customer_id = c.customer_id
            WHERE o.returned = 0 AND {VALID} {yf}
            GROUP BY c.customer_id ORDER BY Spending DESC LIMIT 10
        """)
        st.dataframe(top, hide_index=True)
    with right:
        section("⭐ Top 10 Products")
        prod = run_query(f"""
            SELECT p.product_name AS Product, p.category AS Category, ROUND({REV}, 0) AS Revenue
            FROM orders_clean o JOIN products_clean p ON o.product_id = p.product_id
            WHERE o.returned = 0 AND {VALID} {yf}
            GROUP BY p.product_id ORDER BY Revenue DESC LIMIT 10
        """)
        st.dataframe(prod, hide_index=True)

# =============== CHURN ===============
elif page == "🔮 Churn Prediction":
    hero("🔮 Customer Churn Prediction", "Predict whether a customer will stop buying in the next 3 months")

    c1, c2 = st.columns(2)
    with c1:
        total_orders = st.number_input("Total orders", min_value=1, value=8)
        total_spending = st.number_input("Total spending (PKR)", min_value=0.0, value=80000.0)
        days_since = st.number_input("Days since last order", min_value=0, value=30)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with c2:
        return_rate = st.slider("Return rate", 0.0, 1.0, 0.05)
        delivery = st.slider("Average delivery days", 1.0, 10.0, 3.5)
        membership = st.selectbox("Membership type", ["Standard", "Silver", "Gold", "Premium"])

    if st.button("Predict Churn"):
        customer = pd.DataFrame([{
            "total_orders": total_orders,
            "total_spending": total_spending,
            "average_order_value": total_spending / total_orders,
            "return_rate": return_rate,
            "average_delivery_days": delivery,
            "days_since_last_order": days_since,
            "age": age,
            "membership_type": membership,
        }])
        prob = float(churn_model.predict_proba(customer)[0][1])

        if prob >= 0.5:
            result_card(f"⚠️ Likely to Churn ({prob:.0%})", "This customer may stop buying soon", "red")

            section("💡 Suggested Retention Strategy")
            if days_since > 60:
                tip("Win-back campaign", f"No order in {days_since} days. Send a 'We miss you' email/SMS with a 10–15% discount code.")
            if return_rate > 0.15:
                tip("Fix product experience", "High return rate. Call the customer for feedback and recommend better-rated products.")
            if delivery > 5:
                tip("Faster / free delivery", "Slow deliveries hurt loyalty. Offer free express delivery on the next order.")
            if total_orders <= 3:
                tip("Second-order incentive", "New customer with few orders. Give a coupon on their next purchase to build the habit.")
            if membership in ["Standard", "Silver"]:
                tip("Membership upgrade", f"Offer a free 1-month upgrade from {membership} to Gold to increase engagement.")
            if total_spending > 100000:
                tip("VIP treatment", "High-value customer. Assign priority support and send exclusive early-access deals.")
            tip("Personalised recommendations", "Send product suggestions based on their past categories via email or WhatsApp.")
        else:
            result_card(f"✅ Likely to Stay ({prob:.0%} churn risk)", "This customer looks loyal", "green")
            section("💡 Suggested Action")
            tip("Keep them engaged", "Reward loyalty with points, ask for a review, and promote related products.")

# =============== SENTIMENT ===============
else:
    hero("💬 Review Sentiment Analysis", "Type a customer review and the NLP model will classify its sentiment")

    review = st.text_area("Customer review", "Very happy with this purchase, great quality.", height=120)

    if st.button("Analyze"):
        text = clean_text(review)
        vectorizer = sentiment_model.steps[0][1]          # first step of the pipeline = TF-IDF
        known_words = vectorizer.transform([text]).nnz if text else 0

        if text == "":
            st.warning("Please enter a review.")
        elif known_words == 0:
            result_card("🤔 Not sure", "Please write a proper product review (e.g. 'quality is bad')", "grey")
        else:
            result = sentiment_model.predict([text])[0]
            emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}[result]
            color = {"Positive": "green", "Negative": "red", "Neutral": "blue"}[result]
            result_card(f"{emoji} {result}", "Predicted sentiment", color)

html("FOOTER")