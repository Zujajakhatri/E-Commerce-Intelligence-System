# 🛒 EcomIQ – AI-Powered E-Commerce Customer Intelligence System

**Prepared by:** Zujaja

An end-to-end data science project that starts from a raw SQLite database and ends with a deployed Streamlit app. It cleans the data, analyses sales with SQL, predicts customer churn and classifies review sentiment.

🔗 **Live App:** [Add your Streamlit link here]

---

## 📌 Project Summary

- **Data:** One SQLite database (`ecommerce_hackathon.db`) with 4 tables: customers, products, orders and reviews (100,000 rows).
- **Cleaning:** Fixed missing values, inconsistent city/category names and invalid dates; removed negative prices, invalid quantities, invalid ratings and empty reviews.
- **SQL Analysis:** Total net revenue, top customers, category performance, monthly trend and top products.
- **Machine Learning:** Churn prediction with Logistic Regression and Random Forest.
- **Deep Learning:** A small neural network (32 → 16 → 1) compared with the ML models.
- **NLP:** Review sentiment classification with TF-IDF and Logistic Regression.
- **App:** A Streamlit dashboard with 3 pages: Dashboard, Churn Prediction and Sentiment Analysis.

---

## 💡 Key Business Insights

1. **Heavy dependence on Electronics:** Electronics brings in about 70% of total revenue (about PKR 605M of PKR 865M). A drop in electronics demand would hurt the business badly, so other categories like Home & Kitchen and Fashion should be grown.

2. **Fashion has the highest return rate (11.3%):** Almost 1 in 9 fashion orders is returned, which means extra delivery costs and lost revenue. Better size guides and real product photos could reduce returns.

3. **Karachi and Lahore generate about 45% of revenue:** These cities should get priority in marketing and fast delivery, while cities like Multan, Faisalabad and Peshawar offer room for growth.

---

## 🤖 Model Results

| Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.711 | 0.827 | 0.782 |
| **Random Forest (selected)** | **0.718** | **0.841** | **0.783** |
| Neural Network | 0.709 | 0.838 | 0.773 |

Random Forest was selected because it catches the most customers who are about to leave (highest Recall). The neural network did not improve the results, so it was not worth the extra complexity.

---

## 📁 Project Structure

```
├── app.py                    # Streamlit application
├── style.css                 # App styling
├── index.html                # HTML components for the app
├── ecommerce_hackathon.db    # SQLite database (raw + clean tables)
├── queries.sql               # The 5 required SQL queries
├── ecomiq_analysis.ipynb     # Cleaning, EDA, ML, DL and NLP notebook
├── requirements.txt          # Python libraries
├── models/
│   ├── churn_model.pkl       # Random Forest churn model
│   └── sentiment_model.pkl   # TF-IDF + Logistic Regression sentiment model
└── .streamlit/config.toml    # App theme
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tools Used

Python, SQLite, Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib, Plotly and Streamlit.

---

👩‍💻 **Author:** Zujaja · Data Science Final Hackathon 2026
