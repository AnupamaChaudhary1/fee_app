import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from word2number import w2n
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import joblib

st.set_page_config(page_title="School Fee Analysis", layout="wide")
st.title("📊 School Fee Analysis & Prediction App")

# 1. Upload CSV
uploaded_file = st.file_uploader("Upload your School Fee CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1️⃣ Initial Dataset Preview")
    st.write(df.head())

    # ====== Data Cleaning Steps ======
    st.subheader("2️⃣ Data Cleaning")

    # Fill Missing
    df['Admission Fee (NPR)'] = df['Admission Fee (NPR)'].fillna(df['Admission Fee (NPR)'].median())
    df['Technology Access Index'] = df['Technology Access Index'].fillna(df['Technology Access Index'].mean())

    # Word to Number (Monthly Fee)
    def convert_to_number(value):
        try:
            return w2n.word_to_num(value)
        except:
            try:
                return float(value)
            except:
                return None
    df['Monthly Fee (NPR)'] = df['Monthly Fee (NPR)'].astype(str).apply(convert_to_number)

    # Clean Student-Teacher Ratio
    df['Student-Teacher Ratio'] = pd.to_numeric(df['Student-Teacher Ratio'].astype(str).str.strip(), errors='coerce')
    df['Student-Teacher Ratio'] = df['Student-Teacher Ratio'].fillna(df['Student-Teacher Ratio'].mean())

    # Remove Duplicates
    df = df.drop_duplicates()

    # Handle Outliers in Annual Tuition Fee
    Q1 = df['Annual Tuition Fee (NPR)'].quantile(0.25)
    Q3 = df['Annual Tuition Fee (NPR)'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['Annual Tuition Fee (NPR)'] = df['Annual Tuition Fee (NPR)'].clip(lower, upper)

    st.success("✅ Cleaning Completed")
    st.write(df.head())

    # ========== Visualizations ==========
    st.subheader("3️⃣ Visualizations")

    tab1, tab2, tab3 = st.tabs(["Histogram", "Boxplot", "Heatmap"])

    with tab1:
        st.write("Annual Tuition Fee Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Annual Tuition Fee (NPR)'], bins=50, kde=True, ax=ax)
        st.pyplot(fig)

    with tab2:
        st.write("Boxplot - Monthly Fee")
        fig, ax = plt.subplots()
        sns.boxplot(x=df['Monthly Fee (NPR)'], ax=ax)
        st.pyplot(fig)

    with tab3:
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # ========== Linear Regression ==========
    st.subheader("4️⃣ Predict Tuition Fee (Linear Regression)")

    features = ['Infrastructure Score', 'Technology Access Index', 'Average Academic Score (%)']
    target = 'Annual Tuition Fee (NPR)'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    st.write(f"📉 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    st.write(f"Sample Prediction: {lr.predict([[75, 80, 70]])[0]:,.2f} NPR")

    # ========== Save Cleaned Data ==========
    cleaned_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Cleaned CSV", cleaned_csv, file_name="cleaned_data.csv")

    # ========== Save Model ==========
    joblib.dump(lr, "linear_model.pkl")
    st.success("Linear Regression model saved as linear_model.pkl")

else:
    st.info("👆 Please upload a CSV file to proceed.")
