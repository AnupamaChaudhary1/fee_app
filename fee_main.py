# import pandas as pd
# import numpy as np
# import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt
# from word2number import w2n
# from scipy import stats
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
# import joblib

# st.set_page_config(page_title="School Fee Analysis", layout="wide")
# st.title("📊 School Fee Analysis & Prediction App")

# # 1. Upload CSV
# uploaded_file = st.file_uploader("Upload your School Fee CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     st.subheader("1️⃣ Initial Dataset Preview")
#     st.write(df.head())

#     # ====== Data Cleaning Steps ======
#     st.subheader("2️⃣ Data Cleaning")

#     # Fill Missing
#     df['Admission Fee (NPR)'] = df['Admission Fee (NPR)'].fillna(df['Admission Fee (NPR)'].median())
#     df['Technology Access Index'] = df['Technology Access Index'].fillna(df['Technology Access Index'].mean())

#     # Word to Number (Monthly Fee)
#     def convert_to_number(value):
#         try:
#             return w2n.word_to_num(value)
#         except:
#             try:
#                 return float(value)
#             except:
#                 return None
#     df['Monthly Fee (NPR)'] = df['Monthly Fee (NPR)'].astype(str).apply(convert_to_number)

#     # Clean Student-Teacher Ratio
#     df['Student-Teacher Ratio'] = pd.to_numeric(df['Student-Teacher Ratio'].astype(str).str.strip(), errors='coerce')
#     df['Student-Teacher Ratio'] = df['Student-Teacher Ratio'].fillna(df['Student-Teacher Ratio'].mean())

#     # Remove Duplicates
#     df = df.drop_duplicates()

#     # Handle Outliers in Annual Tuition Fee
#     Q1 = df['Annual Tuition Fee (NPR)'].quantile(0.25)
#     Q3 = df['Annual Tuition Fee (NPR)'].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     df['Annual Tuition Fee (NPR)'] = df['Annual Tuition Fee (NPR)'].clip(lower, upper)

#     st.success("✅ Cleaning Completed")
#     st.write(df.head())

#     # ========== Visualizations ==========
#     st.subheader("3️⃣ Visualizations")

#     tab1, tab2, tab3 = st.tabs(["Histogram", "Boxplot", "Heatmap"])

#     with tab1:
#         st.write("Annual Tuition Fee Distribution")
#         fig, ax = plt.subplots()
#         sns.histplot(df['Annual Tuition Fee (NPR)'], bins=50, kde=True, ax=ax)
#         st.pyplot(fig)

#     with tab2:
#         st.write("Boxplot - Monthly Fee")
#         fig, ax = plt.subplots()
#         sns.boxplot(x=df['Monthly Fee (NPR)'], ax=ax)
#         st.pyplot(fig)

#     with tab3:
#         st.write("Correlation Heatmap")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
#         st.pyplot(fig)

#     # ========== Linear Regression ==========
#     st.subheader("4️⃣ Predict Tuition Fee (Linear Regression)")

#     features = ['Infrastructure Score', 'Technology Access Index', 'Average Academic Score (%)']
#     target = 'Annual Tuition Fee (NPR)'

#     X = df[features]
#     y = df[target]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     lr = LinearRegression()
#     lr.fit(X_train, y_train)
#     y_pred = lr.predict(X_test)

#     st.write(f"📉 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
#     st.write(f"Sample Prediction: {lr.predict([[75, 80, 70]])[0]:,.2f} NPR")

#     # ========== Save Cleaned Data ==========
#     cleaned_csv = df.to_csv(index=False).encode('utf-8')
#     st.download_button("⬇️ Download Cleaned CSV", cleaned_csv, file_name="cleaned_data.csv")

#     # ========== Save Model ==========
#     joblib.dump(lr, "linear_model.pkl")
#     st.success("Linear Regression model saved as linear_model.pkl")

# else:
#     st.info("👆 Please upload a CSV file to proceed.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

st.set_page_config(page_title="Nepal School Fee Predictor", layout="wide")
st.title("Private School Fee Analysis in Nepal")

# File upload section
st.sidebar.header("Upload Dataset")
user_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

def load_data():
    if user_file:
        return pd.read_csv(user_file)
    elif os.path.exists("cleaned_data.csv"):
        return pd.read_csv("cleaned_data.csv")
    else:
        st.warning("No data available. Please upload a file or check local dataset.")
        return pd.DataFrame()

# Load data
df = load_data()

if not df.empty:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Allow user to manually enter data
    st.subheader("Manual Data Entry")
    infra = st.number_input("Infrastructure Score", min_value=0.0, max_value=100.0, value=50.0)
    tech = st.number_input("Technology Access Index", min_value=0.0, max_value=100.0, value=50.0)
    academic = st.number_input("Average Academic Score (%)", min_value=0.0, max_value=100.0, value=50.0)

    if st.button("Add Entry"):
        new_row = {
            'Infrastructure Score': infra,
            'Technology Access Index': tech,
            'Average Academic Score (%)': academic,
            'Annual Tuition Fee (NPR)': np.nan
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.success("Entry added!")

    # Prepare features and labels
    if all(col in df.columns for col in ['Infrastructure Score', 'Technology Access Index', 'Average Academic Score (%)', 'Annual Tuition Fee (NPR)']):
        df_clean = df.dropna(subset=['Infrastructure Score', 'Technology Access Index', 'Average Academic Score (%)', 'Annual Tuition Fee (NPR)'])

        X = df_clean[['Infrastructure Score', 'Technology Access Index', 'Average Academic Score (%)']]
        y = df_clean['Annual Tuition Fee (NPR)']

        # Handle any inf or NaNs
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]

        if not X.empty:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.subheader("Model Evaluation")
            st.write("MAE:", mean_absolute_error(y_test, y_pred))
            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

            st.subheader("Prediction on Manual Input")
            pred_fee = model.predict([[infra, tech, academic]])[0]
            st.success(f"Predicted Tuition Fee: NPR {pred_fee:,.2f}")
        else:
            st.warning("Not enough clean data to train the model.")
    else:
        st.warning("Required columns not found in dataset.")
else:
    st.warning("Please upload or load a valid dataset.")
