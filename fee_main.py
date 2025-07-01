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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="School Fee Analyzer", layout="wide")

st.title("📚 School Fee Analysis & Prediction App")

# --- 1. Load default data or user file ---
st.sidebar.header("Upload or Use Sample Data")

uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File Uploaded")
else:
    # Sample default data
    df = pd.DataFrame({
        "Annual Tuition Fee (NPR)": np.random.randint(20000, 100000, 100),
        "Monthly Fee (NPR)": np.random.randint(1500, 8000, 100),
        "Technology Access Index": np.random.uniform(0, 1, 100),
        "Infrastructure Score": np.random.uniform(1, 10, 100),
        "Scholarship % Availability": np.random.randint(0, 100, 100),
        "Fee Increase % (YoY)": np.random.uniform(1, 15, 100)
    })

# --- 2. Manual Entry ---
st.sidebar.header("➕ Add New Entry")

with st.sidebar.form("manual_form"):
    annual_fee = st.number_input("Annual Tuition Fee (NPR)", 1000, 200000, 30000)
    monthly_fee = st.number_input("Monthly Fee (NPR)", 500, 15000, 2500)
    tech_index = st.slider("Technology Access Index", 0.0, 1.0, 0.5)
    infra_score = st.slider("Infrastructure Score", 0.0, 10.0, 5.0)
    scholarship_pct = st.slider("Scholarship % Availability", 0, 100, 20)
    fee_increase = st.slider("Fee Increase % (YoY)", 0.0, 25.0, 5.0)

    submitted = st.form_submit_button("Add to Dataset")
    if submitted:
        new_row = {
            "Annual Tuition Fee (NPR)": annual_fee,
            "Monthly Fee (NPR)": monthly_fee,
            "Technology Access Index": tech_index,
            "Infrastructure Score": infra_score,
            "Scholarship % Availability": scholarship_pct,
            "Fee Increase % (YoY)": fee_increase
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.sidebar.success("✅ Entry Added to Data")

# --- 3. Data Preview ---
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# --- 4. Model Training & Prediction ---
st.subheader("📈 Predict Annual Tuition Fee")

features = ['Monthly Fee (NPR)', 'Technology Access Index', 'Infrastructure Score',
            'Scholarship % Availability', 'Fee Increase % (YoY)']
target = 'Annual Tuition Fee (NPR)'

# Ensure no NaN in training
df.dropna(inplace=True)

X = df[features]
y = df[target]

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    st.markdown(f"**Model Accuracy (R² Score)**: `{score:.2f}`")

    st.markdown("### 🔮 Try Custom Prediction")

    col1, col2 = st.columns(2)
    with col1:
        input_monthly = st.number_input("Monthly Fee", 1000, 20000, 3000)
        input_tech = st.slider("Technology Access Index", 0.0, 1.0, 0.6)
        input_infra = st.slider("Infrastructure Score", 0.0, 10.0, 6.0)
    with col2:
        input_scholarship = st.slider("Scholarship %", 0, 100, 10)
        input_fee_increase = st.slider("Fee Increase %", 0.0, 20.0, 4.0)

    if st.button("Predict Fee"):
        prediction = model.predict([[input_monthly, input_tech, input_infra, input_scholarship, input_fee_increase]])
        st.success(f"Predicted Annual Tuition Fee: NPR {prediction[0]:,.2f}")

except Exception as e:
    st.error("❌ Model training failed. Ensure enough valid data and no missing values.")
    st.exception(e)

# --- 5. Visualizations ---
st.subheader("📉 Data Visualizations")

# 1. Histogram of Annual Tuition Fee
st.markdown("#### Histogram of Annual Tuition Fee")
fig1, ax1 = plt.subplots()
sns.histplot(df['Annual Tuition Fee (NPR)'], bins=30, kde=True, ax=ax1)
ax1.set_title("Distribution of Annual Tuition Fee")
st.pyplot(fig1)

# 2. Boxplot of Monthly Fee
st.markdown("#### Boxplot of Monthly Fee")
fig2, ax2 = plt.subplots()
sns.boxplot(x=df['Monthly Fee (NPR)'], ax=ax2)
ax2.set_title("Boxplot of Monthly Fee")
st.pyplot(fig2)

# 3. Correlation Heatmap
st.markdown("#### Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
ax3.set_title("Feature Correlation")
st.pyplot(fig3)

# 4. Scatterplot - Infra vs Tuition
st.markdown("#### Infrastructure Score vs Tuition Fee")
fig4, ax4 = plt.subplots()
sns.scatterplot(x='Infrastructure Score', y='Annual Tuition Fee (NPR)', data=df, ax=ax4)
ax4.set_title("Infra Score vs Annual Tuition Fee")
st.pyplot(fig4)

# 5. Violin Plot - Tech Index
st.markdown("#### Violin Plot of Technology Access Index")
fig5, ax5 = plt.subplots()
sns.violinplot(y=df['Technology Access Index'], ax=ax5)
st.pyplot(fig5)

# 6. Barplot - Scholarship vs Fee Increase
st.markdown("#### Fee Increase % by Scholarship Availability")
fig6, ax6 = plt.subplots()
sns.barplot(x='Scholarship % Availability', y='Fee Increase % (YoY)', data=df, ax=ax6)
st.pyplot(fig6)

# 7. Pairplot
st.markdown("#### Pairplot of Key Features (May take time)")
pairplot_fig = sns.pairplot(df[features + [target]])
st.pyplot(pairplot_fig.figure)

st.success("✅ App Ready")
