import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore

# Page config
st.set_page_config(page_title="Student Math Score Predictor", layout="wide")

# Load data
df = pd.read_csv('./src/StudentsPerformance.csv')
df_encoded = df.copy()

# Label Encoding
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df_encoded[col] = le.fit_transform(df[col])

# Prepare features and target
X = df_encoded.drop(columns='math score')
y = df_encoded['math score']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
model = LinearRegression()
model.fit(X_scaled, y)

# Metrics
y_pred = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# App sections
st.title("📊 Student Math Score Predictor")
st.markdown("### Predict student performance using Machine Learning")

tab1, tab2, tab3, tab4 = st.tabs(["🏁 Introduction", "📈 Exploratory Data Analysis", "🧠 Model & Prediction", "✅ Conclusion"])

# Introduction Section
with tab1:
    st.subheader("📌 About the Dataset")
    st.markdown("""
    The **StudentsPerformance.csv** dataset includes various attributes such as gender, ethnicity, parental education level, lunch type, and test preparation course.
    
    The goal of this project is to **predict the math score** of a student based on their attributes using a **Linear Regression** model.
    """)
    st.dataframe(df.head(), use_container_width=True)

# EDA Section
with tab2:
    st.subheader("🔍 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Distribution of Math Scores")
        fig, ax = plt.subplots()
        sns.histplot(df['math score'], bins=20, kde=True, ax=ax, color='skyblue')
        st.pyplot(fig)

    with col2:
        st.markdown("#### 🧪 Average Scores by Test Preparation")
        avg_scores = df.groupby("test preparation course")["math score"].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=avg_scores, x="test preparation course", y="math score", ax=ax, palette="viridis")
        st.pyplot(fig)

    st.markdown("#### 📉 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Model Section
with tab3:
    st.subheader("⚙️ Linear Regression Model")
    st.markdown(f"""
    - Model: **Linear Regression**  
    - Mean Squared Error (MSE): `{mse:.2f}`  
    - R² Score: `{r2:.2f}`  
    """)

    st.markdown("### 🎯 Predict a Student's Math Score")
    st.info("Fill in the student information below:")

    input_data = {}
    for col in X.columns:
        if df[col].dtype == 'object':
            options = df[col].unique()
            input_data[col] = st.selectbox(f"{col}", options)
        else:
            input_data[col] = st.number_input(f"{col}", min_value=0, max_value=100, value=50)

    input_df = pd.DataFrame([input_data])

    # Encode input
    for col in input_df.select_dtypes(include='object'):
        input_df[col] = le.fit(df[col]).transform(input_df[col])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    if st.button("🔮 Predict Math Score"):
        prediction = model.predict(input_scaled)[0]
        st.success(f"📘 **Predicted Math Score: {prediction:.2f}**")

# Conclusion
with tab4:
    st.subheader("📝 Conclusion")
    st.markdown("""
    - The **Linear Regression model** effectively predicts math scores based on student information.
    - Key factors influencing scores include **test preparation**, **parental education**, and **lunch type**.
    - This project demonstrates the application of **EDA** and **ML modeling** to make informed predictions.
    
    ---  
    💡 *This interactive tool can be further enhanced by exploring non-linear models or using external deployment platforms.*
    """)

    st.balloons()
