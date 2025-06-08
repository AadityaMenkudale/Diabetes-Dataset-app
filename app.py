import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Diabetes Prediction - SVM", layout="centered")

# Title
st.title("🧠 Diabetes Prediction using SVM")
st.markdown("Upload your dataset to apply SVM classification.")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Check if necessary column exists
    if "Outcome" not in df.columns:
        st.error("❌ The dataset must contain an 'Outcome' column as the target.")
    else:
        # Basic Info
        st.write("✅ Dataset Loaded. Shape:", df.shape)

        # Missing Values
        if df.isnull().sum().sum() > 0:
            st.warning(
                "⚠️ Dataset contains missing values. Rows with missing data will be dropped.")
            df.dropna(inplace=True)

        # Split Features and Target
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42)

        # SVM Model
        model = SVC(kernel='linear', C=1.0, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        st.metric(label="🎯 Accuracy", value=f"{acc * 100:.2f}%")

        # Classification Report
        st.subheader("📊 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
