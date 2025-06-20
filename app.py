import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# Sidebar Navigation
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Dataset Preview", "Data Visualization"])

# Preprocessing
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Model
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy in sidebar
st.sidebar.markdown("### 📊 Model Accuracy")
st.sidebar.metric("SVM Accuracy", f"{accuracy * 100:.2f}%")

# Page 1: Prediction
if page == "Prediction":
    st.title("🧠 Diabetes Prediction")

    st.markdown("### Enter Patient Data:")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(col, min_value=0.0, format="%.2f")

    if st.button("Predict"):
        input_array = np.array([list(input_data.values())])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        st.success(f"🔍 The model predicts: **{result}**")

# Page 2: Dataset Preview
elif page == "Dataset Preview":
    st.title("📄 Dataset Preview")
    st.write("Shape of dataset:", df.shape)
    st.dataframe(df.head(50))

# Page 3: Data Visualization
elif page == "Data Visualization":
    st.title("📊 Data Visualization")

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt.gcf())

    st.subheader("Outcome Count")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Outcome", ax=ax)
    st.pyplot(fig)
