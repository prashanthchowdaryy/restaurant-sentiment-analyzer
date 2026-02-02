import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Comparison", layout="centered")

st.title("📊 Machine Learning Model Performance Dashboard")
st.write("Compare different algorithms used for sentiment analysis.")

# ==========================
# Accuracy Scores (your real results)
# ==========================
models = {
    "Logistic Regression": 0.72,
    "KNN": 0.68,
    "SVM": 0.755,
    "Naive Bayes": 0.70,
    "Decision Tree": 0.69,
    "Random Forest": 0.73,
    "XGBoost": 0.74
}

df = pd.DataFrame(models.items(), columns=["Model", "Accuracy"])

# ==========================
# Select Model
# ==========================
selected_model = st.selectbox("Select a model to view performance", df["Model"])

accuracy_value = df[df["Model"] == selected_model]["Accuracy"].values[0]

st.subheader(f"🔍 {selected_model} Performance")
st.metric("Accuracy", f"{accuracy_value*100:.2f}%")

# ==========================
# Accuracy Comparison Graph
# ==========================
st.subheader("📈 Model Accuracy Comparison")

plt.figure(figsize=(8,4))
plt.bar(df["Model"], df["Accuracy"], color="skyblue")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
st.pyplot(plt)

# ==========================
# Difference From Best Model
# ==========================
best_model = df.loc[df["Accuracy"].idxmax()]
difference = best_model["Accuracy"] - accuracy_value

st.subheader("⚖️ Difference From Best Model")

if selected_model != best_model["Model"]:
    st.warning(f"{selected_model} is {difference*100:.2f}% lower than best model ({best_model['Model']}).")
else:
    st.success(f"{selected_model} is the BEST performing model 🎉")

# ==========================
# Model Descriptions
# ==========================
descriptions = {
    "Logistic Regression": "Linear model good for baseline text classification.",
    "KNN": "Distance-based model; slower on large text data.",
    "SVM": "Great for high-dimensional text data, best performer here.",
    "Naive Bayes": "Fast probabilistic model often used in NLP.",
    "Decision Tree": "Tree-based model but can overfit text features.",
    "Random Forest": "Ensemble tree model with improved stability.",
    "XGBoost": "Boosting algorithm with strong performance."
}

st.subheader("🧠 Model Info")
st.info(descriptions[selected_model])
