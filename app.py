nltk.download('stopwords')
import streamlit as st
import pickle
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# =========================
# Load Model + Vectorizer
# =========================
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

ps = PorterStemmer()

# =========================
# Text Cleaning Function
# =========================
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# =========================
# Voice Input Function
# =========================
def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except:
            return ""

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🍽️", layout="centered")

# Dark Theme Styling
st.markdown("""
<style>
body {background-color: #0e1117;}
.stTextArea textarea {
    background-color: #1c1f26;
    color: white;
    border-radius: 10px;
}
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# UI LAYOUT
# =========================
st.title("🍽️ Restaurant Review Sentiment Analyzer")
st.markdown("### ⚡ AI-Powered Sentiment Detection")
st.write("Enter a review or use voice input to detect sentiment.")

review = st.text_area("✍️ Write your review here")

# Voice Button
if st.button("🎙️ Use Voice Input"):
    spoken_text = voice_to_text()
    if spoken_text:
        st.success(f"You said: {spoken_text}")
        review = spoken_text

# =========================
# Prediction
# =========================
if st.button("Analyze Sentiment") and review.strip() != "":
    cleaned = clean_text(review)
    vector_input = vectorizer.transform([cleaned]).toarray()

    prediction = model.predict(vector_input)

    # Probability
    try:
        probs = model.predict_proba(vector_input)[0]
        confidence = max(probs)
    except:
        # For models without predict_proba (like SVM)
        decision = model.decision_function(vector_input)[0]
        probs = [1 - decision, decision]
        confidence = max(probs)

    # Result Display
    if prediction[0] == 1:
        st.success(f"😊 Positive Review (Confidence: {confidence:.2f})")
    else:
        st.error(f"😡 Negative Review (Confidence: {confidence:.2f})")

    # =========================
    # Probability Graph
    # =========================
    labels = ["Negative", "Positive"]
    plt.figure(figsize=(4,3))
    plt.bar(labels, probs, color=["red","green"])
    plt.title("Prediction Confidence")
    plt.ylabel("Probability")
    st.pyplot(plt)
