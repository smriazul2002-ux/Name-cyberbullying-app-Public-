import streamlit as st
import pickle
import re
import string

# Load model & vectorizer
model = pickle.load(open("cyberbullying_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Prediction function
def predict(text):
    text = clean_text(text)
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    prob = model.predict_proba(text_tfidf)[0][1]
    return prediction, prob

# UI
st.title("🚫 Cyberbullying Detection App")

user_input = st.text_area("Enter a sentence:")

if st.button("Analyze"):
    if user_input:
        pred, prob = predict(user_input)

        if pred == 1:
            st.error(f"Cyberbullying ❌ ({round(prob*100,2)}%)")
        else:
            st.success(f"Not Cyberbullying ✅ ({round(prob*100,2)}%)")