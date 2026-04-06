import streamlit as st
import pickle
import re
import string
import os

# 📝 History setup
if "history" not in st.session_state:
    st.session_state.history = []

# 📦 Load model
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "cyberbullying_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb"))

# 🧹 Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# 🌍 Simple translation (optional)
def translate_text(text):
    if "tui" in text or "tor" in text:
        return "bad language"
    return text

# 🤖 Prediction
def predict(text):
    text = translate_text(text)
    text = clean_text(text)
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    prob = model.predict_proba(text_tfidf)[0][1]
    return prediction, prob

# 🎨 UI (TOP PART)
st.title("🚫😡 AI Cyberbullying Detector 🔥")
st.markdown("<h3 style='text-align: center;'>Type your message below 👇</h3>", unsafe_allow_html=True)

# 🧹 Clear history button
if st.button("🧹 Clear History"):
    st.session_state.history = []

# 📥 Input
user_input = st.text_area("Enter a sentence:")

# 🔍 Analyze button
if st.button("Analyze"):
    if user_input:
        pred, prob = predict(user_input)

        # 😡😊 Result
        if pred == 1:
            st.error(f"😡 Cyberbullying ({round(prob*100,2)}%)")
        else:
            st.success(f"😊 Not Cyberbullying ({round(prob*100,2)}%)")

        # 📊 Confidence bar
        st.progress(int(prob * 100))
        st.write(f"Confidence: {round(prob*100,2)}%")

        # 📝 Save history
        st.session_state.history.append((user_input, pred, prob))

# 📜 History display
st.subheader("📝 History")

for text, p, pr in st.session_state.history:
    label = "😡 Cyberbullying" if p == 1 else "😊 Not Cyberbullying"
    st.write(f"{text} → {label} ({round(pr*100,2)}%)")