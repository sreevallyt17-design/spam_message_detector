import streamlit as st
import pickle
from preprocess import clean_text

# Load trained model and vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("📩 Spam Message Detector")

message = st.text_area("Enter your message")

if st.button("Check"):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)

    # Correct mapping: 1 = spam, 0 = ham
    if result[0] == 1:
        st.error("🚨 This is SPAM")
    else:
        st.success("✅ This is NOT spam")
