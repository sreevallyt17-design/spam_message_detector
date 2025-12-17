import streamlit as st
import pickle
import os
import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL_PATH = "model/spam_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

def train_and_save_model():
    df = pd.read_csv("data/spam.csv", encoding="latin-1")
    df = df[['v1','v2']]
    df.columns = ['label','message']
    df = df.dropna(subset=['message'])

    df['message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham':0, 'spam':1})

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    model = MultinomialNB()
    model.fit(X, y)

    os.makedirs("model", exist_ok=True)
    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

    return model, vectorizer


# 🔁 Load or Train
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
else:
    model, vectorizer = train_and_save_model()


# ---------------- Streamlit UI ----------------
st.title("📩 Spam Message Detector")

message = st.text_area("Enter your message")

if st.button("Check"):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)

    if result[0] == 1:
        st.error("🚨 This is SPAM")
    else:
        st.success("✅ This is NOT spam")
