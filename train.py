import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[['v1','v2']]  # Keep only necessary columns
df.columns = ['label','message']

# Drop rows with missing messages
df = df.dropna(subset=['message'])

# -----------------------------
# STEP 2: CLEAN MESSAGES
# -----------------------------
df['message'] = df['message'].apply(clean_text)

# -----------------------------
# STEP 3: ENCODE LABELS
# -----------------------------
# 0 = ham (not spam), 1 = spam
df['label'] = df['label'].map({'ham':0, 'spam':1})

# -----------------------------
# STEP 4: VECTORIZE TEXT
# -----------------------------
# TF-IDF with n-grams (1,2) for better spam detection
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(df['message'])
y = df['label']

# -----------------------------
# STEP 5: TRAIN MODEL
# -----------------------------
model = MultinomialNB()
model.fit(X, y)

# -----------------------------
# STEP 6: TEST MODEL ON SAMPLE MESSAGES
# -----------------------------
sample_messages = [
    "you won 1 crore cashback",
    "how are you today",
    "free entry in a competition"
]

print("\n--- Test Predictions ---")
for msg in sample_messages:
    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    print(msg, "=>", "SPAM" if pred[0]==1 else "NOT SPAM")

# -----------------------------
# STEP 7: SAVE MODEL AND VECTORIZER
# -----------------------------
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/spam_model.pkl","wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl","wb"))

print("\nModel trained and saved successfully")
