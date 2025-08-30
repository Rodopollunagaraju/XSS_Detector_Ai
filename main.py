import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ------------------------------
# Load & preprocess dataset
# ------------------------------
df = pd.read_csv("XSS_dataset.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.rename(columns={"Sentence": "input", "Label": "label"})

def clean_text(text):
    text = re.sub(r"[\t\n]", " ", text)
    return text.lower().strip()

df["input"] = df["input"].apply(clean_text)

# ------------------------------
# Train ML model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["input"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI XSS Detector", page_icon="🤖", layout="centered")

st.title("🤖 AI-Powered XSS Detector")
st.write("Enter any text or HTML input. The ML model will predict if it's malicious (XSS) or safe.")

user_input = st.text_area("🔎 Enter text to check:", height=150)

if st.button("Check for XSS"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        prob = model.predict_proba(vec_input)[0][1]

        st.write(f"### 🧮 Probability of Attack: {prob:.2f}")
        st.progress(int(prob*100))

        if prediction == 1:
            st.error("🚨 Potential XSS Attack Detected!")
        else:
            st.success("✅ Input seems Safe.")
