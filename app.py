import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = "model/fake_news_pipeline.pkl"

    # If model already exists, just load it
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # Otherwise train it on the spot
    st.info("Training model for first time — this takes ~1 minute...")

    df_true = pd.read_csv("True.csv");  df_true["label"] = 1
    df_fake = pd.read_csv("Fake.csv");  df_fake["label"] = 0
    df = pd.concat([df_true, df_fake], ignore_index=True)

    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2),
                                   sublinear_tf=True, stop_words="english")),
        ("clf",   LogisticRegression(C=5, max_iter=1000, random_state=42))
    ])
    pipeline.fit(df["content"], df["label"])

    os.makedirs("model", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline

pipeline = load_model()

with st.sidebar:
    st.title("Settings")
    threshold = st.slider(
        "Classification threshold",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05
    )
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("Paste any article → the model returns a probability of it being real news.")



st.title("Fake News Detector")
st.markdown("Paste a news article below to check its credibility.")

with st.form("prediction_form"):
    title = st.text_input("Article Title", placeholder="Enter the headline...")
    body  = st.text_area("Article Body",   placeholder="Paste the full text...", height=250)
    submitted = st.form_submit_button("Analyze Article")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

if submitted:
    combined = clean_text(f"{title} {body}")
    prob_real = pipeline.predict_proba([combined])[0][1]
    prob_fake = 1 - prob_real
    is_real   = prob_real >= threshold

    # Display the results
    if is_real:
        st.success(f"✅ Likely REAL NEWS — {prob_real*100:.1f}% confidence")
    else:
        st.error(f"🚨 Likely FAKE NEWS — {prob_fake*100:.1f}% confidence")

    col1, col2 = st.columns(2)
    col1.metric("P(Real)", f"{prob_real*100:.1f}%")
    col2.metric("P(Fake)", f"{prob_fake*100:.1f}%")

    st.bar_chart(
        pd.DataFrame({"Probability": [prob_real, prob_fake]},
                    index=["Real", "Fake"])
    )

    # Show influential words/why the model made its decision
    st.markdown("#### Most Influential Words")
    try:
        vectorizer    = pipeline.named_steps["tfidf"]
        feature_names = np.array(vectorizer.get_feature_names_out())
        tfidf_vector  = vectorizer.transform([combined]).toarray()[0]
        top_indices   = np.argsort(tfidf_vector)[-15:][::-1]

        top_df = pd.DataFrame({
            "Term":        feature_names[top_indices],
            "TF-IDF Score": tfidf_vector[top_indices]
        })
        st.dataframe(top_df.set_index("Term"))
    except Exception:
        st.info("Could not extract terms.")