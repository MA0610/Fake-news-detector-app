import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import torch

# Page Config

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)

# Custom CSS

st.markdown("""
<style>
    .result-box {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1.5rem 0;
    }
    .real-news { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .fake-news { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


# Text Cleaning

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# URL Scraper

def scrape_article(url: str):
    """Extract title and body text from a news URL."""
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception:
        return None, None


# Model Loaders

@st.cache_resource
def load_lr_model():
    """Load or train the Logistic Regression pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "fake_news_pipeline.pkl")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    st.info("Training Logistic Regression model for the first time — ~1 minute...")

    df_true = pd.read_csv("True.csv");  df_true["label"] = 1
    df_fake = pd.read_csv("Fake.csv");  df_fake["label"] = 0
    df = pd.concat([df_true, df_fake], ignore_index=True)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000, ngram_range=(1, 2),
            sublinear_tf=True, stop_words="english"
        )),
        ("clf", LogisticRegression(C=5, max_iter=1000, random_state=42))
    ])
    pipeline.fit(df["content"], df["label"])

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline


@st.cache_resource
def load_bert_model():
    """Load the fine-tuned DistilBERT model."""
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    bert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_model")

    if not os.path.exists(bert_path):
        st.error("DistilBERT model not found. Run `python train_bert.py` first.")
        st.stop()

    tokenizer = DistilBertTokenizerFast.from_pretrained(bert_path)
    model     = DistilBertForSequenceClassification.from_pretrained(bert_path)
    model.eval()
    return tokenizer, model


# Prediction Functions

def predict_lr(pipeline, title: str, body: str):
    combined  = clean_text(f"{title} {body}")
    prob_real = pipeline.predict_proba([combined])[0][1]
    return prob_real, combined


def predict_bert(tokenizer, model, title: str, body: str):
    text = f"{title} " + " ".join(body.split()[:200])
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs     = torch.softmax(outputs.logits, dim=1)[0]
    prob_real = probs[1].item()
    return prob_real


# Sidebar

with st.sidebar:
    st.title("Settings")

    model_choice = st.radio(
        "Choose Model",
        ["Logistic Regression (fast)", "DistilBERT (accurate)"],
        help="LR is instant. BERT takes ~3 seconds but understands context better."
    )

    threshold = st.slider(
        "Classification threshold",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="P(Real) ≥ threshold → classified as Real"
    )

    show_top_words = st.checkbox(
        "Show top TF-IDF terms",
        value=True,
        disabled="DistilBERT" in model_choice,
        help="Only available with Logistic Regression"
    )

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "Trained on the [Kaggle Fake News Dataset]"
        "(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). "
        "For educational purposes only."
    )
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.code("Python · Scikit-learn\nTF-IDF · Logistic Regression\nDistilBERT · Streamlit", language="text")


# Main Header

st.title("Fake News Detector")
st.markdown("Paste a news article or drop in a URL to check its credibility using NLP & ML.")

if "DistilBERT" in model_choice:
    st.info("Using DistilBERT — fine-tuned transformer model")
else:
    st.info("Using Logistic Regression + TF-IDF — fast classical ML")


# Input Form

with st.form("prediction_form"):

    # URL input
    url = st.text_input(
        "Article URL",
        placeholder="https://www.bbc.com/news/... (optional)"
    )

    st.markdown("**— or enter manually —**")

    title = st.text_input("Article Title", placeholder="Enter the headline...")
    body  = st.text_area("Article Body",   placeholder="Paste the full article text here...", height=200)
    submitted = st.form_submit_button("Analyze Article", use_container_width=True)


# Prediction Logic

if submitted:

    # Step 1: scrape URL if provided
    if url.strip():
        with st.spinner("Fetching article from URL..."):
            scraped_title, scraped_body = scrape_article(url.strip())

        if scraped_title or scraped_body:
            # URL takes priority — overrides manual input
            title = scraped_title or title
            body  = scraped_body  or body
            st.success(f"Scraped: **{scraped_title}**")
        else:
            st.error("Could not fetch the article. The site may be paywalled or blocking scrapers. Try pasting the text manually.")
            st.stop()

    # Step 2: validate we have something
    if not title.strip() and not body.strip():
        st.error("Please enter a URL, title, or article body to analyze.")
        st.stop()

    # Step 3: run prediction
    with st.spinner("Analyzing..."):
        if "DistilBERT" in model_choice:
            tokenizer, bert_model = load_bert_model()
            prob_real = predict_bert(tokenizer, bert_model, title, body)
            combined  = None
        else:
            pipeline  = load_lr_model()
            prob_real, combined = predict_lr(pipeline, title, body)

    prob_fake = 1.0 - prob_real
    is_real   = prob_real >= threshold

    # Result banner
    if is_real:
        st.markdown(
            f"<div class='result-box real-news'>"
            f"Likely REAL NEWS &nbsp;|&nbsp; Confidence: {prob_real*100:.1f}%"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-box fake-news'>"
            f"Likely FAKE NEWS &nbsp;|&nbsp; Confidence: {prob_fake*100:.1f}%"
            f"</div>",
            unsafe_allow_html=True
        )

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("P(Real)", f"{prob_real*100:.1f}%")
    col2.metric("P(Fake)", f"{prob_fake*100:.1f}%")
    col3.metric("Threshold", f"{threshold:.2f}")

    # Probability bar chart
    st.markdown("#### Confidence Score")

    confidence = prob_real if is_real else prob_fake
    label      = "Real" if is_real else "Fake"

    st.markdown(f"""
    <div style="background:#e9ecef; border-radius:10px; height:28px; width:100%; margin-bottom:8px;">
        <div style="
            background: {'#28a745' if is_real else '#dc3545'};
            width: {confidence*100:.1f}%;
            height: 100%;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.9rem;
        ">
            {confidence*100:.1f}% {label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top TF-IDF terms (LR only)
    if show_top_words and combined and "DistilBERT" not in model_choice:
        st.markdown("#### Most Influential Terms (TF-IDF)")
        try:
            vectorizer    = pipeline.named_steps["tfidf"]
            feature_names = np.array(vectorizer.get_feature_names_out())
            tfidf_vector  = vectorizer.transform([combined]).toarray()[0]
            top_indices   = np.argsort(tfidf_vector)[-15:][::-1]

            top_df = pd.DataFrame({
                "Term":         feature_names[top_indices],
                "TF-IDF Score": tfidf_vector[top_indices].round(4)
            })
            st.dataframe(top_df.set_index("Term"), use_container_width=True)
        except Exception:
            st.info("Could not extract TF-IDF terms.")

    st.markdown("---")
    st.caption("This tool is for educational purposes. Always verify news from multiple reputable sources.")


# Batch Prediction

st.markdown("---")
with st.expander("Batch Prediction — Upload a CSV"):
    st.markdown("Upload a CSV with `title` and/or `text` columns to classify multiple articles at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df):,} rows**. Preview:")
        st.dataframe(df.head(5))

        if st.button("Run Batch Prediction"):
            title_col = "title" if "title" in df.columns else None
            text_col  = "text"  if "text"  in df.columns else None

            if not title_col and not text_col:
                st.error("CSV must have a `title` or `text` column.")
            else:
                titles = df[title_col].fillna("") if title_col else pd.Series([""] * len(df))
                bodies = df[text_col].fillna("")  if text_col  else pd.Series([""] * len(df))

                with st.spinner("Running predictions..."):
                    results = []

                    if "DistilBERT" in model_choice:
                        tokenizer, bert_model = load_bert_model()
                        for t, b in zip(titles, bodies):
                            p_real = predict_bert(tokenizer, bert_model, t, b)
                            results.append({
                                "Prediction": "Real" if p_real >= threshold else "Fake",
                                "P(Real)":    round(p_real, 4),
                                "P(Fake)":    round(1 - p_real, 4),
                            })
                    else:
                        pipeline = load_lr_model()
                        for t, b in zip(titles, bodies):
                            p_real, _ = predict_lr(pipeline, t, b)
                            results.append({
                                "Prediction": "Real" if p_real >= threshold else "Fake",
                                "P(Real)":    round(p_real, 4),
                                "P(Fake)":    round(1 - p_real, 4),
                            })

                result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
                st.success(f"Done! Classified {len(result_df):,} articles.")
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode()
                st.download_button(
                    "Download Results CSV",
                    data=csv_bytes,
                    file_name="fake_news_predictions.csv",
                    mime="text/csv",
                )