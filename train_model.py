import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import pickle


# Load the data
def load_data():
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')

    df_true['label'] = 1
    df_fake['label'] = 0

    df = pd.concat([df_true, df_fake], ignore_index=True)
    print(f"Loaded {len(df_true):,} real and {len(df_fake):,} fake articles")
    return df

# Clean the text data
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)        # keep only letters
    text = re.sub(r"\s+", " ", text).strip()     # collapse spaces
    return text

def prepare_features(df):
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["text"]  = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]
    df["content"] = df["content"].apply(clean_text)
    return df["content"]

# Pipeline
def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )),
        ("clf", model)
    ])


# Train Model & Evaluate

def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Fake","Real"]))

def main():
    df = load_data()

    X = prepare_features(df)   # cleaned text (input)
    y = df["label"]             # 0 or 1 (target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(C=5, max_iter=1000, random_state=42),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
    }

    best_acc, best_pipeline = 0, None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        evaluate(name, pipeline, X_test, y_test)
        if acc > best_acc:
            best_acc, best_pipeline = acc, pipeline

    # Save best model
    with open("model/fake_news_pipeline.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"\nBest model saved (accuracy: {best_acc*100:.2f}%)")


if __name__ == "__main__":
    main()