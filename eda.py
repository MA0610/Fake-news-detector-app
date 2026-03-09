import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# Load the data
os.makedirs("eda_outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

df_true = pd.read_csv("True.csv"); df_true["label"] = 1
df_fake = pd.read_csv("Fake.csv"); df_fake["label"] = 0
df = pd.concat([df_true, df_fake], ignore_index=True)

df["word_count"] = df["text"].fillna("").apply(lambda x: len(x.split()))
df["text_len"]   = df["text"].fillna("").apply(len)
df["title_len"]  = df["title"].fillna("").apply(len)
df["label_name"] = df["label"].map({1: "Real", 0: "Fake"})

print(df.groupby("label_name")[["word_count", "text_len"]].describe())



# Chart 1 Class Distribution
fig, ax = plt.subplots(figsize=(6, 4))

counts = df["label_name"].value_counts()
ax.bar(counts.index, counts.values,
       color=["#28a745", "#dc3545"], width=0.5)

ax.set_title("Class Distribution: Real vs Fake")
ax.set_ylabel("Article Count")
plt.tight_layout()
plt.savefig("eda_outputs/class_distribution.png")
plt.show()

#Chart 2 Article Length by Class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col, title in zip(axes,
                           ["word_count", "text_len"],
                           ["Word Count", "Character Count"]):
    for name, color in [("Real", "#28a745"), ("Fake", "#dc3545")]:
        subset = df[df["label_name"] == name][col]
        ax.hist(subset.clip(upper=subset.quantile(0.99)),
                bins=60, alpha=0.6, color=color, label=name)
    ax.set_title(f"{title} Distribution")
    ax.set_xlabel(title)
    ax.legend()

plt.tight_layout()
plt.savefig("eda_outputs/article_lengths.png")
plt.show()

# Chart 3 Most Common Words
STOPWORDS = {
    "the","and","a","to","of","in","is","that","it","for","on",
    "are","was","with","this","at","be","by","as","have","said",
    "he","she","they","we","you","or","an","but","from","not"
}

def top_words(series, n=20):
    words = []
    for text in series.fillna(""):
        for word in re.findall(r"[a-z]{3,}", text.lower()):
            if word not in STOPWORDS:
                words.append(word)
    return Counter(words).most_common(n)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, name, color in zip(axes, ["Real", "Fake"], ["#28a745", "#dc3545"]):
    subset = df[df["label_name"] == name]["text"]
    words, counts = zip(*top_words(subset))
    ax.barh(list(words)[::-1], list(counts)[::-1], color=color)
    ax.set_title(f"Top 20 Words — {name} News")
    ax.set_xlabel("Frequency")

plt.tight_layout()
plt.savefig("eda_outputs/top_words.png")
plt.show()

# Chart 4 Correlation Heatmap
fig, ax = plt.subplots(figsize=(5, 4))

numeric = df[["text_len", "word_count", "title_len", "label"]]
sns.heatmap(numeric.corr(), annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1, square=True, ax=ax)

ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.show()