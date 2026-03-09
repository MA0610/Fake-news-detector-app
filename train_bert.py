import pandas as pd
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Device setup
# Uses GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data 
print("Loading data...")
df_true = pd.read_csv("True.csv"); df_true["label"] = 1
df_fake = pd.read_csv("Fake.csv"); df_fake["label"] = 0
df = pd.concat([df_true, df_fake], ignore_index=True)

# Combine title + first 200 words of body
# BERT has a 512 token limit, truncate to stay under it
df["content"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("").apply(lambda x: " ".join(x.split()[:200]))
)

X_train, X_test, y_train, y_test = train_test_split(
    df["content"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        # Tokenize all texts at once
        self.encodings = tokenizer(
            texts,
            truncation=True,      # Cut off at 512 tokens
            padding=True,         # Pad shorter texts to same length
            max_length=512,
            return_tensors="pt"   # Return PyTorch tensors
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

train_dataset = NewsDataset(X_train, y_train)
test_dataset  = NewsDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16)

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2    # Fake or Real
)
model.to(device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)

# Training loop
print("\nFine-tuning DistilBERT...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_num, batch in enumerate(train_loader):
        # Move batch to GPU/CPU
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Forward pass — run the model
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss

        # Backward pass — update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_num % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_num}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}")

# Evaluation
print("\nEvaluating...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():    # Don't calculate gradients during evaluation
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy: {acc*100:.2f}%")
print(classification_report(all_labels, all_preds, target_names=["Fake","Real"]))

# Save
os.makedirs("bert_model", exist_ok=True)
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")
print("\nModel saved to bert_model/")