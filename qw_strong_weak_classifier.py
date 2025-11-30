#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strong vs Weak Query Classifier
Fine-tunes DistilRoBERTa + trains a TF-IDF+SVM baseline.
"""

# ------------ Imports ------------
import json
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    confusion_matrix,
    accuracy_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from transformers import get_linear_schedule_with_warmup

print("Loaded imports successfully")

# ------------ Config ------------
DATA_DIR = "query-wellformedness-master"
TRAIN_PATH = f"{DATA_DIR}/train.tsv"
DEV_PATH   = f"{DATA_DIR}/dev.tsv"
TEST_PATH  = f"{DATA_DIR}/test.tsv"

OUTDIR = Path("output/distilroberta")
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilroberta-base"
MAX_LEN = 64
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 32
SEED = 13
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------ Dataset Loader ------------
def load_split(path: Path, threshold: float = 0.6) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, engine="python")

    if df.shape[1] < 2:
        raise ValueError(f"Expected text + score columns. Got: {df.shape}")

    # Handle multiple text columns due to stray tabs
    text = df.iloc[:, :-1].agg(" ".join, axis=1)
    score = pd.to_numeric(df.iloc[:, -1].str.strip(), errors="coerce")

    labels = (score >= threshold).astype(int)
    return pd.DataFrame({"text": text.astype(str), "label": labels})

# ------------ Load Dataset ------------
print("Loading dataset...")
train_df = load_split(TRAIN_PATH)
dev_df   = load_split(DEV_PATH)
test_df  = load_split(TEST_PATH)

print(train_df.head())
print("Label counts:", train_df["label"].value_counts())

# ------------ Tokenizer & Model ------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ------------ Dataset Class ------------
@dataclass
class QWDataset(torch.utils.data.Dataset):
    encodings: dict
    labels: list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ------------ Encode dataset ------------
def encode(df):
    enc = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    return QWDataset(encodings=enc, labels=df["label"].tolist())

ds_train = encode(train_df)
ds_dev   = encode(dev_df)

# ------------ Class Weights ------------
w = torch.tensor(
    compute_class_weight(
        class_weight="balanced",
        classes=np.array([0,1]),
        y=train_df["label"].values
    ),
    dtype=torch.float
)

# ------------ Metrics Function ------------
def compute_metrics(eval_tuple):
    logits, labels = eval_tuple
    preds = logits.argmax(-1)
    from sklearn.metrics import f1_score
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

# ------------ Training Setup ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = torch.utils.data.DataLoader(
    ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator
)
dev_loader = torch.utils.data.DataLoader(
    ds_dev, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
num_training_steps = EPOCHS * len(train_loader)
num_warmup = int(0.06 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup, num_training_steps
)

criterion = torch.nn.CrossEntropyLoss(weight=w.to(device))

# ------------ Manual Train Loop ------------
def evaluate_manual():
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dev_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out.logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return compute_metrics((logits, labels))

best_f1 = -1
print("Starting training...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0

    for step, batch in enumerate(train_loader, 1):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(**batch)
        loss = criterion(out.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total += loss.item()

    metrics = evaluate_manual()
    print(f"Epoch {epoch} →", metrics)

    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        model.save_pretrained(OUTDIR)
        tokenizer.save_pretrained(OUTDIR)
        print("Saved new best model.")

print("Finished training. Best F1 =", best_f1)

# ------------ Inference Utilities ------------
def batched_probs(model, tokenizer, texts, max_len=64, batch_size=32):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i:i+batch_size],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_len
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(out)

# ------------ Threshold Tuning ------------
dev_probs = batched_probs(model, tokenizer, dev_df["text"].tolist())
p_weak = dev_probs[:, 0]
y_dev = dev_df["label"].values

best_thr, best_score = 0.5, -1

for t in np.linspace(0.05, 0.95, 181):
    pred = (p_weak < t).astype(int)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_dev, pred, average="macro")
    if f1 > best_score:
        best_score = f1
        best_thr = t

print("Chosen weak-threshold:", best_thr)

json.dump({"weak_threshold": float(best_thr)}, open(OUTDIR / "routing_threshold.json", "w"), indent=2)

# ------------ Baseline TF-IDF + SVM ------------
Xtr = train_df["text"].tolist()
ytr = train_df["label"].values
Xdv = dev_df["text"].tolist()
ydv = dev_df["label"].values

tfidf_svm = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3,5), min_df=2)),
    ("svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5))
])

tfidf_svm.fit(Xtr, ytr)

print("TF-IDF baseline evaluation:")
preds = tfidf_svm.predict(Xdv)
print(classification_report(ydv, preds))

BL_OUT = OUTDIR.parent / "tfidf_svm"
BL_OUT.mkdir(parents=True, exist_ok=True)
joblib.dump(tfidf_svm, BL_OUT/"model.joblib")

print("Saved baseline model.")

# ------------ Transformer Test Evaluation ------------
Xte = test_df["text"].tolist()
yte = test_df["label"].values

probs_test = batched_probs(model, tokenizer, Xte)
y_pred = probs_test.argmax(1)

print("\nTransformer Test Results:")
print(classification_report(yte, y_pred, target_names=["WEAK","STRONG"]))

cm = confusion_matrix(yte, y_pred)
print("Confusion Matrix:\n", cm)

print("\nDONE ✓")
