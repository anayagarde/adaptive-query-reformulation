#!/usr/bin/env python3
import argparse
import json
import torch
from pathlib import Path
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUTDIR = Path("output/distilroberta")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(OUTDIR)
    model = AutoModelForSequenceClassification.from_pretrained(OUTDIR).to(DEVICE)
    model.eval()

    # load threshold
    thr_path = OUTDIR / "routing_threshold.json"
    if thr_path.exists():
        threshold = json.load(open(thr_path))["weak_threshold"]
    else:
        threshold = 0.5

    return tokenizer, model, threshold

def classify(text, tokenizer, model, threshold, max_len=64):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    p_weak = float(probs[0])
    label = "WEAK" if p_weak >= threshold else "STRONG"

    return {
        "input": text,
        "label": label,
        "p_weak": p_weak,
        "threshold": threshold
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    tokenizer, model, threshold = load_model()
    out = classify(args.text, tokenizer, model, threshold)

    print("\n=== Query Classification ===")
    print("Text      :", out["input"])
    print("Prediction:", out["label"])
    print("Weak prob :", out["p_weak"])
    print("Threshold :", out["threshold"])
    print("============================\n")
