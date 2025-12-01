# Adaptive Query Reformulation AQR

This project trains a DistilRoBERTa-based classifier to label user queries as STRONG (well-formed) or WEAK (ill-formed).
It also includes a separate inference script so you can classify any query from the command line.

---
## 💻 Demo

https://github.com/user-attachments/assets/dd8147f7-5a8b-4f4b-abc7-a0cdbffb4151

---

## 📊 Dataset

This project uses the Google Query Wellformedness (QWF) dataset:

Dataset link:  
https://github.com/google-research-datasets/query-wellformedness

Download and extract the ZIP. You must place:

```
train.tsv
dev.tsv
test.tsv
README.md
```

inside a folder named exactly:

```
query-wellformedness-master
```

The folder name must match exactly, otherwise the training script will not find the dataset.

---

## 📁 Project Folder Structure

Your project directory should look like this:

```
project/
│
├── infer.py
├── qw_strong_weak_classifier.py
│
├── query-wellformedness-master/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   └── README.md
│
└── output/           # automatically created after training
```

---

## 🏋️‍♂️ Training the Model

Run:

```bash
python3 qw_strong_weak_classifier.py
```

This script will:

- Load & preprocess the QWF dataset  
- Fine-tune DistilRoBERTa  
- Apply class balancing  
- Tune a probability threshold for identifying weak queries  
- Evaluate performance on dev/test sets  
- Save the best-performing model inside:

```
output/distilroberta/
```

---

## 🔍 Running Inference

After training, classify any query using:

```bash
python3 infer.py --text "weather tomorrow"
```

Example output:

```
=== Query Classification ===
Text      : weather tomorrow
Prediction: WEAK
Weak prob : 0.9825
Threshold : 0.36
============================
```

---

## 📦 Files Included

| File | Description |
|------|-------------|
| qw_strong_weak_classifier.py | Training pipeline |
| infer.py | Inference script |
| query-wellformedness-master/ | Required dataset folder |
| output/ | Contains saved model (after training) |

---

## Project Flow
<img width="1786" height="824" alt="image" src="https://github.com/user-attachments/assets/f10c8f61-ffc0-4445-b342-6bff3fdeeb34" />

---

## 📝 Notes

- The model `.safetensors` file is large — do NOT commit `output/` to GitHub.
- Add `output/` and `query-wellformedness-master/` to `.gitignore`.
