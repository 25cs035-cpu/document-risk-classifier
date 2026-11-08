"""Train TF-IDF + Logistic Regression, evaluate, and save everything the app needs.

Run:  python train.py
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split

from classifier import DEFAULT_THRESHOLD, MODEL_PATH
from extract import clean_text

LABELS = ["Low", "Medium", "High"]
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 1. Load + clean
df = pd.read_csv("data/labels.csv").dropna()
df["clean"] = df["text"].map(clean_text)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# 2. Features + model
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english",
                      sublinear_tf=True, min_df=2)
X_train = vec.fit_transform(train_df["clean"])
X_test = vec.transform(test_df["clean"])

clf = LogisticRegression(max_iter=1000, C=5, class_weight="balanced")
cv_scores = cross_val_score(clf, X_train, train_df["label"], cv=5)
clf.fit(X_train, train_df["label"])

# 3. Predict on held-out test set
proba = clf.predict_proba(X_test)
pred = clf.classes_[proba.argmax(axis=1)]
conf = proba.max(axis=1)
y_true = test_df["label"].values

# 4. Evaluation
report = classification_report(y_true, pred, labels=LABELS, output_dict=True, zero_division=0)
print("5-fold CV accuracy: %.3f (+/- %.3f)" % (cv_scores.mean(), cv_scores.std()))
print("Test accuracy: %.3f\n" % accuracy_score(y_true, pred))
print(classification_report(y_true, pred, labels=LABELS, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_true, pred, labels=LABELS, cmap="Blues")
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()

# 5. Confidence threshold sweep
sweep = []
for t in np.arange(0.4, 0.91, 0.05):
    mask = conf >= t
    sweep.append({
        "threshold": round(float(t), 2),
        "auto_classified_pct": round(100 * mask.mean(), 1),
        "manual_review_pct": round(100 * (1 - mask.mean()), 1),
        "accuracy_on_auto": round(float((pred[mask] == y_true[mask]).mean()), 3) if mask.any() else None,
    })
sweep_df = pd.DataFrame(sweep)
sweep_df.to_csv("reports/threshold_sweep.csv", index=False)
print("\nThreshold sweep:\n", sweep_df.to_string(index=False))

mask = conf >= DEFAULT_THRESHOLD
metrics = {
    "cv_accuracy_mean": float(cv_scores.mean()),
    "test_accuracy": float(accuracy_score(y_true, pred)),
    "threshold": DEFAULT_THRESHOLD,
    "manual_review_pct": float(100 * (1 - mask.mean())),
    "accuracy_on_auto": float((pred[mask] == y_true[mask]).mean()) if mask.any() else None,
    "classification_report": report,
}
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 6. Predictions table for the dashboard
pd.DataFrame({
    "document": ["test_doc_%d" % i for i in test_df.index],
    "snippet": test_df["text"].str[:160].values,
    "true_label": y_true,
    "predicted": pred,
    "confidence": conf.round(3),
    "source": "test set",
}).to_csv("data/predictions.csv", index=False)

# 7. Save model bundle
joblib.dump({
    "vectorizer": vec,
    "clf": clf,
    "X_train": X_train,
    "train_texts": train_df["text"].tolist(),
    "train_labels": train_df["label"].tolist(),
}, MODEL_PATH)
print("\nSaved model to", MODEL_PATH)
