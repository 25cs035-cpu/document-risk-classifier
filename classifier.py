"""Prediction, confidence threshold and explainability helpers."""
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from extract import clean_text

MODEL_PATH = "models/model.joblib"
DEFAULT_THRESHOLD = 0.6


def load_bundle(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)


def classify(text: str, bundle: dict, threshold: float = DEFAULT_THRESHOLD,
             top_terms: int = 8, top_docs: int = 3) -> dict:
    vec, clf = bundle["vectorizer"], bundle["clf"]
    X = vec.transform([clean_text(text)])

    proba = clf.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label = str(clf.classes_[idx])
    confidence = float(proba[idx])
    status = "Auto-classified" if confidence >= threshold else "Manual Review"

    # Explainability 1: words that pushed the prediction toward the predicted class
    contrib = X.toarray()[0] * clf.coef_[idx]
    names = vec.get_feature_names_out()
    order = np.argsort(contrib)[::-1][:top_terms]
    terms = [(names[i], round(float(contrib[i]), 3)) for i in order if contrib[i] > 0]

    # Explainability 2: nearest training documents
    sims = cosine_similarity(X, bundle["X_train"])[0]
    nearest = np.argsort(sims)[::-1][:top_docs]
    similar = [
        {
            "similarity": round(float(sims[i]), 3),
            "label": bundle["train_labels"][i],
            "text": bundle["train_texts"][i][:250] + "...",
        }
        for i in nearest
    ]

    return {
        "label": label,
        "confidence": confidence,
        "status": status,
        "probabilities": {str(c): float(p) for c, p in zip(clf.classes_, proba)},
        "top_terms": terms,
        "similar_docs": similar,
    }
