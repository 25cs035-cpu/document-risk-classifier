"""Streamlit UI.  Run:  streamlit run app.py"""
import datetime
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from classifier import DEFAULT_THRESHOLD, MODEL_PATH, classify, load_bundle
from extract import clean_text, extract_text

LOG_PATH = "data/uploads_log.csv"

st.set_page_config(page_title="Document Risk & Compliance Classifier", layout="wide")
st.title("AI Document Risk & Compliance Classifier")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python generate_data.py` and `python train.py` first.")
    st.stop()


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()

page = st.sidebar.radio("Page", ["Classify a document", "Dashboard"])
threshold = st.sidebar.slider(
    "Confidence threshold", 0.3, 0.95, DEFAULT_THRESHOLD, 0.05,
    help="Predictions below this confidence are sent to manual review.",
)

# ---------------------------------------------------------------- Classify
if page == "Classify a document":
    uploaded = st.file_uploader("Upload a PDF, DOCX or TXT file", type=["pdf", "docx", "txt"])
    pasted = st.text_area("...or paste document text", height=150)

    if st.button("Classify", type="primary"):
        text, name = "", "pasted text"
        if uploaded is not None:
            text, name = extract_text(uploaded, uploaded.name), uploaded.name
        elif pasted.strip():
            text = pasted

        if not clean_text(text):
            st.warning("No readable text found. Upload a text-based file or paste some text.")
            st.stop()

        res = classify(text, bundle, threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted risk", res["label"])
        c2.metric("Confidence", f"{res['confidence']:.0%}")
        c3.metric("Status", res["status"])
        if res["status"] == "Manual Review":
            st.warning("Confidence is below the threshold, so this document needs manual review.")
        else:
            st.success("Confidence is above the threshold, so the label was assigned automatically.")

        st.subheader("Class probabilities")
        st.bar_chart(pd.Series(res["probabilities"]))

        left, right = st.columns(2)
        with left:
            st.subheader("Why? Top contributing terms")
            if res["top_terms"]:
                st.dataframe(pd.DataFrame(res["top_terms"], columns=["term", "contribution"]),
                             hide_index=True, use_container_width=True)
            else:
                st.write("No strongly contributing terms found.")
        with right:
            st.subheader("Most similar training documents")
            for d in res["similar_docs"]:
                st.markdown(f"**{d['label']}** (similarity {d['similarity']})")
                st.caption(d["text"])

        os.makedirs("data", exist_ok=True)
        row = pd.DataFrame([{
            "document": name,
            "snippet": clean_text(text)[:160],
            "true_label": "",
            "predicted": res["label"],
            "confidence": round(res["confidence"], 3),
            "source": "uploaded " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }])
        row.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

# --------------------------------------------------------------- Dashboard
else:
    frames = []
    if os.path.exists("data/predictions.csv"):
        frames.append(pd.read_csv("data/predictions.csv"))
    if os.path.exists(LOG_PATH):
        frames.append(pd.read_csv(LOG_PATH))
    if not frames:
        st.info("No documents processed yet.")
        st.stop()

    df = pd.concat(frames, ignore_index=True)
    df["status"] = np.where(df["confidence"] >= threshold, "Auto-classified", "Manual Review")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", len(df))
    m2.metric("Flagged for manual review", int((df["status"] == "Manual Review").sum()))
    m3.metric("High-risk (predicted)", int((df["predicted"] == "High").sum()))
    m4.metric("Avg confidence", f"{df['confidence'].mean():.0%}")

    st.subheader("Documents by predicted risk")
    chart = df.groupby(["predicted", "status"]).size().unstack(fill_value=0)
    st.bar_chart(chart)

    st.subheader("Document table")
    f1, f2 = st.columns(2)
    risk_filter = f1.multiselect("Risk level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    status_filter = f2.radio("Show", ["All", "Manual Review only", "Auto-classified only"], horizontal=True)
    view = df[df["predicted"].isin(risk_filter)]
    if status_filter == "Manual Review only":
        view = view[view["status"] == "Manual Review"]
    elif status_filter == "Auto-classified only":
        view = view[view["status"] == "Auto-classified"]
    st.dataframe(view.sort_values("confidence"), hide_index=True, use_container_width=True)

    st.subheader("Model performance (held-out test set)")
    if os.path.exists("reports/metrics.json"):
        with open("reports/metrics.json") as f:
            metrics = json.load(f)
        p1, p2 = st.columns(2)
        p1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        p2.metric("Accuracy on auto-classified docs", f"{metrics['accuracy_on_auto']:.1%}")
        rep = pd.DataFrame(metrics["classification_report"]).T.round(3)
        st.dataframe(rep, use_container_width=True)
    if os.path.exists("reports/confusion_matrix.png"):
        st.image("reports/confusion_matrix.png", width=450)
    if os.path.exists("reports/threshold_sweep.csv"):
        st.caption("Threshold trade-off: higher threshold = more manual review, higher accuracy on the rest.")
        st.dataframe(pd.read_csv("reports/threshold_sweep.csv"), hide_index=True)
