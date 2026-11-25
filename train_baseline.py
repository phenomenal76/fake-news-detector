"""Train the TF-IDF + Logistic Regression baseline.

Usage:
    python -m src.train_baseline                      # uses data/sample_news.csv (toy data)
    python -m src.train_baseline --data data/         # folder with Fake.csv and True.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_loader import load_dataset
from .preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=2.0)),
    ])


def train(data=None, out=None, test_size=0.2, seed=42, quiet=False):
    out = Path(out) if out else ROOT / "models" / "baseline.joblib"
    df = load_dataset(data)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["content"], df["label"], test_size=test_size, random_state=seed, stratify=df["label"]
    )
    pipe = build_pipeline().fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    metrics = {
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
    }
    if not quiet:
        print(classification_report(y_te, pred, target_names=["real", "fake"]))
        print(metrics)

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    (out.parent / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["real", "fake"], cmap="Blues")
    plt.title("TF-IDF + Logistic Regression")
    plt.savefig(out.parent / "baseline_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    return pipe, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV file or folder with Fake.csv + True.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    train(a.data, a.out)
