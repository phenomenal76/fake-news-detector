"""Prediction + explanation. `label 1 = fake / misleading`, `0 = real / reliable`."""
from pathlib import Path

import joblib
import numpy as np

from .heuristics import red_flags

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "models" / "baseline.joblib"
BERT_DIR = ROOT / "models" / "bert"


def _verdict(fake_prob: float) -> str:
    return "Potentially misleading" if fake_prob >= 0.5 else "Looks reliable"


class BaselineDetector:
    """TF-IDF + Logistic Regression. Explains itself by listing the most influential words."""

    def __init__(self, path=BASELINE_PATH):
        self.pipe = joblib.load(path)
        self.vec = self.pipe.named_steps["tfidf"]
        self.clf = self.pipe.named_steps["clf"]
        self.vocab = np.array(self.vec.get_feature_names_out())

    def predict(self, text: str, top_k: int = 6) -> dict:
        X = self.vec.transform([text])
        fake_prob = float(self.clf.predict_proba(X)[0, 1])

        coo = X.tocoo()
        contrib = coo.data * self.clf.coef_[0][coo.col]  # tf-idf value x model weight
        words = self.vocab[coo.col]
        order = np.argsort(contrib)
        toward_real = [(str(words[i]), float(contrib[i])) for i in order[:top_k] if contrib[i] < 0]
        toward_fake = [(str(words[i]), float(contrib[i])) for i in order[::-1][:top_k] if contrib[i] > 0]

        return {
            "model": "TF-IDF + Logistic Regression",
            "verdict": _verdict(fake_prob),
            "fake_prob": fake_prob,
            "confidence": max(fake_prob, 1 - fake_prob),
            "words_pointing_to_fake": toward_fake,
            "words_pointing_to_real": toward_real,
            "red_flags": red_flags(text),
        }


class BertDetector:
    """Fine-tuned BERT / DistilBERT. Needs torch + transformers and a trained model in models/bert."""

    def __init__(self, path=BERT_DIR):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()

    def predict(self, text: str) -> dict:
        enc = self.tok(text, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[0].cpu().numpy()
        fake_prob = float(probs[1])
        return {
            "model": "BERT (fine-tuned)",
            "verdict": _verdict(fake_prob),
            "fake_prob": fake_prob,
            "confidence": max(fake_prob, 1 - fake_prob),
            "red_flags": red_flags(text),
        }


def bert_available() -> bool:
    return (BERT_DIR / "config.json").exists()
