"""Fine-tune a BERT-style model (default: DistilBERT) for fake-news detection.

Usage:
    python -m src.train_bert --data data/ --epochs 2
    python -m src.train_bert --data data/ --limit 4000     # quick run on a small slice

A GPU is strongly recommended (Google Colab free GPU works fine).
"""
import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .data_loader import load_dataset

ROOT = Path(__file__).resolve().parents[1]


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, padding=True,
                             max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(list(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--out", default=str(ROOT / "models" / "bert"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=None, help="use only N rows (quick test)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = load_dataset(a.data)
    if a.limit:
        df = df.sample(n=min(a.limit, len(df)), random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_name, num_labels=2).to(device)

    train_dl = DataLoader(NewsDataset(X_tr, y_tr, tok, a.max_len), batch_size=a.batch_size, shuffle=True)
    test_dl = DataLoader(NewsDataset(X_te, y_te, tok, a.max_len), batch_size=a.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    steps = len(train_dl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for i, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
            if i % 50 == 0:
                print(f"epoch {epoch + 1} step {i}/{len(train_dl)} loss {total / i:.4f}")

    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for batch in test_dl:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            preds += model(**batch).logits.argmax(-1).cpu().tolist()
            truth += labels.tolist()
    print("Accuracy:", accuracy_score(truth, preds))
    print(classification_report(truth, preds, target_names=["real", "fake"]))

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("Saved to", a.out)


if __name__ == "__main__":
    main()
