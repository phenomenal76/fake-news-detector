"""Load a dataset into a DataFrame with columns: content (text), label (1 = fake, 0 = real).

Supported inputs
1. A folder containing Fake.csv and True.csv (Kaggle "Fake and Real News" / ISOT dataset)
2. A single CSV with a `text` column (optional `title`) and a `label` column (1 = fake, 0 = real)
"""
from pathlib import Path

import pandas as pd

DEFAULT_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_news.csv"


def load_dataset(path=None) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_SAMPLE

    if path.is_dir():
        fake_p, true_p = path / "Fake.csv", path / "True.csv"
        if not (fake_p.exists() and true_p.exists()):
            raise FileNotFoundError(f"Expected Fake.csv and True.csv inside {path}")
        fake, true = pd.read_csv(fake_p), pd.read_csv(true_p)
        fake["label"], true["label"] = 1, 0
        df = pd.concat([fake, true], ignore_index=True)
    else:
        df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError("Dataset needs a `label` column (1 = fake, 0 = real)")

    title = df["title"].fillna("") if "title" in df.columns else ""
    text = df["text"].fillna("") if "text" in df.columns else ""
    if isinstance(title, str):
        df["content"] = text
    else:
        df["content"] = title + ". " + text
    df = df[["content", "label"]].dropna()
    df = df[df["content"].str.strip().str.len() > 5]
    df = df.drop_duplicates(subset="content").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df
