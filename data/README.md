# Data

`sample_news.csv` is a tiny **toy** dataset (48 hand-written headlines) so the project runs straight after cloning.
It is NOT enough for real accuracy.

## Use the real dataset (recommended)
1. Download **"Fake and Real News Dataset"** (ISOT) from Kaggle.
2. Put `Fake.csv` and `True.csv` inside this `data/` folder.
3. Train with `python -m src.train_baseline --data data/`

Those two big files are in `.gitignore`, so they will not be pushed to GitHub.

## Your own CSV
Columns: `text` (and optionally `title`) + `label` where `1 = fake`, `0 = real`.
