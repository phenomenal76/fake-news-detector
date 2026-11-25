# 📰 Fake News & Misinformation Detector

A small NLP project that reads a news headline or article and predicts whether it is **potentially misleading**.
It shows a **confidence score** and **supporting evidence** (which words pushed the decision + writing-style red flags).

> Think of it like a spell-checker, but for "does this sound like a real news report or a rumour?".
> It judges *how something is written*, not whether the facts are true. It is a helper, not a fact-checker.

## Features
- Analyse headlines and full articles
- Predicts misleading content (label: fake / real)
- Shows confidence + supporting evidence
  - Top words pushing towards **fake** or **real** (TF-IDF + Logistic Regression)
  - Red flags: ALL-CAPS, many `!!!`, sensational phrases, no source mentioned
- Two models you can compare:
  | Model | Idea | Speed |
  |---|---|---|
  | TF-IDF + Logistic Regression | Counts important words, then draws a line between "fake-sounding" and "real-sounding" | Seconds on a laptop |
  | BERT (DistilBERT fine-tuned) | Reads the whole sentence in context, like a person would | Needs a GPU (Colab is fine) |
- Simple web app built with Streamlit

## Project structure
```
fake-news-detector/
├── app.py                 # Streamlit web app
├── make_sample_data.py    # creates the toy dataset
├── requirements.txt
├── data/                  # sample_news.csv (+ put Fake.csv / True.csv here)
├── models/                # trained models are saved here (not pushed to git)
├── src/
│   ├── preprocess.py      # text cleaning
│   ├── data_loader.py     # loads Kaggle folder or any CSV
│   ├── heuristics.py      # red-flag checks
│   ├── train_baseline.py  # TF-IDF + Logistic Regression
│   ├── train_bert.py      # fine-tune BERT / DistilBERT
│   └── predict.py         # prediction + explanation
└── tests/test_pipeline.py
```

## Setup
```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
(Only want the baseline? You can skip `torch` and `transformers` in `requirements.txt`.)

## Run it
```bash
# 1. Train the baseline (toy data - works instantly)
python -m src.train_baseline

# 2. Or train on the real dataset (put Fake.csv + True.csv in data/)
python -m src.train_baseline --data data/

# 3. (Optional) Fine-tune BERT - use a GPU / Google Colab
python -m src.train_bert --data data/ --epochs 2
#    quick test on a slice:  python -m src.train_bert --data data/ --limit 4000

# 4. Launch the app
streamlit run app.py
```
If you skip step 1, the app trains a quick demo model on the sample data by itself.
The BERT option appears in the app automatically once `models/bert/` exists.

## How it works
1. **Clean** the text (lowercase, remove links/HTML/agency tags like "(Reuters)").
2. **TF-IDF** turns text into numbers: words that are special to a document get a high score, very common words get a low one.
3. **Logistic Regression** learns a weight per word. Positive weight = sounds fake, negative = sounds real.
4. **Evidence** = word score x weight, so we can list exactly which words drove the answer.
5. **BERT** is a pre-trained language model fine-tuned on the same data for context-aware predictions.

## Results
Fill this in after training on the full dataset:

| Model | Accuracy | F1 |
|---|---|---|
| TF-IDF + Logistic Regression | _ | _ |
| DistilBERT | _ | _ |

(The toy sample data gives meaningless scores - 48 rows is too small.)

## Limitations
- Detects *writing style*, not truth. A well-written lie can pass; a badly written true story can be flagged.
- Datasets like ISOT have patterns (e.g. source tags) that make scores look unrealistically high. We strip some, but test on other data too.
- English only.

## Ideas to extend
- Check claims against a fact-check API
- Add LIME/SHAP explanations for BERT
- Browser extension or an NGO-friendly web front end

## Tests
```bash
pytest -q
```

## License
MIT
