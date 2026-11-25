from src.heuristics import red_flags
from src.preprocess import clean_text
from src.predict import BaselineDetector
from src.train_baseline import train


def test_clean_text():
    assert clean_text("WASHINGTON (Reuters) - Hello, WORLD! http://x.com") == "washington hello world"


def test_red_flags():
    assert red_flags("SHOCKING!! You won't believe this SECRET!!")


def test_train_and_predict(tmp_path):
    out = tmp_path / "m.joblib"
    train(out=out, quiet=True)
    det = BaselineDetector(out)
    res = det.predict("SHOCKING secret miracle cure exposed!!! Share before it is banned!!!")
    assert 0.0 <= res["fake_prob"] <= 1.0
    assert "red_flags" in res
