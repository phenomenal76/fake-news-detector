"""Text cleaning shared by training and prediction."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<.*?>")
# Agency tags like "(Reuters)" leak the label in some datasets, so we remove them.
_AGENCY = re.compile(r"\((reuters|ap|afp)\)", re.IGNORECASE)


def clean_text(text) -> str:
    """Lower-case, strip URLs / HTML / agency tags / symbols."""
    text = str(text)
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _AGENCY.sub(" ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()
