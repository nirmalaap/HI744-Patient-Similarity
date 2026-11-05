"""Clinical text cleaning and normalization."""
import re
import unicodedata

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import wordpunct_tokenize
    from nltk import download
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
    try:
        LEMMATIZER.lemmatize("records")
    except LookupError:
        download("wordnet", quiet=True)
        download("omw-1.4", quiet=True)
except Exception:
    STOP_WORDS = set()
    LEMMATIZER = None
    wordpunct_tokenize = None

# Scikit-learn provides an offline stopword list if the NLTK corpus is unavailable.
if not STOP_WORDS:
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOP_WORDS = set(ENGLISH_STOP_WORDS)
    except Exception:
        STOP_WORDS = set()



def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """Normalize one clinical note while preserving clinically meaningful words."""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = text.replace("\\n", " ")
    # Keep letters and digits; replace punctuation/symbols with spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()

    # Do not remove negation terms: they can change clinical meaning.
    protected_negations = {"no", "not", "nor", "without", "denies", "negative"}
    if remove_stopwords:
        tokens = [
            token for token in tokens
            if token not in STOP_WORDS or token in protected_negations
        ]

    if lemmatize and LEMMATIZER is not None:
        tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
    elif lemmatize:
        # Offline fallback, used only when the NLTK WordNet corpus is unavailable.
        # This is a light suffix normalization, not a full linguistic lemmatizer.
        def fallback_lemma(token):
            if len(token) > 5 and token.endswith("ies"):
                return token[:-3] + "y"
            if len(token) > 5 and token.endswith("ing"):
                return token[:-3]
            if len(token) > 4 and token.endswith("ed"):
                return token[:-2]
            if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
                return token[:-1]
            return token
        tokens = [fallback_lemma(token) for token in tokens]

    return " ".join(tokens)


def preprocess_records(records, include_title=True):
    """Create one processed text string per patient record."""
    output = []
    for record in records:
        parts = []
        if include_title:
            parts.append(record.get("title", ""))
        parts.append(record.get("patient", ""))
        output.append(preprocess_text(" ".join(parts)))
    return output
