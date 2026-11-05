"""TF-IDF document representation."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(texts, max_features=50000, min_df=2, max_df=0.95):
    """Build a sparse TF-IDF matrix and return its fitted vectorizer."""
    vectorizer = TfidfVectorizer(
        lowercase=False,       # preprocessing already lowercases text
        token_pattern=r"(?u)\b[a-z0-9]+\b",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        norm="l2",
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer
