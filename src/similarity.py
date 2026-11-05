"""Rank and retrieve the five most similar patient records."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def top_k_from_scores(scores, query_index, uids, k=5):
    scores = np.asarray(scores).ravel().copy()
    scores[query_index] = -np.inf  # never retrieve a patient as its own match
    # Stable sorting makes ties deterministic.
    order = np.argsort(-scores, kind="stable")
    return [uids[i] for i in order[:k] if np.isfinite(scores[i])]


def retrieve_all(tfidf_matrix, bm25_index, records, processed_texts, k=5):
    uids = [record["patient_uid"] for record in records]
    output = {}
    # Compute sparse cosine similarities in one operation for efficiency.
    tfidf_similarity = cosine_similarity(tfidf_matrix, dense_output=False)
    for i, record in enumerate(records):
        tfidf_scores = tfidf_similarity.getrow(i).toarray().ravel()
        bm25_scores = bm25_index.get_scores(bm25_index.documents[i])

        output[uids[i]] = {
            "patient_id": str(record["patient_id"]),
            "tfidf": top_k_from_scores(tfidf_scores, i, uids, k),
            "bm25": top_k_from_scores(bm25_scores, i, uids, k),
        }
        if (i + 1) % 500 == 0:
            print(f"Retrieved similar patients for {i + 1:,}/{len(records):,}")
    return output
