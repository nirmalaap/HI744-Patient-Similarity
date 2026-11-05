
"""BM25 retrieval using an inverted index for efficient ranked lookup."""

import math
import re
from collections import Counter, defaultdict

import numpy as np


class BM25Index:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        # Store all tokenized patient documents.
        self.documents = tokenized_docs

        # Calculate the length of each patient document.
        self.doc_len = np.array(
            [len(doc) for doc in tokenized_docs],
            dtype=np.float32
        )

        # Calculate average document length.
        self.avgdl = (
            float(self.doc_len.mean())
            if len(self.doc_len)
            else 0.0
        )

        self.n_docs = len(tokenized_docs)

        # Inverted index: term -> list of (document ID, weight).
        self.postings = defaultdict(list)

        # Store inverse document frequency for each term.
        self.idf = {}

        # Calculate term frequency and document frequency.
        document_frequency = Counter()
        term_frequencies = []

        for doc_id, tokens in enumerate(tokenized_docs):
            counts = Counter(tokens)

            term_frequencies.append(counts)
            document_frequency.update(counts.keys())

        # Calculate positive BM25 inverse document frequency.
        for term, df in document_frequency.items():
            self.idf[term] = math.log(
                1.0 + (self.n_docs - df + 0.5) / (df + 0.5)
            )

        # Build the inverted index with BM25 term weights.
        for doc_id, counts in enumerate(term_frequencies):

            length = self.doc_len[doc_id]

            length_norm = (
                self.k1 * (
                    1.0 - self.b + self.b * length / self.avgdl
                )
                if self.avgdl
                else self.k1
            )

            for term, tf in counts.items():

                weight = self.idf[term] * (
                    tf * (self.k1 + 1.0) /
                    (tf + length_norm)
                )

                self.postings[term].append(
                    (doc_id, weight)
                )

    def get_scores(self, query_tokens):
        """Calculate BM25 scores for all patient documents."""

        scores = np.zeros(
            self.n_docs,
            dtype=np.float32
        )

        # Calculate scores using unique query terms.
        for term in set(query_tokens):

            for doc_id, weight in self.postings.get(term, ()):
                scores[doc_id] += weight

        return scores


def tokenize_for_bm25(text):
    """Convert processed clinical text into lowercase tokens."""

    return re.findall(
        r"[a-z0-9]+",
        text.lower()
    )


def build_bm25_index(processed_texts, k1=1.5, b=0.75):
    """Build the BM25 index from processed patient documents."""

    tokens = [
        tokenize_for_bm25(text)
        for text in processed_texts
    ]

    return BM25Index(
        tokens,
        k1=k1,
        b=b
    )
