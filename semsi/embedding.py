"""Embedding utilities for turning tags into TF-IDF vectors."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log
from typing import Sequence

from .data import TaggedDocument


@dataclass
class TagEmbeddingModel:
    """Fit and transform :class:`TaggedDocument` collections into embeddings."""

    vocabulary_: dict[str, int] = field(init=False, default_factory=dict)
    idf_: list[float] = field(init=False, default_factory=list)
    fitted: bool = field(init=False, default=False)

    def fit(self, documents: Sequence[TaggedDocument]) -> "TagEmbeddingModel":
        if not documents:
            raise ValueError("No documents were provided to fit the embedding model.")

        doc_count = len(documents)
        document_frequency: Counter[str] = Counter()
        for document in documents:
            unique_tags = set(document.tags)
            for tag in unique_tags:
                document_frequency[tag] += 1

        sorted_tags = sorted(document_frequency.keys())
        self.vocabulary_ = {tag: index for index, tag in enumerate(sorted_tags)}
        self.idf_ = [
            log((1 + doc_count) / (1 + document_frequency[tag])) + 1.0 for tag in sorted_tags
        ]
        self.fitted = True
        return self

    def transform(self, documents: Sequence[TaggedDocument]) -> list[list[float]]:
        if not self.fitted:
            raise RuntimeError("The embedding model must be fitted before calling transform().")

        vectors: list[list[float]] = []
        for document in documents:
            counts = Counter(document.tags)
            total = sum(counts.values()) or 1
            vector = [0.0] * len(self.vocabulary_)
            for tag, count in counts.items():
                index = self.vocabulary_.get(tag)
                if index is None:
                    continue
                tf = count / total
                vector[index] = tf * self.idf_[index]
            vectors.append(vector)
        return vectors

    def fit_transform(self, documents: Sequence[TaggedDocument]) -> list[list[float]]:
        self.fit(documents)
        return self.transform(documents)


__all__ = ["TagEmbeddingModel"]
