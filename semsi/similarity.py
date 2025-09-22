"""Similarity helpers for Semsi."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from math import sqrt
from pathlib import Path
from typing import Sequence
import csv
import json

from .data import TaggedDocument


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class SimilarityMatrix:
    labels: tuple[str, ...]
    values: list[list[float]]

    def row(self, label: str) -> list[tuple[str, float]]:
        index = self.labels.index(label)
        return list(zip(self.labels, self.values[index]))

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            label: {other: score for other, score in zip(self.labels, row)}
            for label, row in zip(self.labels, self.values)
        }

    def save_csv(self, path: Path, decimals: int = 6) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["identifier", *self.labels])
            for label, row in zip(self.labels, self.values):
                writer.writerow([label, *[f"{value:.{decimals}f}" for value in row]])

    def save_pickle(self, path: Path) -> None:
        import pickle

        with path.open("wb") as handle:
            pickle.dump({"labels": self.labels, "values": self.values}, handle)

    def save_json(self, path: Path, decimals: int = 6) -> None:
        payload = {
            label: {other: round(score, decimals) for other, score in zip(self.labels, row)}
            for label, row in zip(self.labels, self.values)
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def preview(self, limit: int = 5, decimals: int = 3) -> str:
        limit = max(1, min(limit, len(self.labels)))
        header = ["identifier", *list(islice(self.labels, limit))]
        rows = ["\t".join(header)]
        for label, row in zip(self.labels, self.values):
            values = [f"{score:.{decimals}f}" for score in row[:limit]]
            rows.append("\t".join([label, *values]))
            if len(rows) > limit:
                break
        return "\n".join(rows)


def build_similarity_matrix(
    documents: Sequence[TaggedDocument],
    embeddings: Sequence[Sequence[float]],
    *,
    decimals: int | None = 6,
) -> SimilarityMatrix:
    if len(documents) == 0:
        raise ValueError("Cannot build a similarity matrix without documents.")

    labels = tuple(doc.identifier for doc in documents)
    matrix: list[list[float]] = []

    for vec_a in embeddings:
        row: list[float] = []
        for vec_b in embeddings:
            similarity = _cosine_similarity(vec_a, vec_b)
            if decimals is not None:
                similarity = round(similarity, decimals)
            row.append(similarity)
        matrix.append(row)

    return SimilarityMatrix(labels=labels, values=matrix)


def get_top_similar(
    similarity_matrix: SimilarityMatrix,
    target: str,
    *,
    top_n: int = 10,
    include_self: bool = False,
) -> list[tuple[str, float]]:
    if target not in similarity_matrix.labels:
        raise KeyError(f"Target '{target}' not present in similarity matrix")

    scores = similarity_matrix.row(target)
    if not include_self:
        scores = [(label, score) for label, score in scores if label != target]
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_n]


__all__ = ["SimilarityMatrix", "build_similarity_matrix", "get_top_similar"]
