"""Semsi core package for building document similarity matrices."""

from .data import TaggedDocument, parse_contents_file
from .embedding import TagEmbeddingModel
from .similarity import SimilarityMatrix, build_similarity_matrix, get_top_similar

__all__ = [
    "TaggedDocument",
    "parse_contents_file",
    "TagEmbeddingModel",
    "SimilarityMatrix",
    "build_similarity_matrix",
    "get_top_similar",
]
