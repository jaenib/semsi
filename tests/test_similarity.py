from math import isclose

from semsi.data import TaggedDocument
from semsi.embedding import TagEmbeddingModel
from semsi.similarity import build_similarity_matrix, get_top_similar


def test_similarity_matrix_roundtrip():
    documents = [
        TaggedDocument(identifier="doc1.txt", tags=("crowd", "city")),
        TaggedDocument(identifier="doc2.txt", tags=("city", "street")),
        TaggedDocument(identifier="doc3.txt", tags=("river", "nature")),
    ]

    model = TagEmbeddingModel()
    embeddings = model.fit_transform(documents)
    matrix = build_similarity_matrix(documents, embeddings, decimals=3)

    assert len(matrix.labels) == 3
    assert len(matrix.values) == 3
    assert isclose(matrix.values[0][1], matrix.values[1][0])
    assert matrix.values[0][2] < 0.5


def test_get_top_similar_returns_sorted_scores():
    documents = [
        TaggedDocument(identifier="doc1.txt", tags=("crowd", "city")),
        TaggedDocument(identifier="doc2.txt", tags=("city", "street")),
        TaggedDocument(identifier="doc3.txt", tags=("river", "nature")),
    ]

    model = TagEmbeddingModel()
    embeddings = model.fit_transform(documents)
    matrix = build_similarity_matrix(documents, embeddings)

    top = get_top_similar(matrix, "doc1.txt", top_n=2)
    assert top[0][0] == "doc2.txt"
    assert len(top) == 2
