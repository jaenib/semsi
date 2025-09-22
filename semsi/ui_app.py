from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Iterable

import streamlit as st

from .data import TaggedDocument, parse_contents_file, parse_contents_lines
from .embedding import TagEmbeddingModel
from .similarity import SimilarityMatrix, build_similarity_matrix, get_top_similar

try:  # pandas offers nicer tables but is optional
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None


st.set_page_config(page_title="Semsi Explorer", layout="wide")
st.title("Semsi Similarity Explorer")
st.write(
    "Upload a `contents.txt` file (or use the bundled example) to build a semantic similarity "
    "matrix powered by Semsi's TF–IDF embedding pipeline."
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_example_path() -> Path | None:
    candidate = _project_root() / "example_data" / "contents.txt"
    return candidate if candidate.exists() else None


def _documents_preview(documents: Iterable[TaggedDocument]) -> list[dict[str, str]]:
    return [{"identifier": doc.identifier, "tags": ", ".join(doc.tags)} for doc in documents]


def _matrix_to_csv(matrix: SimilarityMatrix) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["identifier", *matrix.labels])
    for label, row in zip(matrix.labels, matrix.values):
        writer.writerow([label, *row])
    return buffer.getvalue()


def _matrix_to_json(matrix: SimilarityMatrix) -> str:
    payload = matrix.to_dict()
    return json.dumps(payload, indent=2)


def _load_from_path(path: Path, *, keep_duplicates: bool) -> list[TaggedDocument]:
    return parse_contents_file(path, drop_duplicates=not keep_duplicates)


def _load_from_lines(lines: Iterable[str], *, keep_duplicates: bool) -> list[TaggedDocument]:
    return parse_contents_lines(lines, drop_duplicates=not keep_duplicates)


@st.cache_data(show_spinner=False)
def _compute_similarity(documents: tuple[TaggedDocument, ...]) -> SimilarityMatrix:
    model = TagEmbeddingModel()
    embeddings = model.fit_transform(documents)
    return build_similarity_matrix(documents, embeddings)


def _display_documents(documents: list[TaggedDocument]) -> None:
    preview_rows = _documents_preview(documents)
    st.subheader("Parsed documents")
    if pd is not None:
        frame = pd.DataFrame(preview_rows)
        st.dataframe(frame, hide_index=True, use_container_width=True)
    else:
        st.table(preview_rows)


def _display_similarity(matrix: SimilarityMatrix, target: str, top_n: int) -> None:
    matches = get_top_similar(matrix, target, top_n=top_n)
    table_data = [{"identifier": label, "score": score} for label, score in matches]
    if pd is not None:
        frame = pd.DataFrame(table_data)
        st.dataframe(frame, hide_index=True, use_container_width=True)
    else:
        st.table(table_data)


sidebar = st.sidebar
sidebar.header("Dataset")
source = sidebar.radio(
    "Select data source",
    options=("Example data", "Upload file", "Local path"),
)
keep_duplicates = sidebar.toggle("Keep duplicate identifiers", value=False)

loaded_documents: list[TaggedDocument] | None = None
load_error: str | None = None

if source == "Example data":
    example_path = _load_example_path()
    if example_path is None:
        load_error = "Could not find the bundled example data."
    else:
        loaded_documents = _load_from_path(example_path, keep_duplicates=keep_duplicates)
        try:
            sidebar.success(f"Using {example_path.relative_to(_project_root())}")
        except ValueError:  # pragma: no cover - only triggered outside an editable checkout
            sidebar.success(f"Using {example_path}")
elif source == "Upload file":
    uploaded = sidebar.file_uploader("Upload a contents.txt file", type=["txt"])
    if uploaded is not None:
        try:
            text = uploaded.getvalue().decode("utf-8").splitlines()
            loaded_documents = _load_from_lines(text, keep_duplicates=keep_duplicates)
        except UnicodeDecodeError as exc:
            load_error = f"Could not decode uploaded file: {exc}"
elif source == "Local path":
    default_path = _load_example_path()
    placeholder = str(default_path) if default_path else "example_data/contents.txt"
    user_path = sidebar.text_input("Path to contents.txt", value=placeholder)
    if user_path:
        path = Path(user_path).expanduser()
        if path.exists():
            loaded_documents = _load_from_path(path, keep_duplicates=keep_duplicates)
        else:
            load_error = f"File not found: {path}"  # pragma: no cover

if load_error:
    st.error(load_error)

if not loaded_documents:
    st.info("Select a contents file from the sidebar to get started.")
    st.stop()

st.success(f"Parsed {len(loaded_documents)} documents.")
_display_documents(loaded_documents)

matrix = _compute_similarity(tuple(loaded_documents))

st.subheader("Explore similarities")
col_left, col_right = st.columns((2, 1))
with col_left:
    target = st.selectbox("Target document", options=list(matrix.labels))
with col_right:
    max_top = max(1, len(matrix.labels) - 1)
    top_n = st.slider("Show top N", min_value=1, max_value=max_top, value=min(5, max_top))

_display_similarity(matrix, target, top_n)

st.subheader("Download results")
col_csv, col_json = st.columns(2)
with col_csv:
    csv_data = _matrix_to_csv(matrix)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="semsi_similarity.csv",
        mime="text/csv",
    )
with col_json:
    json_data = _matrix_to_json(matrix)
    st.download_button(
        label="Download JSON",
        data=json_data,
        file_name="semsi_similarity.json",
        mime="application/json",
    )

st.caption(
    "Semsi builds TF–IDF embeddings on the fly using only the tags you supply. "
    "Use the command line (`python -m semsi.cli`) for scripted workflows or this interface "
    "for quick exploration."
)
