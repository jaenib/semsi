# Semsi

Semsi explores the semantic proximity of tagged artefacts so that files can be
sorted and browsed by "the company they keep". The original experiment lived in
Colab notebooks that downloaded GloVe vectors and manually wrangled the
resulting similarity matrix. This repository now exposes a lightweight Python
module and command line interface that reproduce the workflow in a more
reliable and reusable way, without external dependencies.

## Modernised workflow

1. Parse the `contents.txt` file describing your artefacts and their tags.
2. Convert the tags into TF-IDF embeddings (implemented with the standard
   library so it runs anywhere).
3. Build a cosine similarity matrix and explore the most related files.

The logic behind these steps lives in the `semsi` package (`semsi/data.py`,
`semsi/embedding.py`, `semsi/similarity.py`). You can reuse the components in
scripts or notebooks without having to redo the original notebook plumbing.

## Quick start

Run the CLI directly with `python -m` from the repository root:

```bash
python -m semsi.cli example_data/contents.txt --preview 4
```

* `--output` controls where the similarity matrix is stored (CSV, pickle or
  JSON).
* `--top` prints the N closest files to the `--target` identifier directly in
  the terminal.
* `--list` emits the parsed identifiers without computing a matrix. This is
  handy for spotting typos in the metadata file.

## Python usage

```python
from semsi import parse_contents_file, TagEmbeddingModel, build_similarity_matrix, get_top_similar

# Parse the metadata file
documents = parse_contents_file("example_data/contents.txt")

# Fit a TF-IDF model on the tags and construct the similarity matrix
model = TagEmbeddingModel()
embeddings = model.fit_transform(documents)
similarity = build_similarity_matrix(documents, embeddings)

# Inspect the nearest neighbours for a particular document
top = get_top_similar(similarity, documents[0].identifier, top_n=5)
for label, score in top:
    print(label, score)
```

## Tests

Run the unit tests with `pytest`:

```bash
pytest
```

## Legacy notebooks

The original Colab notebooks remain under `semsi_jupyter/` for reference. They
have not been deleted, but the heavy lifting is now handled by the importable
package which can be exercised from notebooks without duplicating setup cells.
