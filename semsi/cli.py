"""Command line interface for Semsi."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .data import parse_contents_file
from .embedding import TagEmbeddingModel
from .similarity import SimilarityMatrix, build_similarity_matrix, get_top_similar


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build semantic similarity matrices from tag lists.")
    parser.add_argument("contents", type=Path, help="Path to a contents.txt style file with tag information.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for persisting the similarity matrix.")
    parser.add_argument(
        "--format",
        choices=("csv", "pickle", "json"),
        default="csv",
        help="Format to use when --output is specified.",
    )
    parser.add_argument("--top", type=int, default=0, help="Print the N most similar documents for the selected target.")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Document identifier to analyse when using --top. Defaults to the first parsed document.",
    )
    parser.add_argument("--list", action="store_true", help="List parsed document identifiers and exit.")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Preserve duplicate document identifiers when parsing the contents file.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of rows/columns to display when printing the similarity matrix preview.",
    )
    return parser


def _save_matrix(matrix: SimilarityMatrix, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        matrix.save_csv(path)
    elif fmt == "pickle":
        matrix.save_pickle(path)
    elif fmt == "json":
        matrix.save_json(path)
    else:  # pragma: no cover - defensive fallback
        raise ValueError(f"Unsupported format: {fmt}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    documents = parse_contents_file(args.contents, drop_duplicates=not args.keep_duplicates)
    if not documents:
        parser.error("No documents could be parsed from the provided contents file.")

    if args.list:
        for doc in documents:
            print(doc.identifier)
        return 0

    model = TagEmbeddingModel()
    embeddings = model.fit_transform(documents)
    similarity = build_similarity_matrix(documents, embeddings)

    if args.output is not None:
        _save_matrix(similarity, args.output, args.format)

    if args.top:
        target = args.target or documents[0].identifier
        try:
            top_scores = get_top_similar(similarity, target, top_n=args.top)
        except KeyError as exc:
            parser.error(str(exc))
        print(f"Top {len(top_scores)} matches for {target}:")
        for label, score in top_scores:
            print(f"  {label}: {score:.3f}")
    else:
        print(similarity.preview(limit=args.preview))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
