"""Data loading utilities for Semsi."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaggedDocument:
    """A simple container describing a tagged document.

    Parameters
    ----------
    identifier:
        A stable name that can be used as an index in similarity matrices.
    tags:
        Normalised list of tags attached to the document.
    source:
        Optional raw string that was used to build the identifier.
    """

    identifier: str
    tags: tuple[str, ...]
    source: str | None = None

    def as_text(self) -> str:
        """Return the tags joined as a single whitespace separated string."""

        return " ".join(self.tags)


def _extract_bracket_content(line: str) -> str | None:
    start = line.find("[")
    end = line.find("]", start + 1)
    if start == -1 or end == -1:
        return None
    return line[start + 1 : end]


def _normalise_tag(token: str) -> str | None:
    cleaned = token.strip().strip("'\"` “”’·")
    if not cleaned:
        return None
    return " ".join(cleaned.split())


def _build_identifier(tags: Sequence[str], line: str) -> str:
    suffix: str | None = None
    closing = line.find("]")
    if closing != -1:
        suffix_candidate = line[closing + 1 :].strip()
        if suffix_candidate.startswith("."):
            suffix = suffix_candidate[1:]
    base = "_".join(tags) if tags else line.strip()
    return f"{base}.{suffix}" if suffix else base


def parse_contents_lines(lines: Iterable[str], *, drop_duplicates: bool = True) -> list[TaggedDocument]:
    """Parse the raw ``contents.txt`` lines into :class:`TaggedDocument` objects.

    The original notebooks stored filenames such as ``['tag', 'other'].txt``.
    The parser is intentionally forgiving and will accept malformed entries by
    stripping stray quotation marks and whitespace.
    """

    documents: list[TaggedDocument] = []
    seen_identifiers: set[str] = set()

    for index, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        bracket_content = _extract_bracket_content(line)
        if bracket_content is None:
            if line.lower().endswith("contents.txt"):
                # Legacy files sometimes list themselves at the bottom.
                continue
            logger.warning("Line %s does not contain bracketed tags: %s", index, line)
            continue

        tags: list[str] = []
        for token in bracket_content.split(","):
            normalised = _normalise_tag(token)
            if normalised:
                tags.append(normalised)

        if not tags:
            logger.warning("Line %s did not yield any valid tags: %s", index, line)
            continue

        identifier = _build_identifier(tags, line)
        if drop_duplicates and identifier in seen_identifiers:
            logger.info("Skipping duplicate identifier %s", identifier)
            continue
        if drop_duplicates:
            seen_identifiers.add(identifier)

        documents.append(TaggedDocument(identifier=identifier, tags=tuple(tags), source=line))

    return documents


def parse_contents_file(path: str | Path, *, drop_duplicates: bool = True) -> list[TaggedDocument]:
    """Read and parse a ``contents.txt`` style file.

    Parameters
    ----------
    path:
        Location of the file. Paths are expanded and read using UTF-8.
    """

    file_path = Path(path).expanduser().resolve()
    raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    return parse_contents_lines(raw_lines, drop_duplicates=drop_duplicates)


__all__ = ["TaggedDocument", "parse_contents_file", "parse_contents_lines"]
