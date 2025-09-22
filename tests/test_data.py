from semsi.data import parse_contents_lines


def test_parse_contents_lines_handles_malformed_entries():
    lines = [
        "['first', 'second'].txt",
        "['first, 'second', 'third'].rtf",
        "[' lone '] .pdf",
        "no brackets here",
        "   ",
    ]

    documents = parse_contents_lines(lines)
    identifiers = [doc.identifier for doc in documents]

    assert identifiers[0] == "first_second.txt"
    assert "third" in documents[1].tags
    assert any(tag == "lone" for tag in documents[2].tags)
    assert len(documents) == 3


def test_parse_contents_lines_can_keep_duplicates():
    lines = ["['a', 'b'].txt", "['a', 'b'].txt"]

    unique = parse_contents_lines(lines)
    duplicates = parse_contents_lines(lines, drop_duplicates=False)

    assert len(unique) == 1
    assert len(duplicates) == 2
