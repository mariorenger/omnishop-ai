"""File text extraction + chunking (pure-python paths, no OCR/network)."""
import json

from app.ingest.parse import extract_text
from app.modules.knowledge import chunk_text


def test_extract_plain_and_csv():
    assert extract_text("a.txt", "chính sách đổi trả".encode()) == "chính sách đổi trả"
    assert "sku,price" in extract_text("p.csv", b"sku,price\nA,10").lower()


def test_extract_json_pretty():
    out = extract_text("d.json", json.dumps({"k": "v"}).encode())
    assert "k" in out and "v" in out


def test_extract_html_strips_tags():
    out = extract_text("p.html", b"<html><body><h1>Xin</h1><p>chao</p></body></html>")
    assert "Xin" in out and "<h1>" not in out


def test_extract_unknown_type_best_effort_then_empty_raises():
    # unknown types are decoded best-effort; readable bytes still yield text...
    assert extract_text("x.bin", b"hello world") == "hello world"
    # ...but an unknown type with no extractable text is rejected (upload shows an error)
    import pytest
    with pytest.raises(ValueError):
        extract_text("x.bin", b"   \n\t  ")


def test_chunk_text_respects_target():
    text = "\n\n".join(f"đoạn số {i} " * 40 for i in range(10))
    chunks = chunk_text(text, target=300)
    assert len(chunks) > 1
    assert all(len(c) <= 320 for c in chunks)   # small slack over target
    assert "".join(chunks).replace("\n", "").replace(" ", "")  # nothing lost to empties
