"""Reciprocal Rank Fusion: an item ranked well by both rankers must beat one
ranked well by only a single ranker."""
from app.providers.vectorstore import _fuse, _rrf


def test_rrf_rewards_agreement():
    # A: rank0 in vector, rank0 in keyword. B: rank0 vector only. C: rank0 keyword only.
    score = _rrf(["A", "B"], ["A", "C"])
    assert score["A"] > score["B"]
    assert score["A"] > score["C"]


def test_fuse_marks_keyword_hits_and_orders():
    vec = [{"id": "A", "content": "a", "score": 0.9},
           {"id": "B", "content": "b", "score": 0.4}]
    kw = [{"id": "A", "content": "a", "score": 0.8},
          {"id": "C", "content": "c", "score": 0.7}]
    out = _fuse(vec, kw, k=3)
    ids = [r["id"] for r in out]
    assert ids[0] == "A"                      # agreed by both -> top
    a = next(r for r in out if r["id"] == "A")
    assert a["kw"] is True and a["vscore"] == 0.9 and a["kwscore"] == 0.8
    c = next(r for r in out if r["id"] == "C")
    assert c["kw"] is True and c["vscore"] == 0.0     # keyword-only hit still surfaced
