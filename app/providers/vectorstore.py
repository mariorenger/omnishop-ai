"""Hybrid retrieval over Postgres (ADR-002).

Each query runs inside a tenant_tx, so Row-Level Security already restricts rows
to the org; we additionally scope by shop and (optionally) bot.

Retrieval is HYBRID: we combine
  1. dense/semantic ranking — pgvector cosine distance on the embedding, and
  2. lexical/keyword ranking — pg_trgm word-similarity on the raw text,
fused with Reciprocal Rank Fusion (RRF). Semantic search catches paraphrases;
lexical search catches exact terms, SKUs, codes and rare words the embedding
blurs. Backend is swappable (pgvector -> Qdrant + BM25) behind these functions.
"""
from __future__ import annotations
from typing import List, Optional

import psycopg

from .embeddings import to_pgvector

_RRF_K = 60          # RRF damping constant (standard default)
_CAND = 20           # candidates pulled from each ranker before fusion


def _rrf(vector_ids: List, keyword_ids: List) -> dict:
    """Reciprocal Rank Fusion: score(id) = sum 1/(K + rank) across rankers."""
    score: dict = {}
    for rank, _id in enumerate(vector_ids):
        score[_id] = score.get(_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, _id in enumerate(keyword_ids):
        score[_id] = score.get(_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return score


def search_chunks(conn: psycopg.Connection, shop_id: str, query_vec: List[float],
                  query_text: str = "", k: int = 4, bot_id: Optional[str] = None):
    bot_clause = "AND (c.bot_id IS NULL OR c.bot_id = %s)" if bot_id else ""

    # 1) semantic candidates
    vp: list = [to_pgvector(query_vec), shop_id]
    if bot_id:
        vp.append(bot_id)
    vp += [to_pgvector(query_vec), _CAND]
    vec_rows = conn.execute(
        f"""SELECT c.id, d.title AS title, c.content AS content,
                   1 - (c.embedding <=> %s::vector) AS score
            FROM chunk c
            JOIN knowledge_base kb ON kb.id = c.knowledge_base_id
            JOIN document d ON d.id = c.document_id
            WHERE kb.shop_id = %s AND c.embedding IS NOT NULL AND d.active {bot_clause}
            ORDER BY c.embedding <=> %s::vector LIMIT %s""",
        tuple(vp),
    ).fetchall()

    # 2) lexical candidates (only when there is a query string)
    kw_rows = []
    if query_text.strip():
        kp: list = [query_text, shop_id]
        if bot_id:
            kp.append(bot_id)
        kp += [query_text, query_text, _CAND]
        kw_rows = conn.execute(
            f"""SELECT c.id, d.title AS title, c.content AS content,
                       word_similarity(%s, c.content) AS score
                FROM chunk c
                JOIN knowledge_base kb ON kb.id = c.knowledge_base_id
                JOIN document d ON d.id = c.document_id
                WHERE kb.shop_id = %s AND c.embedding IS NOT NULL AND d.active {bot_clause}
                  AND word_similarity(%s, c.content) > 0.1
                ORDER BY word_similarity(%s, c.content) DESC LIMIT %s""",
            tuple(kp),
        ).fetchall()

    return _fuse(vec_rows, kw_rows, k)


def search_products(conn: psycopg.Connection, shop_id: str, query_vec: List[float],
                    query_text: str = "", k: int = 4, bot_id: Optional[str] = None):
    bot_clause = "AND (p.bot_id IS NULL OR p.bot_id = %s)" if bot_id else ""
    cols = "p.id, p.name, p.description, p.price, p.currency, p.sku, p.attributes"

    vp: list = [to_pgvector(query_vec), shop_id]
    if bot_id:
        vp.append(bot_id)
    vp += [to_pgvector(query_vec), _CAND]
    vec_rows = conn.execute(
        f"""SELECT {cols}, 1 - (p.embedding <=> %s::vector) AS score
            FROM product p
            WHERE p.shop_id = %s AND p.embedding IS NOT NULL {bot_clause}
            ORDER BY p.embedding <=> %s::vector LIMIT %s""",
        tuple(vp),
    ).fetchall()

    kw_rows = []
    if query_text.strip():
        kp: list = [query_text, shop_id]
        if bot_id:
            kp.append(bot_id)
        kp += [query_text, query_text, _CAND]
        kw_rows = conn.execute(
            f"""SELECT {cols},
                       word_similarity(%s, coalesce(p.name,'') || ' ' || coalesce(p.description,'') || ' ' || coalesce(p.sku,'')) AS score
                FROM product p
                WHERE p.shop_id = %s AND p.embedding IS NOT NULL {bot_clause}
                  AND word_similarity(%s, coalesce(p.name,'') || ' ' || coalesce(p.description,'') || ' ' || coalesce(p.sku,'')) > 0.1
                ORDER BY word_similarity(%s, coalesce(p.name,'') || ' ' || coalesce(p.description,'') || ' ' || coalesce(p.sku,'')) DESC
                LIMIT %s""",
            tuple(kp),
        ).fetchall()

    return _fuse(vec_rows, kw_rows, k)


def _fuse(vec_rows, kw_rows, k: int):
    """Merge two ranked candidate lists with RRF, keep row dicts, return top k.
    The fused RRF weight is exposed as `score` (higher = better) so callers keep
    a single threshold; the raw semantic score is kept as `vscore`."""
    by_id = {}
    for r in list(vec_rows) + list(kw_rows):
        by_id.setdefault(r["id"], r)
    fused = _rrf([r["id"] for r in vec_rows], [r["id"] for r in kw_rows])
    vscore = {r["id"]: float(r["score"]) for r in vec_rows}
    kwscore = {r["id"]: float(r["score"]) for r in kw_rows}
    out = []
    for _id, s in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]:
        row = dict(by_id[_id])
        row["vscore"] = vscore.get(_id, 0.0)       # semantic cosine (0..1)
        row["kwscore"] = kwscore.get(_id, 0.0)     # lexical word-similarity (0..1)
        row["kw"] = _id in kwscore                 # matched the keyword ranker
        row["score"] = float(s)                    # fused RRF weight
        out.append(row)
    return out


def variants_for(conn: psycopg.Connection, product_id: str):
    return conn.execute(
        """SELECT name, sku, price, stock FROM product_variant
           WHERE product_id = %s ORDER BY name""",
        (product_id,),
    ).fetchall()
