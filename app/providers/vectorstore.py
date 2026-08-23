"""Retrieval over pgvector (ADR-002). Every query runs inside a tenant_tx, so
Row-Level Security already restricts rows to the org; we additionally scope by
shop. Backend is swappable (pgvector -> Qdrant) behind these functions.
"""
from __future__ import annotations
from typing import List

import psycopg

from .embeddings import to_pgvector


def search_chunks(conn: psycopg.Connection, shop_id: str, query_vec: List[float], k: int = 4):
    rows = conn.execute(
        """
        SELECT d.title AS title, c.content AS content,
               1 - (c.embedding <=> %s::vector) AS score
        FROM chunk c
        JOIN knowledge_base kb ON kb.id = c.knowledge_base_id
        JOIN document d ON d.id = c.document_id
        WHERE kb.shop_id = %s AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (to_pgvector(query_vec), shop_id, to_pgvector(query_vec), k),
    ).fetchall()
    return rows


def search_products(conn: psycopg.Connection, shop_id: str, query_vec: List[float], k: int = 4):
    rows = conn.execute(
        """
        SELECT p.id, p.name, p.description, p.price, p.currency, p.sku, p.attributes,
               1 - (p.embedding <=> %s::vector) AS score
        FROM product p
        WHERE p.shop_id = %s AND p.embedding IS NOT NULL
        ORDER BY p.embedding <=> %s::vector
        LIMIT %s
        """,
        (to_pgvector(query_vec), shop_id, to_pgvector(query_vec), k),
    ).fetchall()
    return rows


def variants_for(conn: psycopg.Connection, product_id: str):
    return conn.execute(
        """SELECT name, sku, price, stock FROM product_variant
           WHERE product_id = %s ORDER BY name""",
        (product_id,),
    ).fetchall()
