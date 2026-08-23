"""EmbeddingProvider (behind an interface, ADR-002/ADR-007).

Default: LocalEmbeddingProvider — deterministic, key-free, hashed bag-of-words
into EMBEDDING_DIM dims (L2-normalized). Good enough to demo retrieval offline.
Optional: OpenAIEmbeddingProvider (text-embedding-3-small with `dimensions`
matching EMBEDDING_DIM so the pgvector column stays fixed).
"""
from __future__ import annotations
import math
import re
from typing import List, Protocol

from ..config import config

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


class EmbeddingProvider(Protocol):
    dim: int
    def embed(self, texts: List[str]) -> List[List[float]]: ...
    def embed_one(self, text: str) -> List[float]: ...


class LocalEmbeddingProvider:
    """Hashed TF embedding — no external calls, deterministic, offline."""

    name = "local-hash"

    def __init__(self, dim: int):
        self.dim = dim

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in _tokens(text):
            # deterministic across processes (built-in hash() is salted per run)
            h = _stable_hash(tok) % self.dim
            v[h] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> List[float]:
        return self._vec(text)


def _stable_hash(s: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        import httpx
        resp = httpx.post(
            f"{config.OPENAI_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={"model": config.EMBEDDING_MODEL, "input": texts, "dimensions": self.dim},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in data]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


_provider: EmbeddingProvider | None = None


def get_embedder() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        if config.EMBEDDING_PROVIDER == "openai" and config.OPENAI_API_KEY:
            _provider = OpenAIEmbeddingProvider(config.EMBEDDING_DIM)
        else:
            _provider = LocalEmbeddingProvider(config.EMBEDDING_DIM)
    return _provider


def to_pgvector(vec: List[float]) -> str:
    """Render a python list as a pgvector literal '[a,b,c]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
