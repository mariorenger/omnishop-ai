"""EmbeddingProvider (ADR-002). Platform-managed (one model per platform so the
shared pgvector column stays consistent). Built from a resolved config dict.

Providers: local (key-free, deterministic), openai_compatible (OpenAI / vLLM /
text-embeddings-inference / etc.), gemini (OpenAI-compatible endpoint). Any
provider's output is fit to EMBEDDING_DIM so the fixed column always matches.
"""
from __future__ import annotations
import hashlib
import math
import re
from typing import List, Protocol

from ..config import config

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


def _fit(vec: List[float], dim: int) -> List[float]:
    """Truncate/pad to `dim` and L2-normalize so any model fits the fixed column."""
    if len(vec) > dim:
        vec = vec[:dim]
    elif len(vec) < dim:
        vec = list(vec) + [0.0] * (dim - len(vec))
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class EmbeddingProvider(Protocol):
    dim: int
    def embed(self, texts: List[str]) -> List[List[float]]: ...
    def embed_one(self, text: str) -> List[float]: ...


class LocalEmbeddingProvider:
    name = "local"

    def __init__(self, dim: int):
        self.dim = dim

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in _tokens(text):
            v[_stable_hash(tok) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> List[float]:
        return self._vec(text)


class OpenAICompatibleEmbedding:
    name = "openai_compatible"

    def __init__(self, cfg: dict, gemini: bool = False):
        self.dim = config.EMBEDDING_DIM
        self._model = cfg.get("model") or ("text-embedding-004" if gemini else config.EMBEDDING_MODEL)
        self._base = (cfg.get("base_url") or (GEMINI_OPENAI_BASE if gemini else config.OPENAI_BASE_URL)).rstrip("/")
        self._key = cfg.get("api_key") or ""

    def embed(self, texts: List[str]) -> List[List[float]]:
        import httpx
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        payload = {"model": self._model, "input": texts, "dimensions": self.dim}
        r = httpx.post(f"{self._base}/embeddings", headers=headers, json=payload, timeout=60)
        if r.status_code == 400:  # some servers reject `dimensions`
            payload.pop("dimensions", None)
            r = httpx.post(f"{self._base}/embeddings", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return [_fit(d["embedding"], self.dim) for d in r.json()["data"]]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


def build_embedder(cfg: dict) -> EmbeddingProvider:
    p = (cfg.get("provider") or "local").lower()
    try:
        if p == "openai_compatible":
            return OpenAICompatibleEmbedding(cfg)
        if p == "gemini":
            return OpenAICompatibleEmbedding(cfg, gemini=True)
        return LocalEmbeddingProvider(config.EMBEDDING_DIM)
    except Exception:  # noqa: BLE001
        return LocalEmbeddingProvider(config.EMBEDDING_DIM)


def to_pgvector(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
