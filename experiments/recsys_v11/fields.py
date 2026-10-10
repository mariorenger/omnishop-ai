"""More of the data for the frozen-vector techniques (stage ``moredata``).

What MIND-small's ``news.tsv`` holds per article: category, subcategory, title, abstract, URL, and the Wikidata entities found in the
title and in the abstract.  The release has **no article body** (the URL points to a page the benchmark does not ship), so the
abstract is the longest text there is.  The suite has always embedded ``title. abstract`` (``meta['text']``); what it never fed to an
encoder is the category / subcategory phrase, the entity names, or more than a reader's last 50 clicks.  This module builds those
variants so that each one can be measured on its own against the suite's text:

  t      title only                                    (the difference to ``ta`` is what the abstract adds)
  ta     title. abstract                               (the suite's own text: the control)
  cta    Category: c. Subcategory: s. title. abstract
  ctae   cta + Entities: e1; e2; ...
  views  title, abstract and category phrase embedded *separately* and concatenated with weights (NAML-style view fusion with
         frozen vectors): <u, v> = sum_v w_v <u_v, v_v>, so mean-pooling and late interaction both work on the fused table.
"""
from __future__ import annotations

import hashlib
import os
import re
import time

import numpy as np
import torch

import adapt

MODES = ("t", "ta", "cta", "ctae")
# (title, abstract, category) weights of the view fusion; they sum to 1, so the fused vector has unit norm
VIEW_WEIGHTS = ((0.5, 0.5, 0.0), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2), (0.5, 0.3, 0.2), (0.35, 0.35, 0.3))
MAX_ENTITIES = 8


# ------------------------------------------------------------------------------------------- text variants
def phrase(s):
    """'football_nfl' -> 'football nfl' (MIND subcategories are snake/kebab-case; some, like 'newsworld', are not separated at all)."""
    return re.sub(r"[_\-]+", " ", (s or "").strip()).strip()


def category_phrase(cat, sub):
    cat, sub = phrase(cat), phrase(sub)
    if not cat:
        return ""
    return f"Category: {cat}. Subcategory: {sub}." if sub and sub != cat else f"Category: {cat}."


def entity_phrase(labels, top=MAX_ENTITIES):
    return f"Entities: {'; '.join(labels[:top])}." if labels else ""


def _join(*parts):
    return " ".join(p for p in parts if p).strip()


def build_texts(meta, mode):
    """One string per article (same order as ``meta['ids']``)."""
    if mode == "t":
        return [t.strip() for t in meta["titles"]]
    if mode == "ta":
        return list(meta["text"])
    cat = [category_phrase(c, s) for c, s in zip(meta["cats"], meta["subcats"])]
    cta = [_join(c, text) for c, text in zip(cat, meta["text"])]
    if mode == "cta":
        return cta
    if mode == "ctae":
        return [_join(x, entity_phrase(ls)) for x, ls in zip(cta, meta["ent_labels"])]
    raise ValueError(f"unknown text mode {mode!r}; choose from {MODES + ('views',)}")


def view_texts(meta):
    """The three views of ``views``: title, abstract (the title again when the abstract is empty, so no vector is junk) and category phrase."""
    t = [x.strip() for x in meta["titles"]]
    a = [(x.strip() or tt) for x, tt in zip(meta["abstracts"], t)]
    c = [category_phrase(cc, s) or "Category: unknown." for cc, s in zip(meta["cats"], meta["subcats"])]
    return {"t": t, "a": a, "c": c}


def combine(parts, w):
    """Concatenate unit-norm view tables [n, d] scaled by sqrt(weight): the dot product of two fused vectors is the weighted sum of the
    per-view dot products.  Works on numpy arrays and torch tensors."""
    sq = [float(np.sqrt(x)) for x in w]
    if isinstance(parts[0], torch.Tensor):
        return torch.cat([s * p for s, p in zip(sq, parts)], dim=1)
    return np.concatenate([s * p for s, p in zip(sq, parts)], axis=1).astype(np.float32)


def audit(meta, data):
    """Facts about what the fields hold (logged by the stage, so a surprising result can be traced to the data)."""
    A = meta["abstracts"]
    hist_len = np.array([len(r[2]) for r in data.test])
    return {"articles": len(A), "news_columns": int(meta.get("n_cols", 0)),
            "abstract_empty": float(np.mean([not a.strip() for a in A])),
            "title_words": float(np.mean([len(t.split()) for t in meta["titles"]])),
            "abstract_words": float(np.mean([len(a.split()) for a in A])),
            "with_entity_labels": float(np.mean([bool(x) for x in meta["ent_labels"]])),
            "entities_per_article": float(np.mean([len(x) for x in meta["ent_labels"]])),
            "categories": len(set(meta["cats"])), "subcategories": len(set(meta["subcats"])),
            "test_history_mean": float(hist_len.mean()), "test_history_over_50": float((hist_len > 50).mean()),
            "test_history_max": int(hist_len.max())}


# ------------------------------------------------------------------------------------------- embedding with a content-keyed cache
def digest(texts):
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def embed_texts(texts, name, max_len, prefix, device, bs, cache_dir, dtype=torch.float32):
    """Frozen vectors ``[len(texts), d]``, cached on disk per (model, max_len, prefix, **the texts themselves**): a different text variant
    can never be served from another variant's cache.  Returns ``(vecs, seconds)`` (0 if cached)."""
    key = f"{name}|{max_len}|{prefix}|{len(texts)}|{digest(texts)}"
    path = os.path.join(cache_dir, "emb_cache_" + hashlib.sha1(key.encode()).hexdigest()[:12] + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        if str(z["key"]) == key:
            return z["vecs"], 0.0
    enc = adapt.HFEncoder(name, max_len=max_len, device=device, prefix=prefix, dtype=dtype)
    t0 = time.time()
    vecs = enc.encode(texts, bs=bs)
    secs = time.time() - t0
    del enc
    if device == "cuda":
        torch.cuda.empty_cache()
    if not np.isfinite(vecs).all():
        raise RuntimeError(f"{name}: non-finite embeddings (fp16 overflow?)")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, key=np.array(key), vecs=vecs)
    return vecs, secs


def embed_unique(embed, texts):
    """Embed each distinct string once (the category phrases are ~270 distinct strings for 65k articles) and broadcast."""
    uniq = sorted(set(texts))
    vecs = embed(uniq)
    index = {t: i for i, t in enumerate(uniq)}
    return vecs[[index[t] for t in texts]]
