"""Helpers shared by the suite and its stages."""
from __future__ import annotations

import os
import time

import numpy as np
import torch

import rep_eval as R

TASK = "Given the headlines a reader recently clicked, retrieve the news article they would most likely read next"


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ wall-clock budget
class Clock:
    """One wall-clock budget shared by every ``suite.main`` call of a notebook.  Long loops receive
    ``deadline`` (absolute ``time.time()``) and stop gracefully, so a slow GPU degrades the run instead of killing it."""

    def __init__(self):
        self.t0, self.budget = time.time(), None

    def start(self, hours):
        """The first call of a notebook fixes the budget; later ``suite.main`` calls share it."""
        if self.budget is None and hours:
            self.t0, self.budget = time.time(), float(hours) * 3600.0

    def reset(self):
        self.t0, self.budget = time.time(), None

    @property
    def deadline(self):
        return None if self.budget is None else self.t0 + self.budget

    def left(self):
        return float("inf") if self.budget is None else self.t0 + self.budget - time.time()

    def elapsed(self):
        return time.time() - self.t0


CLOCK = Clock()


def time_left():
    return CLOCK.left()


def out_of_time(margin=0.0):
    return CLOCK.left() < margin


# ------------------------------------------------------------------ prompts / pooling of results
def query_prefix(name, fmt="auto", instruct=True):
    low = name.lower()
    if fmt == "auto":
        fmt = "qwen3" if "qwen3" in low else "bge" if "bge" in low else "e5" if "e5" in low else "plain"
    if fmt == "qwen3":
        return f"Instruct: {TASK}\nQuery:" if instruct else ""
    if fmt == "bge":
        return "Represent this sentence for searching relevant passages: " if instruct else ""
    if fmt == "e5":
        return "query: "
    return ""


def avg_results(rs):
    """Average several seeds of the same variant (per-impression arrays and slice values)."""
    if len(rs) == 1:
        return rs[0]
    out = dict(rs[0])
    out["arr"] = {k: np.nanmean([r["arr"][k] for r in rs], axis=0) for k in R.METRICS}
    out["pos_auc"] = np.mean([r["pos_auc"] for r in rs], axis=0)
    m = {"auc": float(np.nanmean(out["arr"]["auc"])), **{k: float(np.mean(out["arr"][k])) for k in R.METRICS[1:]}}
    if "cold" in out:
        c = out["cold"]
        m["cold_auc"] = float(out["pos_auc"][c].mean()) if c.any() else float("nan")
        m["warm_auc"] = float(out["pos_auc"][~c].mean()) if (~c).any() else float("nan")
    out["mean"] = m
    return out


# ------------------------------------------------------------------ Hugging Face loading
def load_hf(auto_cls, name, dtype=None, trust_remote_code=False):
    """``from_pretrained`` that copes with the ``torch_dtype`` -> ``dtype`` rename between transformers versions
    and always ends up in the requested dtype."""
    kw = {"trust_remote_code": trust_remote_code}
    if dtype is None:
        return auto_cls.from_pretrained(name, **kw)
    try:
        model = auto_cls.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        model = auto_cls.from_pretrained(name, torch_dtype=dtype, **kw)
    return model if model.dtype == dtype else model.to(dtype)


def decoder_only(name):
    """Last-token-pooled embedding models are decoder-only LLMs (Qwen3-Embedding, e5-mistral, ...)."""
    import adapt
    return adapt.guess_pooling(name) == "last"


# ------------------------------------------------------------------ evaluation subsets
def subset_evalset(rows, ids, device="cpu"):
    """EvalSet over ``rows[ids]`` that remembers the original positions (``.ids``) so that results over
    different subsets of the same split can be paired impression by impression."""
    ids = np.asarray(ids, np.int64)
    es = R.EvalSet([rows[i] for i in ids], device=device)
    es.ids = ids
    return es


def random_subset(n_total, n, seed=0):
    """Nested random subsets: ``random_subset(N, a) ⊂ random_subset(N, b)`` for a <= b (same seed)."""
    n = min(int(n), n_total)
    return np.sort(np.random.default_rng(seed).permutation(n_total)[:n])


# ------------------------------------------------------------------ score fusion and caching
def zscore(x, mask):
    """Standardise scores within each impression over its real candidates (same as V10 ``_znorm``)."""
    m = mask.float()
    n = m.sum(-1, keepdim=True).clamp(min=1)
    mu = (x * m).sum(-1, keepdim=True) / n
    sd = (((x - mu) * m) ** 2).sum(-1, keepdim=True).div(n).sqrt()
    return (x - mu) / (sd + 1e-6)


def fuse(parts):
    """Weighted sum of per-impression z-scored scorers: ``parts = [(scorer, weight), ...]``."""
    def f(b):
        return sum(w * zscore(s(b).float(), b["cand_mask"]) for s, w in parts)
    return f


class ScoreCache:
    """Raw per-candidate scores of an expensive scorer (SID language model, LLM tower, LLM judge), computed once
    per EvalSet so that fusion-weight grids are free."""

    def __init__(self, evalset, score_fn):
        self.n, self.rows = evalset.n, [None] * evalset.n
        for b in evalset:
            s = score_fn(b).float().cpu().numpy()
            cm = b["cand_mask"].cpu().numpy()
            for k, r in enumerate(b["rows"]):
                self.rows[r] = s[k, :int(cm[k].sum())].copy()

    def scorer(self, device, shift=None):
        """``shift``: optional callable ``(batch) -> [B,C]`` tensor subtracted from the cached scores."""
        def f(b):
            C = b["cand"].shape[1]
            out = np.zeros((len(b["rows"]), C), np.float32)
            for k, r in enumerate(b["rows"]):
                v = self.rows[r]
                out[k, :len(v)] = v
            out = torch.from_numpy(out).to(device)
            return out if shift is None else out - shift(b)
        return f


# ------------------------------------------------------------------ history-as-text queries
class HistQueries:
    """Unique last-K histories of validation/test rows, their query texts, and a scorer
    ``score = <query vector, candidate vector>`` (history-as-text user representation)."""

    def __init__(self, data, titles, K):
        self.titles, self.K = titles, K
        self.keys, self.rowq = {}, {}
        for sp, rows in (("val", data.validation), ("test", data.test)):
            rq = []
            for r in rows:
                h = tuple(r[2][-K:])
                rq.append(self.keys.setdefault(h, len(self.keys)) if h else -1)
            self.rowq[sp] = np.array(rq, np.int64)
        self.uniq = sorted(self.keys, key=self.keys.get)

    def needed(self, split, ids=None):
        """Indices of the unique queries used by the rows ``ids`` (default: every row) of a split."""
        rq = self.rowq[split] if ids is None else self.rowq[split][np.asarray(ids)]
        return np.unique(rq[rq >= 0])

    def texts(self, qidx=None):
        qidx = range(len(self.uniq)) if qidx is None else qidx
        return ["; ".join(self.titles[i] for i in self.uniq[q]) for q in qidx]

    def alloc(self, d):
        return np.zeros((max(len(self.uniq), 1), d), np.float32)

    def scorer(self, Q, E, split, device, ids=None):
        """``Q``: [n_unique, d] numpy; ``E``: candidate matrix tensor on ``device``;
        ``ids``: original row ids when scoring a subset EvalSet."""
        Qt = torch.from_numpy(np.concatenate([np.asarray(Q, np.float32), np.zeros((1, Q.shape[1]), np.float32)])).to(device)
        rq = self.rowq[split].copy()
        rq[rq < 0] = len(Q)                              # empty history -> zero query (ties like the mean-pool control)
        rq_t = torch.from_numpy(rq).to(device)
        ids_t = None if ids is None else torch.from_numpy(np.asarray(ids)).to(device)

        def f(b):
            r = torch.as_tensor(b["rows"], device=device)
            r = r if ids_t is None else ids_t[r]
            return torch.einsum("bd,bcd->bc", Qt[rq_t[r]], E[b["cand"]])
        return f


def atomic_write(path, data, mode="w"):
    tmp = path + ".tmp"
    with open(tmp, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
        f.write(data)
    os.replace(tmp, path)
