"""Helpers shared by the suite and its stages."""
from __future__ import annotations

import numpy as np
import torch

import rep_eval as R

TASK = "Given the headlines a reader recently clicked, retrieve the news article they would most likely read next"


def log(*a):
    print(*a, flush=True)


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

    def texts(self):
        return ["; ".join(self.titles[i] for i in h) for h in self.uniq]

    def scorer(self, Q, E, split, device, ids=None):
        """``Q``: [n_unique, d] numpy; ``E``: candidate matrix tensor on ``device``;
        ``ids``: original row ids when scoring a subset EvalSet."""
        Qt = torch.from_numpy(np.concatenate([np.asarray(Q, np.float32), np.zeros((1, Q.shape[1]), np.float32)])).to(device)
        rq = self.rowq[split].copy()
        rq[rq < 0] = len(self.uniq)                     # empty history -> zero query (ties like the mean-pool control)
        rq_t = torch.from_numpy(rq).to(device)
        ids_t = None if ids is None else torch.from_numpy(np.asarray(ids)).to(device)

        def f(b):
            r = torch.as_tensor(b["rows"], device=device)
            r = r if ids_t is None else ids_t[r]
            return torch.einsum("bd,bcd->bc", Qt[rq_t[r]], E[b["cand"]])
        return f
