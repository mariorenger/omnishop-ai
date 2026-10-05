"""Representation-level evaluation on the V10 protocol.

Zero-shot scorers (no learned user model) isolate the quality of the *item
embeddings*: user = mean of the clicked-history vectors, score = dot product with
the candidate vector.  Everything is vectorised in torch, so a full 73k-impression
pass takes seconds; the metric definitions are identical to ``metrics.py``
(tie-aware AUC, MRR over clicked items, nDCG@k) -- ``selftest.py`` asserts this.

Besides the usual means we keep:
* per-impression arrays -> paired confidence intervals between two variants;
* per-*positive* AUC contributions -> cold/warm slices (is the clicked article one
  the training data never saw?), the quantity LLM embeddings are meant to help.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import metrics as ref_metrics  # vendored V10 metrics.py (tie handling reference)

METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


class EvalSet:
    """Padded, slate-size-sorted view of validation/test rows
    ``(impr_id, user, hist, cands, labels, pop)``."""

    def __init__(self, rows, max_hist=50, device="cpu", max_elems=3_000_000, max_batch=2048):
        self.rows, self.n = rows, len(rows)
        self.max_hist, self.device = max_hist, device
        self.max_elems, self.max_batch = max_elems, max_batch
        self.lens = np.array([len(r[3]) for r in rows], dtype=np.int64)
        self.order = np.argsort(self.lens, kind="stable")

    def __iter__(self):
        o, L, i = self.order, self.lens, 0
        while i < self.n:
            b = 1
            while (i + b < self.n and b < self.max_batch
                   and (b + 1) * max(int(L[o[i + b]]), 1) ** 2 <= self.max_elems):
                b += 1
            idx = o[i:i + b]
            i += b
            yield self._pack(idx)

    def _pack(self, idx):
        B, H = len(idx), self.max_hist
        C = max(int(self.lens[idx].max()), 1)
        hist = np.zeros((B, H), np.int64)
        hm = np.zeros((B, H), bool)
        cand = np.zeros((B, C), np.int64)
        cm = np.zeros((B, C), bool)
        lab = np.zeros((B, C), np.float32)
        pop = np.zeros((B, C), np.float32)
        user = np.zeros(B, np.int64)
        for k, r in enumerate(idx):
            _, u, h, c, y, pp = self.rows[r]
            h = list(h)[-H:]
            hist[k, :len(h)], hm[k, :len(h)] = h, True
            cand[k, :len(c)], cm[k, :len(c)], lab[k, :len(c)], pop[k, :len(c)] = c, True, y, pp
            user[k] = u
        t = lambda x: torch.from_numpy(x).to(self.device)
        return {"rows": idx, "hist": t(hist), "hist_mask": t(hm), "cand": t(cand),
                "cand_mask": t(cm), "labels": t(lab), "user": t(user), "pop": t(pop)}


def batch_metrics(S, Y, M):
    """Per-impression metrics for padded scores ``S`` [B,C], labels ``Y``, mask ``M``."""
    B, C = S.shape
    S = torch.nan_to_num(S.float(), nan=0.0)
    Sm = S.masked_fill(~M, float("-inf"))
    pos = (Y > 0.5) & M
    neg = (Y <= 0.5) & M
    # exact tie-aware AUC through pairwise comparison (B*C*C is bounded by EvalSet)
    win = (Sm[:, :, None] > Sm[:, None, :]).float() + 0.5 * (Sm[:, :, None] == Sm[:, None, :]).float()
    n_neg = neg.sum(1).float()
    eq = (Sm[:, :, None] == Sm[:, None, :]) & M[:, :, None] & M[:, None, :]
    tie = (eq.sum((1, 2)) > M.sum(1)).cpu()  # more equalities than the diagonal alone
    pos_auc = (win * neg[:, None, :].float()).sum(2) / n_neg.clamp(min=1)[:, None]
    P = pos.sum(1).float()
    auc = (pos_auc * pos.float()).sum(1) / P.clamp(min=1)
    auc = torch.where((P > 0) & (n_neg > 0), auc, torch.full_like(auc, float("nan")))
    # ranking metrics
    order = Sm.argsort(dim=1, descending=True)
    ys = pos.float().gather(1, order)
    ranks = torch.arange(1, C + 1, device=S.device).float()[None, :]
    mrr = (ys / ranks).sum(1) / P.clamp(min=1)
    disc = 1.0 / torch.log2(ranks + 1.0)

    def ndcg(k):
        dcg = (ys[:, :k] * disc[:, :k]).sum(1)
        cum = torch.cumsum(disc[0, :min(k, C)], 0)
        kk = torch.clamp(P, max=min(k, C)).long()
        ideal = torch.where(kk > 0, cum[(kk - 1).clamp(min=0)], torch.zeros_like(dcg))
        return torch.where(ideal > 0, dcg / ideal.clamp(min=1e-12), torch.zeros_like(dcg))

    return {"auc": auc, "mrr": mrr, "ndcg@5": ndcg(5), "ndcg@10": ndcg(10),
            "pos_auc": pos_auc, "pos_mask": pos, "tie": tie, "scores": Sm}


@torch.no_grad()
def evaluate(score_fn, evalset, seen=None):
    """Run ``score_fn(batch) -> [B,C]`` over an EvalSet.

    ``seen`` (bool array over news ids): news the training data observed.  Used for
    the cold/warm slices, which are over *clicked* articles only.
    """
    arr = {k: np.full(evalset.n, np.nan, np.float64) for k in METRICS}
    pnews, pauc, prow = [], [], []
    for b in evalset:
        m = batch_metrics(score_fn(b), b["labels"], b["cand_mask"])
        r = b["rows"]
        for k in METRICS:
            arr[k][r] = m[k].cpu().numpy()
        if m["tie"].any():  # exact ties (e.g. users with empty history): defer to V10 metrics.py
            n_valid = b["cand_mask"].sum(1).cpu().numpy()
            sc = m["scores"].cpu().numpy()
            lb = b["labels"].cpu().numpy()
            for k in np.nonzero(m["tie"].numpy())[0]:
                n = int(n_valid[k])
                s_k, y_k = sc[k, :n].astype(np.float32), lb[k, :n]
                arr["mrr"][r[k]] = ref_metrics.mrr_score(y_k, s_k)
                arr["ndcg@5"][r[k]] = ref_metrics.ndcg_score(y_k, s_k, 5)
                arr["ndcg@10"][r[k]] = ref_metrics.ndcg_score(y_k, s_k, 10)
            evalset.n_tied = getattr(evalset, "n_tied", 0) + int(m["tie"].sum())
        pm = m["pos_mask"].cpu().numpy()
        pnews.append(b["cand"].cpu().numpy()[pm])
        pauc.append(m["pos_auc"].cpu().numpy()[pm])
        prow.append(np.repeat(r, pm.sum(1)))
    out = {"arr": arr, "ids": np.asarray(getattr(evalset, "ids", np.arange(evalset.n))),
           "pos_news": np.concatenate(pnews), "pos_auc": np.concatenate(pauc),
           "pos_row": np.concatenate(prow)}
    out["mean"] = {"auc": float(np.nanmean(arr["auc"])),
                   **{k: float(np.mean(arr[k])) for k in METRICS[1:]}}
    if seen is not None:
        cold = ~seen[out["pos_news"]]
        out["cold"] = cold
        out["cold_frac"] = float(cold.mean()) if len(cold) else float("nan")
        out["mean"]["cold_auc"] = float(out["pos_auc"][cold].mean()) if cold.any() else float("nan")
        out["mean"]["warm_auc"] = float(out["pos_auc"][~cold].mean()) if (~cold).any() else float("nan")
    return out


# ------------------------------------------------------------------ scorers
def user_vectors(E, b):
    m = b["hist_mask"].unsqueeze(-1).to(E.dtype)
    return (E[b["hist"]] * m).sum(1) / m.sum(1).clamp(min=1)


def mean_pool_scorer(E):
    """User = mean clicked-history vector; score = dot with the candidate (== cosine
    ranking for L2-normalised E).  Same scorer as the V10 ``bge_zeroshot`` baseline."""
    def f(b):
        return torch.einsum("bd,bcd->bc", user_vectors(E, b), E[b["cand"]])
    return f


# --------------------------------------------------------------- statistics
def ratio_ci(d, groups=None, z=1.96):
    """CI for the mean of ``d`` with (optional) clustered observations: positives of the
    same impression are correlated, so the SE uses per-impression sums (ratio estimator)."""
    d = np.asarray(d, np.float64)
    ok = ~np.isnan(d)
    d = d[ok]
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan")
    if groups is None:
        m = d.mean()
        se = d.std(ddof=1) / math.sqrt(len(d))
        return float(m), float(m - z * se), float(m + z * se)
    g = np.asarray(groups)[ok]
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    s = np.bincount(inv, weights=d, minlength=G)
    n = np.bincount(inv, minlength=G).astype(np.float64)
    m = s.sum() / n.sum()
    resid = s - m * n
    var = (G / max(G - 1, 1)) * (resid ** 2).sum() / (n.sum() ** 2)
    se = math.sqrt(var)
    return float(m), float(m - z * se), float(m + z * se)


def _align(res, ref):
    """Indices that pair up the impressions two results have in common (full set vs a random subset)."""
    a, b = res["ids"], ref["ids"]
    if len(a) == len(b) and np.array_equal(a, b):
        return slice(None), slice(None)
    _, ia, ib = np.intersect1d(a, b, return_indices=True)
    return ia, ib


def compare(res, ref, key="ndcg@10"):
    """Paired difference ``res - ref`` of an impression-level metric with a 95% CI (common impressions only)."""
    ia, ib = _align(res, ref)
    return ratio_ci(res["arr"][key][ia] - ref["arr"][key][ib])


def compare_slice(res, ref, which="cold"):
    """Paired difference of the positive-level AUC on cold (or warm) clicked articles.  Clicked articles
    are matched on (original impression id, article id), so results over different EvalSets compare."""
    big = int(max(res["pos_news"].max(initial=0), ref["pos_news"].max(initial=0))) + 1
    ka = res["ids"][res["pos_row"]].astype(np.int64) * big + res["pos_news"]
    kb = ref["ids"][ref["pos_row"]].astype(np.int64) * big + ref["pos_news"]
    _, pa, pb = np.intersect1d(ka, kb, return_indices=True)
    sel = res["cold"][pa] if which == "cold" else ~res["cold"][pa]
    if not sel.any():
        return float("nan"), float("nan"), float("nan")
    d = (res["pos_auc"][pa] - ref["pos_auc"][pb])[sel]
    return ratio_ci(d, groups=(ka[pa] // big)[sel])


def verdict(lo, hi):
    if np.isnan(lo):
        return "n/a"
    if abs(lo) < 1e-7 and abs(hi) < 1e-7:                      # the variant reproduces the reference exactly (e.g. an adapter that kept epoch 0)
        return "identical"
    return "BETTER" if lo > 0 else ("WORSE" if hi < 0 else "no sig. difference")
