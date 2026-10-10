"""H8: user pooling over frozen item vectors -- single mean vector vs late interaction / recency.

The V10 control collapses the whole history into ONE mean vector.  Multi-interest and late-interaction
(ColBERT-style) matching instead score a candidate against *individual* history items, which should help
readers with several interests.  Everything here is zero-shot (no parameters), so the comparison with the
mean-pool control isolates the aggregation rule.
"""
from __future__ import annotations

import torch


def _sims(E, b):
    """cosine/dot similarity between every candidate and every history item -> [B, C, H]."""
    return torch.einsum("bcd,bhd->bch", E[b["cand"]], E[b["hist"]])


def recency_scorer(E, lam):
    """Mean of the history with weights exp(-lam * age); age 0 = most recent click (history is chronological)."""
    def f(b):
        mask = b["hist_mask"].float()
        n = mask.sum(1, keepdim=True)
        age = (n - 1 - torch.arange(mask.shape[1], device=mask.device)[None, :]).clamp(min=0)
        w = torch.exp(-lam * age) * mask
        u = (E[b["hist"]] * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-8)
        return torch.einsum("bd,bcd->bc", u, E[b["cand"]])
    return f


def interest_means(E, b, K, iters=3):
    """Group every reader's clicked items into at most ``K`` interests with spherical k-means (cosine assignment), started
    deterministically from the most recent click and then, repeatedly, the click least similar to the centres chosen so far.
    Returns ``(means [B,K,d], share [B,K])``: the plain mean vector of each group and the share of the reader's clicks in it
    (0 for an empty group).  ``K=1`` gives exactly the mean-pool user vector."""
    h = E[b["hist"]]                                                       # [B,H,d]
    mask = b["hist_mask"]
    B, H, d = h.shape
    ar = torch.arange(B, device=h.device)
    n = mask.sum(1).clamp(min=1).to(h.dtype)
    last = (mask.long() * torch.arange(1, H + 1, device=h.device)).argmax(1)       # history is chronological and left-aligned
    cen = h.new_zeros(B, K, d)
    cen[:, 0] = h[ar, last]
    nearest = torch.einsum("bhd,bd->bh", h, cen[:, 0])                     # similarity of every click to its closest centre so far (max over centres)
    for k in range(1, K):
        far = nearest.masked_fill(~mask, float("inf")).argmin(1)
        cen[:, k] = h[ar, far]
        nearest = torch.maximum(nearest, torch.einsum("bhd,bd->bh", h, cen[:, k]))
    for _ in range(iters):
        asg = torch.einsum("bhd,bkd->bhk", h, cen).argmax(-1)
        one = torch.nn.functional.one_hot(asg, K).to(h.dtype) * mask.unsqueeze(-1).to(h.dtype)
        sums = torch.einsum("bhk,bhd->bkd", one, h)
        cen = torch.where((one.sum(1) > 0).unsqueeze(-1), torch.nn.functional.normalize(sums, dim=-1), cen)
    asg = torch.einsum("bhd,bkd->bhk", h, cen).argmax(-1)
    one = torch.nn.functional.one_hot(asg, K).to(h.dtype) * mask.unsqueeze(-1).to(h.dtype)
    cnt = one.sum(1)
    return torch.einsum("bhk,bhd->bkd", one, h) / cnt.clamp(min=1).unsqueeze(-1), cnt / n.unsqueeze(-1)


def cluster_scorer(E, K, tau, iters=3):
    """Zero-shot multi-interest: score(c) = tau * logsumexp_k( <c, mean of interest k> / tau + log share_k ).  ``K=1`` is the mean-pool
    control (exactly); ``K >= the number of clicks`` (every click its own interest) is the ``lse`` late interaction of ``late_scorer`` with
    the same ``tau`` -- so the family interpolates between the two ends that are already measured.  Readers without history score 0."""
    def f(b):
        has = b["hist_mask"].any(1)[:, None]
        mu, share = interest_means(E, b, K, iters)
        sim = torch.einsum("bcd,bkd->bck", E[b["cand"]], mu)              # [B,C,K]
        s = tau * torch.logsumexp(sim / tau + torch.log(share).unsqueeze(1), -1)
        return torch.where(has, s, torch.zeros_like(s))
    return f


def late_scorer(E, kind, param=None):
    """Late interaction: ``max`` (MaxSim), ``topk`` (mean of the k most similar history items) or
    ``lse`` (soft maximum with temperature ``param``).  Users without history score 0 (ties, like the control)."""
    def f(b):
        mask = b["hist_mask"]
        has = mask.any(1)[:, None]
        sim = _sims(E, b)
        n = mask.sum(1).clamp(min=1).float()[:, None]
        if kind == "max":
            s = sim.masked_fill(~mask[:, None, :], float("-inf")).max(-1).values
        elif kind == "topk":
            k = int(param)
            vals = sim.masked_fill(~mask[:, None, :], -1e4).topk(min(k, sim.shape[-1]), -1).values      # [B,C,k]
            cnt = n.clamp(max=k)                                                                         # items actually available
            keep = (torch.arange(vals.shape[-1], device=vals.device)[None, None, :] < cnt[:, :, None]).float()
            s = (vals * keep).sum(-1) / cnt
        elif kind == "lse":
            t = float(param)
            s = t * (torch.logsumexp(sim.masked_fill(~mask[:, None, :], float("-inf")) / t, -1) - torch.log(n))
        else:
            raise ValueError(kind)
        return torch.where(has, s, torch.zeros_like(s))
    return f
