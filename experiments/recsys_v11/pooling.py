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
