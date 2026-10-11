"""History / slate features of every candidate, computed from the frozen article vectors (no parameters, nothing learned, no label).

The linear supermodel sees one number per candidate and component (similarity to the user's mean vector, to the closest click, ...).  The
stacker (``stacker.py``) can use more of what the frozen LLM vectors say about the *situation* a candidate is in:

  ``last1`` / ``last3``   cosine similarity to the most recent click / mean cosine to the three most recent ones (what the reader is reading *now*);
  ``in_hist``             the article is already in the reader's click history;
  ``cat_share`` / ``sub_share``   share of the reader's clicks in the candidate's category / sub-category (the ``cta`` text variants put the
                          category into the *vector*; this is the same fact as a plain number);
  ``cat_last``            the candidate has the category of the reader's most recent click;
  ``n_hist`` / ``n_cand`` length of the history / size of the slate (how much the other columns can be trusted);
  ``slate_mean`` / ``slate_max``  mean / maximum cosine to the OTHER candidates of the slate (an outlier among the candidates, or one of several
                          near-duplicates about the same story);
  ``slate_cat``           share of the slate in the candidate's category.

All of them are functions of the impression's history, its candidate list and the frozen vectors only.
"""
from __future__ import annotations

import numpy as np
import torch

NAMES = ("last1", "last3", "in_hist", "cat_share", "sub_share", "cat_last", "n_hist", "n_cand", "slate_mean", "slate_max", "slate_cat")


def code_arrays(meta, n_news):
    """Integer codes of the category / sub-category of every article (row 0 = padding = code 0, codes start at 1)."""
    out = []
    for key in ("cats", "subcats"):
        vals = list(meta[key])
        codes = {v: i + 1 for i, v in enumerate(sorted(set(vals)))}
        arr = np.zeros(n_news, np.int64)
        arr[1:1 + len(vals)] = [codes[v] for v in vals]
        out.append(arr)
    return out


@torch.no_grad()
def batch_features(b, E, cat, sub):
    """``{name: [B, C] float32}`` for one EvalSet batch.  ``E``: [n_news, d] vectors (L2-normalised, row 0 = padding); ``cat`` / ``sub``: long tensors
    [n_news] of category / sub-category codes on the same device.  Entries of padded candidates are 0."""
    cand, hist, hm, cm = b["cand"], b["hist"], b["hist_mask"], b["cand_mask"]
    B, C = cand.shape
    H = hist.shape[1]
    dev = cand.device
    n = hm.sum(1)                                                              # [B] clicks in the history (chronological, left-aligned)
    has = (n > 0)[:, None]
    nf = n.clamp(min=1).to(torch.float32)[:, None]
    sim = torch.einsum("bcd,bhd->bch", E[cand], E[hist])                       # [B, C, H]
    pos = torch.arange(H, device=dev)[None, :]
    last = (n - 1).clamp(min=0)
    out = {}
    out["last1"] = torch.where(has, sim.gather(2, last[:, None, None].expand(B, C, 1)).squeeze(-1), torch.zeros(B, C, device=dev))
    k3 = ((pos >= (n[:, None] - 3)) & hm).to(torch.float32)                    # the (up to) three most recent clicks
    out["last3"] = torch.where(has, (sim * k3[:, None, :]).sum(-1) / k3.sum(1).clamp(min=1)[:, None], torch.zeros(B, C, device=dev))
    same_item = (cand[:, :, None] == hist[:, None, :]) & hm[:, None, :]
    out["in_hist"] = same_item.any(-1).to(torch.float32)
    for key, codes in (("cat_share", cat), ("sub_share", sub)):
        same = (codes[cand][:, :, None] == codes[hist][:, None, :]) & hm[:, None, :]
        out[key] = same.sum(-1).to(torch.float32) / nf
    last_code = cat[hist.gather(1, last[:, None])].squeeze(1)                  # category of the most recent click
    out["cat_last"] = ((cat[cand] == last_code[:, None]) & has).to(torch.float32)
    out["n_hist"] = n.to(torch.float32)[:, None].expand(B, C)
    ncand = cm.sum(1).to(torch.float32)
    out["n_cand"] = ncand[:, None].expand(B, C)
    cc = torch.einsum("bcd,bkd->bck", E[cand], E[cand])                        # [B, C, C] similarity among the candidates of the slate
    other = cm[:, None, :] & ~torch.eye(C, dtype=torch.bool, device=dev)[None]
    cnt = other.sum(-1).clamp(min=1).to(torch.float32)
    out["slate_mean"] = (cc * other).sum(-1) / cnt
    out["slate_max"] = cc.masked_fill(~other, float("-inf")).max(-1).values.clamp(min=-1.0)
    out["slate_max"] = torch.where(other.any(-1), out["slate_max"], torch.zeros_like(out["slate_max"]))
    samec = (cat[cand][:, :, None] == cat[cand][:, None, :]) & cm[:, None, :]
    out["slate_cat"] = samec.sum(-1).to(torch.float32) / ncand.clamp(min=1)[:, None]
    return {k: torch.where(cm, v.to(torch.float32), torch.zeros_like(v, dtype=torch.float32)) for k, v in out.items()}


def reference_features(hist, cands, vec, cat, sub):
    """Plain-python version of ``batch_features`` for ONE impression (``hist``: chronological list of article ids, ``cands``: list of ids,
    ``vec``: numpy [n_news, d], ``cat`` / ``sub``: numpy code arrays).  Used by the tests as the definition the vectorised code must agree with."""
    n, C = len(hist), len(cands)
    out = {k: np.zeros(C, np.float64) for k in NAMES}
    for j, c in enumerate(cands):
        if n:
            sims = [float(vec[c] @ vec[h]) for h in hist]
            out["last1"][j] = sims[-1]
            out["last3"][j] = float(np.mean(sims[-3:]))
            out["cat_share"][j] = sum(cat[h] == cat[c] for h in hist) / n
            out["sub_share"][j] = sum(sub[h] == sub[c] for h in hist) / n
            out["cat_last"][j] = float(cat[hist[-1]] == cat[c])
        out["in_hist"][j] = float(c in hist)
        out["n_hist"][j] = n
        out["n_cand"][j] = C
        others = [k for k in range(C) if k != j]
        if others:
            ss = [float(vec[c] @ vec[cands[k]]) for k in others]
            out["slate_mean"][j] = float(np.mean(ss))
            out["slate_max"][j] = max(ss)
        out["slate_cat"][j] = sum(cat[x] == cat[c] for x in cands) / C
    return out
