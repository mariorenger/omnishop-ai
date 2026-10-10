"""Downstream consistency check: does a better article representation survive a *learned* user model?

``LLMEncCA`` is a line-by-line clone of the V10 model of the same name (candidate-aware attention over the
history, on top of a frozen article embedding): 2-layer head to ``dim``, candidate projection, scaled dot-product
attention of the candidate over the clicked history, score = <attended user, candidate>.  Training follows the V10
protocol (Adam 1e-3, wd 1e-5, batch 64, 1 positive + 4 sampled negatives, cross-entropy, early stopping on
validation nDCG@10 with patience 2 / min_delta 0.001 / at least 3 epochs, at most 12).

Only the embedding table changes between runs, so differences are attributable to the representation.
"""
from __future__ import annotations

import math
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMEncCA(nn.Module):
    def __init__(self, emb, dim=64, dropout=0.2, device="cpu"):
        super().__init__()
        self.set_emb(emb, device)
        self.head = nn.Sequential(nn.Linear(self.emb.shape[1], dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout),
                                  nn.Linear(dim, dim))
        self.cand_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** 0.5
        self.to(device)

    def set_emb(self, emb, device=None):
        self.emb = torch.as_tensor(np.ascontiguousarray(emb, np.float32)).to(device or self.emb.device)

    def score(self, hist, hist_mask, cand):
        h, c = self.head(self.emb[hist]), self.head(self.emb[cand])
        att = torch.einsum("bcd,bhd->bch", self.cand_proj(c), h) / self.scale
        att = torch.softmax(att.masked_fill(~hist_mask[:, None, :], -1e9), -1)
        return (torch.einsum("bch,bhd->bcd", att, h) * c).sum(-1)


class LLMEncUser(nn.Module):
    """Same frozen article table and 2-layer head as ``LLMEncCA``, but the reader is encoded *without* looking at the candidate:
    ``add`` additive attention pooling (NAML-style), ``nrms`` multi-head self-attention + additive pooling, ``ff`` Fastformer block +
    additive pooling.  Only the user encoder differs from the V10 clone, so the comparison isolates the attention module."""

    def __init__(self, emb, kind="ff", dim=64, heads=2, dropout=0.2, device="cpu"):
        super().__init__()
        from refmodels import AddAttn, FastEncoder, SelfAttn
        self.set_emb(emb, device)
        self.kind = kind
        self.head = nn.Sequential(nn.Linear(self.emb.shape[1], dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.enc = {"add": None, "nrms": SelfAttn(dim, heads, dim // heads, out_proj=True),
                    "ff": FastEncoder(dim, dim, heads, 1, 50)}[kind]
        self.pool = AddAttn(dim, 200)
        self.to(device)

    def set_emb(self, emb, device=None):
        self.emb = torch.as_tensor(np.ascontiguousarray(emb, np.float32)).to(device or self.emb.device)

    def score(self, hist, hist_mask, cand):
        from refmodels import safe_mask
        h, c = self.head(self.emb[hist]), self.head(self.emb[cand])
        safe = safe_mask(hist_mask)
        if self.enc is not None:
            h = self.enc(h, safe)
        u = self.pool(h, safe) * hist_mask.any(-1, keepdim=True).to(h.dtype)
        return torch.einsum("bd,bcd->bc", u, c)


class LLMEncMI(nn.Module):
    """Multi-interest reader over the same frozen article table and 2-layer head as the other readers: ``K`` learned queries each attend
    over the clicked history (candidate-independent, in the spirit of multi-interest extractors such as MIND / ComiRec) and give one interest vector;
    a candidate is scored by the soft maximum over interests, ``tau * (logsumexp_k(<u_k, c> / tau) - log K)`` with a learned ``tau``.
    ``K=1`` is a plain single-vector attention reader (the control that isolates "several interests"); readers without history score 0.
    Queries start at N(0, 1) so the interests attend differently from step 0 (N(0, 1/d) queries gave a mean cosine of 0.9998 between the
    K interests at step 0 on a toy fixture, i.e. no room to specialise; N(0, 1) gives an attention total-variation distance of ~0.17).
    ``div > 0`` adds a disagreement penalty
    (in the spirit of disagreement regularisation for multi-head attention): ``div`` times the mean squared cosine between different
    interests of the same reader."""

    def __init__(self, emb, K=4, dim=64, dropout=0.2, device="cpu", div=0.0):
        super().__init__()
        self.set_emb(emb, device)
        self.K, self.div = int(K), float(div)
        self.head = nn.Sequential(nn.Linear(self.emb.shape[1], dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.queries = nn.Parameter(torch.randn(self.K, dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.scale = dim ** 0.5
        self._aux = None
        self.to(device)

    def set_emb(self, emb, device=None):
        self.emb = torch.as_tensor(np.ascontiguousarray(emb, np.float32)).to(device or self.emb.device)

    def interests(self, hist, hist_mask, with_att=False):
        """[B, K, dim] interest vectors (zero for a reader without history); ``with_att`` also returns the attention [B, K, H]."""
        from refmodels import safe_mask
        h = self.head(self.emb[hist])
        safe = safe_mask(hist_mask)
        att = torch.einsum("kd,bhd->bkh", self.queries, self.key(h)) / self.scale
        att = torch.softmax(att.masked_fill(~safe[:, None, :], -1e9), -1)
        u = torch.einsum("bkh,bhd->bkd", att, h) * hist_mask.any(-1)[:, None, None].to(h.dtype)
        return (u, att) if with_att else u

    @staticmethod
    def _pair_cos(u, has):
        """Per reader (with history): mean squared cosine between different interests -> [B]."""
        un = F.normalize(u, dim=-1)
        cos = torch.einsum("bkd,bld->bkl", un, un) ** 2
        K = u.shape[1]
        return (cos.sum((1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(1)) / (K * (K - 1)) * has.to(u.dtype)

    def score(self, hist, hist_mask, cand):
        u = self.interests(hist, hist_mask)
        s = torch.einsum("bkd,bcd->bkc", u, self.head(self.emb[cand]))
        if self.K == 1:
            return s[:, 0]
        if self.training and self.div > 0:
            has = hist_mask.any(-1)
            self._aux = self.div * self._pair_cos(u, has).sum() / has.sum().clamp(min=1)
        tau = self.log_tau.exp().clamp(min=0.05)
        return tau * (torch.logsumexp(s / tau, 1) - math.log(self.K))

    def aux_loss(self):
        """Regulariser of the last ``score`` call (0 when ``div`` is 0); the trainer adds it to the loss."""
        aux, self._aux = self._aux, None
        return aux if aux is not None else 0.0

    @torch.no_grad()
    def diversity(self, hist, hist_mask):
        """Collapse diagnostics over the readers that have a history: ``interest_cos`` = mean pairwise cosine between a reader's interest
        vectors (1 = identical; note the shared head output keeps it high even for different attention), ``attn_tv`` = mean pairwise
        total-variation distance between the interests' attention distributions (0 = every interest attends identically, 1 = disjoint)."""
        has = hist_mask.any(-1)
        if self.K < 2 or not bool(has.any()):
            return {"interest_cos": float("nan"), "attn_tv": float("nan")}
        u, att = self.interests(hist, hist_mask, with_att=True)
        u, att = F.normalize(u[has], dim=-1), att[has]
        cos = torch.einsum("bkd,bld->bkl", u, u)
        pairs = self.K * (self.K - 1)
        tv = 0.5 * (att[:, :, None, :] - att[:, None, :, :]).abs().sum(-1)                   # [B, K, K]
        return {"interest_cos": float(((cos.sum((1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(1)) / pairs).mean()),
                "attn_tv": float((tv.sum((1, 2)) / pairs).mean())}


def _mi_spec(kind):
    """``mi4`` -> (4, 0.0); ``mi4d`` -> (4, 0.1) (with the disagreement regulariser); anything else -> None."""
    m = re.fullmatch(r"mi(\d+)(d?)", kind)
    return None if m is None or int(m.group(1)) < 1 else (int(m.group(1)), 0.1 if m.group(2) else 0.0)


def is_reader(kind):
    """``ca`` | ``add`` | ``nrms`` | ``ff`` | ``mi<K>`` (K learned interests, e.g. ``mi4``) | ``mi<K>d`` (same with the disagreement regulariser)."""
    return kind in ("ca", "add", "nrms", "ff") or _mi_spec(kind) is not None


def make_model(kind, emb, dim=64, dropout=0.2, device="cpu"):
    """``ca``: the V10 clone; ``add`` / ``nrms`` / ``ff``: candidate-independent readers (see ``LLMEncUser``); ``mi<K>[d]``: ``LLMEncMI``."""
    if kind == "ca":
        return LLMEncCA(emb, dim, dropout, device)
    spec = _mi_spec(kind)
    if spec is not None:
        return LLMEncMI(emb, spec[0], dim, dropout, device, spec[1])
    return LLMEncUser(emb, kind, dim, 2, dropout, device)


def pack_train(rows, max_hist=50):
    """train_core rows ``(user, hist, pos, negs)`` -> arrays (hist [N,H], mask [N,H], cands [N,1+n_neg], pos first)."""
    N = len(rows)
    H = np.zeros((N, max_hist), np.int64)
    M = np.zeros((N, max_hist), bool)
    C = np.zeros((N, 1 + len(rows[0][3])), np.int64)
    for i, (_u, hist, pos, negs) in enumerate(rows):
        h = list(hist)[-max_hist:]
        H[i, :len(h)], M[i, :len(h)] = h, True
        C[i] = [pos] + list(negs)
    return H, M, C


def train_user_model(emb, pack, val_fn, seed=0, dim=64, dropout=0.2, lr=1e-3, wd=1e-5, bs=64, max_epochs=12,
                     min_epochs=3, patience=2, min_delta=1e-3, device="cpu", deadline=None, log=print, kind="ca"):
    """``val_fn(model) -> float`` (validation nDCG@10, evaluated with the embedding table valid for the validation
    period).  ``kind``: user encoder, see ``make_model``.  Returns ``(model restored to its best epoch, info)``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(kind, emb, dim, dropout, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    H, M, C = (torch.from_numpy(a).to(device) for a in pack)
    N = len(H)
    target = torch.zeros(bs, dtype=torch.long, device=device)
    best, best_state, stale, hist, t0 = None, None, 0, [], time.time()
    for epoch in range(1, max_epochs + 1):
        if deadline is not None and time.time() > deadline and best_state is not None:
            log(f"      [user-model] deadline reached after epoch {epoch - 1}")
            break
        model.train()
        perm = torch.from_numpy(rng.permutation(N)).to(device)
        tot, nb = 0.0, 0
        for s in range(0, N, bs):
            ix = perm[s:s + bs]
            logits = model.score(H[ix], M[ix], C[ix])
            loss = F.cross_entropy(logits, target[:len(ix)])
            if hasattr(model, "aux_loss"):
                loss = loss + model.aux_loss()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        val = val_fn(model)
        hist.append((epoch, tot / max(nb, 1), val))
        improved = best is None or val > best + min_delta
        if improved:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch >= min_epochs:
            stale += 1
        log(f"      [user-model] epoch {epoch:02d} loss={tot / max(nb, 1):.4f} val nDCG@10={val:.4f}")
        if epoch >= min_epochs and stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val": float(best), "epochs": len(hist), "train_s": time.time() - t0}


def scorer(model, emb=None):
    """Slate scorer for ``rep_eval.evaluate``; ``emb`` swaps the embedding table (e.g. test-period retrieval memory)."""
    if emb is not None:
        model.set_emb(emb)

    def f(b):
        return model.score(b["hist"], b["hist_mask"], b["cand"])
    return f
