"""H7: Semantic IDs -- a zero-shot SID profile scorer and a TIGER-style generative model used as a slate ranker.

Why a *ranker*: MIND asks us to order a **given** slate, so generative retrieval (decode the next ID) does not
apply.  The SID language model instead scores every candidate by the likelihood of its code sequence given the
reading history, ``log P(codes(c) | history)``, optionally minus the history-free ``log P(codes(c))`` (a PMI
score that removes the "popular semantic region" prior).  This is the generative-recommendation paradigm
(TIGER, NeurIPS'23 -> LETTER, SIGIR'25 -> LSIG, WWW'26) adapted to slate ranking.

Tokenizer: RQ-KMeans (residual k-means), the simple tokenizer used by recent industrial generative recommenders.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- tokenizer
def _assign(X, C, chunk=65536):
    c2 = (C * C).sum(1)
    out = []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        out.append(((x * x).sum(1, keepdim=True) - 2 * x @ C.T + c2[None, :]).argmin(1))
    return torch.cat(out)


def kmeans(X, K, iters=25, seed=0):
    """Lloyd's algorithm on a tensor; empty clusters are re-seeded from random points."""
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    K = min(K, n)
    C = X[torch.randperm(n, generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        a = _assign(X, C)
        S = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=K).float()
        C = S / cnt.clamp(min=1)[:, None]
        empty = cnt == 0
        if empty.any():
            C[empty] = X[torch.randint(0, n, (int(empty.sum()),), generator=g).to(X.device)]
    return C


def rq_kmeans(E, fit_mask, L=3, K=256, seed=0, device="cpu"):
    """Residual k-means: level l quantises what levels < l left over.  Codebooks are fitted on ``fit_mask`` rows
    (content-only, causal); every article gets codes by nearest-centroid.  Returns int64 [n, L]."""
    X = torch.from_numpy(np.ascontiguousarray(E, np.float32)).to(device)
    fit = torch.from_numpy(np.nonzero(fit_mask)[0]).to(device)
    R = X.clone()
    codes = torch.zeros(len(E), L, dtype=torch.long, device=device)
    for l in range(L):
        C = kmeans(R[fit], K, seed=seed + l)
        a = _assign(R, C)
        codes[:, l] = a
        R = R - C[a]
    return codes.cpu().numpy().astype(np.int64)


def prefix_ids(codes, K):
    """Integer id of the length-l code prefix for l = 1..L -> int64 [L, n]."""
    out, p = [], np.zeros(len(codes), np.int64)
    for l in range(codes.shape[1]):
        p = p * K + codes[:, l]
        out.append(p.copy())
    return np.stack(out)


def sid_profile_scorer(codes, K, levels, device="cpu"):
    """Zero-shot: score = sum over the chosen prefix levels of the share of the history that falls in the
    candidate's code prefix (coarse levels generalise to cold articles, fine levels are specific)."""
    pref = torch.from_numpy(prefix_ids(codes, K)).to(device)

    def f(b):
        mask = b["hist_mask"]
        n = mask.sum(1).clamp(min=1).float()[:, None]
        s = 0.0
        for l in levels:
            ph, pc = pref[l - 1][b["hist"]], pref[l - 1][b["cand"]]
            s = s + ((pc[:, :, None] == ph[:, None, :]) & mask[:, None, :]).float().sum(-1) / n
        return s
    return f


# ------------------------------------------------------------------- model
class _Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads, self.drop = heads, drop

    def forward(self, x, allow):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class SIDGPT(nn.Module):
    """Causal transformer over [BOS] item_1 codes ... item_n codes (left padded).  Positions are counted over
    valid tokens only, so the output does not depend on how much padding precedes the sequence."""

    def __init__(self, K, L, d=256, layers=4, heads=4, max_items=32, drop=0.1):
        super().__init__()
        self.K, self.L = K, L
        self.PAD, self.BOS = L * K, L * K + 1
        self.tok = nn.Embedding(L * K + 2, d)
        self.pos = nn.Embedding(1 + max_items * L, d)
        self.lvl = nn.Embedding(L + 1, d)
        self.blocks = nn.ModuleList(_Block(d, heads, drop) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, L * K, bias=False)

    def forward(self, tokens, last_n=None):
        valid = tokens != self.PAD
        pos = (valid.long().cumsum(1) - 1).clamp(min=0)
        lvl = torch.where(tokens < self.L * self.K, tokens // self.K, torch.full_like(tokens, self.L))
        x = self.tok(tokens) + self.pos(pos) + self.lvl(lvl)
        T = tokens.shape[1]
        allow = torch.ones(T, T, dtype=torch.bool, device=tokens.device).tril()[None, None] & valid[:, None, None, :]
        allow = allow | torch.eye(T, dtype=torch.bool, device=tokens.device)[None, None]   # no fully-masked rows
        for blk in self.blocks:
            x = blk(x, allow)
        x = self.ln(x if last_n is None else x[:, -last_n:])
        return self.head(x)                                     # [B, T or last_n, L*K] logits over code tokens

    def code_logprob(self, logits, targets):
        """log P(target token) with the softmax restricted to the target's own level (constrained decoding).
        ``targets``: code tokens (level*K + code), same leading shape as ``logits[..., 0]``."""
        lvl, code = targets // self.K, targets % self.K
        lp = F.log_softmax(logits.view(*logits.shape[:-1], self.L, self.K), dim=-1)          # [..., L, K]
        sel = lp.gather(-2, lvl[..., None, None].expand(*lvl.shape, 1, self.K)).squeeze(-2)  # [..., K]
        return sel.gather(-1, code[..., None]).squeeze(-1)


# ---------------------------------------------------------------- sequences
def right_align(hist, mask, n_max):
    """Keep the last ``n_max`` valid history items, right aligned (padding on the left)."""
    n = mask.sum(1)
    take = n.clamp(max=n_max)
    j = torch.arange(n_max, device=hist.device)[None, :]
    valid = j >= (n_max - take)[:, None]
    src = ((n - take)[:, None] + (j - (n_max - take)[:, None])).clamp(0, hist.shape[1] - 1)
    return hist.gather(1, src) * valid, valid


def item_tokens(codes_t, items, valid, K, L):
    """[..., N] article ids -> [..., N*L] code tokens (PAD where the item is not valid)."""
    off = torch.arange(L, device=items.device) * K
    tok = codes_t[items] + off                                    # [..., N, L]
    tok = torch.where(valid[..., None], tok, torch.full_like(tok, L * K))
    return tok.flatten(-2)


def with_bos(tok, valid_items, L, bos, pad):
    """[..., N*L] code tokens (PAD where an item is invalid, invalid items on the left) -> [..., 1+N*L] where BOS sits
    immediately before the first valid token: [PAD..., BOS, item_1 codes, ...].  Putting BOS anywhere else would make
    the position that predicts the first code a padding position."""
    pad_len = (~valid_items).sum(-1) * L
    full = torch.cat([torch.full_like(tok[..., :1], pad), tok], -1)
    return full.scatter(-1, pad_len[..., None], torch.full_like(pad_len[..., None], bos))


def make_training_arrays(rows, n_max):
    """(history items right aligned, target) from train_core rows ``(user, hist, pos, negs)``."""
    H = np.zeros((len(rows), n_max), np.int64)
    M = np.zeros((len(rows), n_max), bool)
    Y = np.zeros(len(rows), np.int64)
    for i, (_u, hist, pos, _n) in enumerate(rows):
        h = list(hist)[-n_max:]
        if h:
            H[i, n_max - len(h):], M[i, n_max - len(h):] = h, True
        Y[i] = pos
    keep = M.any(1)
    return H[keep], M[keep], Y[keep]


def train_sid_gpt(codes, K, rows, n_max=30, epochs=3, bs=128, lr=1e-3, hist_w=0.5, seed=0, device="cpu",
                  val_fn=None, d=256, layers=4, heads=4, max_steps=None, deadline=None, log=print):
    """Next-code-token language modelling over reading sequences (history items weighted ``hist_w``, the
    clicked target 1.0).  Validation (``val_fn(model) -> float``, higher is better) picks the epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    L = codes.shape[1]
    codes_t = torch.from_numpy(codes).to(device)
    model = SIDGPT(K, L, d, layers, heads, max_items=n_max + 2).to(device)
    H, M, Y = make_training_arrays(rows, n_max)
    total = max(1, epochs * math.ceil(len(H) / bs))
    if max_steps:
        total = min(total, max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score, best_state, best_ep, step = (val_fn(model) if val_fn else -math.inf), None, 0, 0
    w_item = torch.cat([torch.full((n_max * L,), hist_w), torch.ones(L)]).to(device)            # weight per code token
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(H))
        tot, nb = 0.0, 0
        for s in range(0, len(H), bs):
            ix = perm[s:s + bs]
            h, m = torch.from_numpy(H[ix]).to(device), torch.from_numpy(M[ix]).to(device)
            y = torch.from_numpy(Y[ix]).to(device)
            items = torch.cat([h, y[:, None]], 1)
            valid = torch.cat([m, torch.ones(len(ix), 1, dtype=torch.bool, device=device)], 1)
            tok = item_tokens(codes_t, items, valid, K, L)                                          # [B, (n+1)*L]
            seq = with_bos(tok, valid, L, model.BOS, model.PAD)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(seq[:, :-1])
            tgt = seq[:, 1:]
            ok = tgt != model.PAD
            lp = model.code_logprob(logits.float(), tgt.clamp(max=L * K - 1))
            w = w_item[None, :] * ok
            loss = -(lp * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1; step += 1
            if step % 200 == 0:
                log(f"        sid-gpt step {step}/{total} loss={tot / nb:.4f}")
            if (max_steps and step >= max_steps) or (deadline is not None and time.time() > deadline):
                break
        score = val_fn(model) if val_fn else float(ep)
        log(f"      [sid-gpt] epoch {ep} loss={tot / max(nb, 1):.4f} val={score:.4f}")
        if score > best_score:
            best_score, best_ep, best_state = score, ep, copy.deepcopy(model.state_dict())
        if (max_steps and step >= max_steps) or (deadline is not None and time.time() > deadline):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_ep, "val": float(best_score), "steps": step}


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def history_free_logprob(model, codes_t, n_items, bs=4096):
    """log P(codes(article) | empty history) for every article -> [n]."""
    L, K, dev = model.L, model.K, codes_t.device
    out = torch.zeros(n_items, device=dev)
    for s in range(0, n_items, bs):
        c = codes_t[s:s + bs]
        seq = torch.cat([torch.full((len(c), 1), model.BOS, device=dev), c + torch.arange(L, device=dev) * K], 1)
        lp = model.code_logprob(model(seq[:, :-1]).float(), seq[:, 1:])
        out[s:s + bs] = lp.sum(1)
    return out


@torch.no_grad()
def sid_scorer(model, codes, n_max=30, pmi=False, rows_budget=2048, device="cpu"):
    """Slate scorer: ``log P(codes(c) | history)`` (minus the history-free log-prob when ``pmi``)."""
    L, K = model.L, model.K
    codes_t = torch.from_numpy(codes).to(device)
    base = history_free_logprob(model, codes_t, len(codes)) if pmi else None
    amp = device == "cuda"

    @torch.no_grad()
    def f(b):
        model.eval()
        B, C = b["cand"].shape
        items, valid = right_align(b["hist"], b["hist_mask"], n_max)
        out = torch.empty(B, C, device=device)
        step = max(1, rows_budget // C)
        for s in range(0, B, step):
            sl = slice(s, s + step)
            n = items[sl].shape[0]
            ht = with_bos(item_tokens(codes_t, items[sl], valid[sl], K, L), valid[sl], L, model.BOS, model.PAD)  # [n, 1+n_max*L]
            ct = codes_t[b["cand"][sl]] + torch.arange(L, device=device) * K         # [n, C, L]
            seq = torch.cat([ht[:, None, :].expand(n, C, ht.shape[1]), ct], -1).view(n * C, -1)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(seq[:, :-1], last_n=L)
            lp = model.code_logprob(logits.float(), seq[:, -L:]).sum(-1).view(n, C)
            out[sl] = lp
        return out - base[b["cand"]] if pmi else out
    return f
