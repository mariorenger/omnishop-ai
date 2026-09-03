"""Model zoo for the benchmark, all behind one interface.

Every model implements ``score(batch) -> logits[B, C]`` where C is the number of
candidates (training: 1 positive at slot 0 + negatives; eval: the padded slate).
The training loss and the eval metrics therefore treat every model identically,
so differences in the results table come only from the architecture — not from
the plumbing.

Families
--------
- NRMS       content, multi-head self-attention (Wu et al. 2019)
- NAML       content, CNN + multi-view additive attention (Wu et al. 2019)
- Fastformer content, additive (linear-complexity) attention (Wu et al. 2021)
- LightGCN   PURE GRAPH: user/news id embeddings on the click bipartite graph,
             no text at all (He et al. 2020)
- LLMEnc     LLM-as-encoder: frozen precomputed news embeddings + light head
             (the ONCE/DIRE-style discriminative LLM pipeline)
- HybridOpt  our optimisation: Fastformer news encoder (cheap) + candidate-aware
             user attention (CAUM-style) + a fused collaborative term, so it
             carries both content and graph signal and degrades gracefully on
             cold news where pure LightGCN fails.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD, Batch


# --------------------------------------------------------------------- blocks
class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention pooling over a sequence -> single vector."""

    def __init__(self, dim: int, hidden: int = 200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, mask=None):  # x [..., N, D], mask [..., N] bool
        a = self.query(torch.tanh(self.proj(x))).squeeze(-1)  # [..., N]
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=-1)
        return torch.einsum("...n,...nd->...d", w, x)


class FastformerBlock(nn.Module):
    """One Fastformer layer: additive attention with linear complexity."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh, self.dim = heads, dim // heads, dim
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.q_att = nn.Linear(self.dh, 1)
        self.k_att = nn.Linear(self.dh, 1)
        self.Wr = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def _split(self, x, B, N):
        return x.view(B, N, self.h, self.dh).transpose(1, 2)  # [B, h, N, dh]

    def forward(self, x, mask=None):  # x [B, N, dim], mask [B, N] bool
        B, N, _ = x.shape
        Q = self._split(self.Wq(x), B, N)
        K = self._split(self.Wk(x), B, N)
        V = self._split(self.Wv(x), B, N)
        m = None if mask is None else mask[:, None, :]  # [B,1,N]

        aq = (self.q_att(Q).squeeze(-1)) / (self.dh ** 0.5)  # [B,h,N]
        if m is not None:
            aq = aq.masked_fill(~m, -1e9)
        aq = torch.softmax(aq, dim=-1)
        global_q = torch.einsum("bhn,bhnd->bhd", aq, Q)      # [B,h,dh]

        p = K * global_q.unsqueeze(2)                         # key-query interaction
        ak = (self.k_att(p).squeeze(-1)) / (self.dh ** 0.5)
        if m is not None:
            ak = ak.masked_fill(~m, -1e9)
        ak = torch.softmax(ak, dim=-1)
        global_k = torch.einsum("bhn,bhnd->bhd", ak, p)       # [B,h,dh]

        u = V * global_k.unsqueeze(2)                         # [B,h,N,dh]
        r = self.Wr(u.transpose(1, 2).reshape(B, N, self.dim))
        return self.drop(r + self.Wq(x))                      # residual with query


# --------------------------------------------------------------------- models
class NRMS(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        D, drop, heads = cfg.dim, cfg.dropout, cfg.heads
        self.word_emb = nn.Embedding(d.vocab_size, D, padding_idx=PAD)
        self.news_mha = nn.MultiheadAttention(D, heads, dropout=drop, batch_first=True)
        self.news_pool = AdditiveAttention(D)
        self.user_mha = nn.MultiheadAttention(D, heads, dropout=drop, batch_first=True)
        self.user_pool = AdditiveAttention(D)
        self.drop = nn.Dropout(drop)

    def _news(self, title):  # [X, L] -> [X, D]
        e = self.drop(self.word_emb(title))
        o, _ = self.news_mha(e, e, e)
        return self.news_pool(self.drop(o))

    def score(self, b: Batch):
        B, H, L = b.hist_title.shape
        C = b.cand_title.shape[1]
        hist = self._news(b.hist_title.reshape(B * H, L)).view(B, H, -1)
        cand = self._news(b.cand_title.reshape(B * C, L)).view(B, C, -1)
        # MultiheadAttention returns NaN for a fully-padded row (cold user on
        # real MIND); let such rows attend freely, the additive pool masks them.
        kpm = ~b.hist_mask
        kpm = kpm.masked_fill(kpm.all(dim=1, keepdim=True), False)
        o, _ = self.user_mha(hist, hist, hist, key_padding_mask=kpm)
        user = self.user_pool(o, b.hist_mask)                 # [B, D]
        return torch.einsum("bd,bcd->bc", user, cand)


class NAML(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        D, drop = cfg.dim, cfg.dropout
        self.word_emb = nn.Embedding(d.vocab_size, D, padding_idx=PAD)
        self.cat_emb = nn.Embedding(d.n_cat, D, padding_idx=PAD)
        self.conv = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.title_pool = AdditiveAttention(D)
        self.cat_dense = nn.Linear(D, D)
        self.view_pool = AdditiveAttention(D)
        self.user_pool = AdditiveAttention(D)
        self.drop = nn.Dropout(drop)

    def _news(self, title, cat):  # [X, L], [X] -> [X, D]
        e = self.drop(self.word_emb(title)).transpose(1, 2)   # [X, D, L]
        c = torch.relu(self.conv(e)).transpose(1, 2)          # [X, L, D]
        title_vec = self.title_pool(self.drop(c))             # [X, D]
        cat_vec = torch.relu(self.cat_dense(self.cat_emb(cat)))
        views = torch.stack([title_vec, cat_vec], dim=1)      # [X, 2, D]
        return self.view_pool(views)

    def score(self, b: Batch):
        B, H, L = b.hist_title.shape
        C = b.cand_title.shape[1]
        hist = self._news(b.hist_title.reshape(B * H, L), b.hist_cat.reshape(B * H)).view(B, H, -1)
        cand = self._news(b.cand_title.reshape(B * C, L), b.cand_cat.reshape(B * C)).view(B, C, -1)
        user = self.user_pool(hist, b.hist_mask)
        return torch.einsum("bd,bcd->bc", user, cand)


class Fastformer(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        D, drop, heads = cfg.dim, cfg.dropout, cfg.heads
        self.word_emb = nn.Embedding(d.vocab_size, D, padding_idx=PAD)
        self.news_ff = FastformerBlock(D, heads, drop)
        self.news_pool = AdditiveAttention(D)
        self.user_ff = FastformerBlock(D, heads, drop)
        self.user_pool = AdditiveAttention(D)
        self.drop = nn.Dropout(drop)

    def _news(self, title):
        e = self.drop(self.word_emb(title))
        return self.news_pool(self.news_ff(e))

    def score(self, b: Batch):
        B, H, L = b.hist_title.shape
        C = b.cand_title.shape[1]
        hist = self._news(b.hist_title.reshape(B * H, L)).view(B, H, -1)
        cand = self._news(b.cand_title.reshape(B * C, L)).view(B, C, -1)
        user = self.user_pool(self.user_ff(hist, b.hist_mask), b.hist_mask)
        return torch.einsum("bd,bcd->bc", user, cand)


class LightGCN(nn.Module):
    """Pure collaborative graph model — no text is ever read."""

    def __init__(self, d, cfg):
        super().__init__()
        D = cfg.dim
        self.n_users = d.n_users
        self.n_layers = cfg.gcn_layers
        self.user_emb = nn.Embedding(d.n_users, D)
        self.news_emb = nn.Embedding(d.n_news, D)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.news_emb.weight, std=0.1)
        self.register_buffer("adj", d.build_adjacency())

    def _propagate(self):
        x = torch.cat([self.user_emb.weight, self.news_emb.weight], dim=0)
        agg = x
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            agg = agg + x
        agg = agg / (self.n_layers + 1)
        return agg[: self.n_users], agg[self.n_users:]

    def score(self, b: Batch):
        u_all, n_all = self._propagate()
        user = u_all[b.user_idx]                              # [B, D]
        cand = n_all[b.cand_idx]                              # [B, C, D]
        return torch.einsum("bd,bcd->bc", user, cand)


class LLMNewsEncoder(nn.Module):
    """Frozen pretrained news embeddings + a light trainable projection.

    ``data.llm_emb`` holds one vector per news id — on synthetic data a simulated
    embedding, on real MIND the output of a strong lightweight sentence embedder
    (BGE-M3, Jina v3, or any sentence-transformer; see ``llm_embed.py``). Only the
    small projection head trains online, so this is cheap to serve. Maps a news
    index tensor ``[...]`` to ``[..., D]``."""

    def __init__(self, d, cfg):
        super().__init__()
        emb = torch.from_numpy(d.llm_emb)
        self.register_buffer("emb", emb)
        self.head = nn.Sequential(
            nn.Linear(emb.shape[1], cfg.dim), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.dim, cfg.dim),
        )

    def forward(self, idx):
        return self.head(self.emb[idx])


class LLMEnc(nn.Module):
    """LLM-as-encoder base: frozen pretrained news vectors + additive user pool.
    Mirrors the production pipeline (news embeddings computed offline once, almost
    no online cost)."""

    def __init__(self, d, cfg):
        super().__init__()
        self.news_idx = LLMNewsEncoder(d, cfg)
        self.user_pool = AdditiveAttention(cfg.dim)

    def score(self, b: Batch):
        hist = self.news_idx(b.hist_idx)                      # [B, H, D]
        cand = self.news_idx(b.cand_idx)                      # [B, C, D]
        user = self.user_pool(hist, b.hist_mask)
        return torch.einsum("bd,bcd->bc", user, cand)


class HybridOpt(nn.Module):
    """Our optimisation: linear-attention content + candidate-aware user +
    fused collaborative signal.

    - Fastformer news encoder keeps the per-token cost linear (vs NRMS' O(L^2)).
    - Candidate-aware attention (CAUM idea) re-weights the reading history
      *per candidate*, which plain additive/self attention cannot do.
    - A learnable-weighted collaborative dot product adds the graph signal, and
      because the content path always works, cold news does not collapse the way
      it does for pure LightGCN.
    """

    def __init__(self, d, cfg):
        super().__init__()
        D, drop, heads = cfg.dim, cfg.dropout, cfg.heads
        self.word_emb = nn.Embedding(d.vocab_size, D, padding_idx=PAD)
        self.news_ff = FastformerBlock(D, heads, drop)
        self.news_pool = AdditiveAttention(D)
        self.cand_proj = nn.Linear(D, D, bias=False)          # bilinear attn key
        self.user_id = nn.Embedding(d.n_users, D)
        self.news_id = nn.Embedding(d.n_news, D)
        nn.init.normal_(self.user_id.weight, std=0.1)
        nn.init.normal_(self.news_id.weight, std=0.1)
        self.log_lambda = nn.Parameter(torch.zeros(()))       # fusion weight
        self.drop = nn.Dropout(drop)
        self.scale = D ** 0.5

    def _news(self, title):
        e = self.drop(self.word_emb(title))
        return self.news_pool(self.news_ff(e))

    def score(self, b: Batch):
        B, H, L = b.hist_title.shape
        C = b.cand_title.shape[1]
        hist = self._news(b.hist_title.reshape(B * H, L)).view(B, H, -1)   # [B,H,D]
        cand = self._news(b.cand_title.reshape(B * C, L)).view(B, C, -1)   # [B,C,D]
        # candidate-aware user representation: attend history w.r.t. each candidate
        att = torch.einsum("bcd,bhd->bch", self.cand_proj(cand), hist) / self.scale
        att = att.masked_fill(~b.hist_mask[:, None, :], -1e9)
        att = torch.softmax(att, dim=-1)
        user_ca = torch.einsum("bch,bhd->bcd", att, hist)     # [B, C, D]
        content = (user_ca * cand).sum(-1)                    # [B, C]
        collab = torch.einsum("bd,bcd->bc", self.user_id(b.user_idx),
                              self.news_id(b.cand_idx))
        return content + torch.exp(self.log_lambda) * collab


# ===================================================================== #
#  TOP base: CAUM (candidate-aware) — strongest reproducible MIND model  #
# ===================================================================== #
class SelfAttnNewsEncoder(nn.Module):
    """NRMS-style news encoder. Shared by every improvement variant below so
    that only the *user* encoder differs between them (clean ablation)."""

    def __init__(self, d, cfg):
        super().__init__()
        D, drop, heads = cfg.dim, cfg.dropout, cfg.heads
        self.word_emb = nn.Embedding(d.vocab_size, D, padding_idx=PAD)
        self.mha = nn.MultiheadAttention(D, heads, dropout=drop, batch_first=True)
        self.pool = AdditiveAttention(D)
        self.drop = nn.Dropout(drop)

    def forward(self, title):  # [X, L] -> [X, D]
        e = self.drop(self.word_emb(title))
        o, _ = self.mha(e, e, e)
        return self.pool(self.drop(o))


def _safe_kpm(mask):
    """Key-padding mask that never masks a whole row (avoids MHA NaN)."""
    kpm = ~mask
    return kpm.masked_fill(kpm.all(dim=1, keepdim=True), False)


class CAUM(nn.Module):
    """Candidate-Aware User Modeling (Qi et al. 2022) — among the strongest
    reproducible content models on MIND. History is contextualised by
    self-attention, then re-weighted *per candidate* via an MLP interaction."""

    def __init__(self, d, cfg):
        super().__init__()
        D, drop, heads = cfg.dim, cfg.dropout, cfg.heads
        self.news = SelfAttnNewsEncoder(d, cfg)
        self.hist_sa = nn.MultiheadAttention(D, heads, dropout=drop, batch_first=True)
        self.inter = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))

    def score(self, b: Batch):
        B, H, L = b.hist_title.shape
        C = b.cand_title.shape[1]
        hist = self.news(b.hist_title.reshape(B * H, L)).view(B, H, -1)
        cand = self.news(b.cand_title.reshape(B * C, L)).view(B, C, -1)
        h, _ = self.hist_sa(hist, hist, hist, key_padding_mask=_safe_kpm(b.hist_mask))
        D = h.size(-1)
        hh = h.unsqueeze(1).expand(B, C, H, D)
        cc = cand.unsqueeze(2).expand(B, C, H, D)
        a = self.inter(torch.cat([hh, cc], dim=-1)).squeeze(-1)   # [B, C, H]
        a = a.masked_fill(~b.hist_mask[:, None, :], -1e9)
        a = torch.softmax(a, dim=-1)
        user_ca = torch.einsum("bch,bchd->bcd", a, hh)            # [B, C, D]
        return (user_ca * cand).sum(-1)


# ===================================================================== #
#  Improvement blocks (2024-2026 LLM attention, ported to the user side) #
# ===================================================================== #
class DiffAttention(nn.Module):
    """Differential attention (Ye et al., ICLR 2025): two softmax maps minus one
    another cancel common-mode noise — here, denoises the noisy click history."""

    def __init__(self, dim, heads, dropout=0.0, depth=1):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, 2 * dim)
        self.k = nn.Linear(dim, 2 * dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))
        self.lq1 = nn.Parameter(torch.randn(self.dh) * 0.1)
        self.lk1 = nn.Parameter(torch.randn(self.dh) * 0.1)
        self.lq2 = nn.Parameter(torch.randn(self.dh) * 0.1)
        self.lk2 = nn.Parameter(torch.randn(self.dh) * 0.1)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):  # x [B, N, dim]
        B, N, _ = x.shape
        q = self.q(x).view(B, N, 2, self.h, self.dh)
        k = self.k(x).view(B, N, 2, self.h, self.dh)
        v = self.v(x).view(B, N, self.h, self.dh)

        def attn(qi, ki):
            s = torch.einsum("bnhd,bmhd->bhnm", qi, ki) / (self.dh ** 0.5)
            if mask is not None:
                s = s.masked_fill(~mask[:, None, None, :], -1e9)
            return torch.softmax(s, dim=-1)

        a1 = attn(q[:, :, 0], k[:, :, 0])
        a2 = attn(q[:, :, 1], k[:, :, 1])
        lam = (torch.exp((self.lq1 * self.lk1).sum())
               - torch.exp((self.lq2 * self.lk2).sum()) + self.lambda_init)
        a = self.drop(a1 - lam * a2)
        out = torch.einsum("bhnm,bmhd->bnhd", a, v).reshape(B, N, -1)
        out = self.norm(out) * (1 - self.lambda_init)
        return self.o(out)


class MLASelfAttn(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2/V3): K and V come from a low-rank
    latent projection of the history — the per-position cache is the small
    latent, not full K/V."""

    def __init__(self, dim, heads, latent, dropout=0.0):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv_down = nn.Linear(dim, latent)     # compress (the cached state)
        self.k_up = nn.Linear(latent, dim)
        self.v_up = nn.Linear(latent, dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def _split(self, t, B, N):
        return t.view(B, N, self.h, self.dh).transpose(1, 2)

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        c = self.kv_down(x)
        q = self._split(self.q(x), B, N)
        k = self._split(self.k_up(c), B, N)
        v = self._split(self.v_up(c), B, N)
        s = torch.einsum("bhnd,bhmd->bhnm", q, k) / (self.dh ** 0.5)
        if mask is not None:
            s = s.masked_fill(~mask[:, None, None, :], -1e9)
        a = self.drop(torch.softmax(s, dim=-1))
        o = torch.einsum("bhnm,bhmd->bhnd", a, v).transpose(1, 2).reshape(B, N, -1)
        return self.o(o)


class SSMBlock(nn.Module):
    """Lightweight selective diagonal state-space layer (Mamba-style): an O(N)
    input-dependent linear recurrence over the history. Not the full Mamba CUDA
    kernel — a compact SSM that captures the linear-time long-range idea."""

    def __init__(self, dim):
        super().__init__()
        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)
        self.c = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x, mask=None):  # x [B, N, D]
        B, N, D = x.shape
        decay = torch.sigmoid(self.a(x))     # selective (input-dependent) forget
        bx = self.b(x) * x
        cc = self.c(x)
        h = x.new_zeros(B, D)
        ys = []
        for t in range(N):
            new_h = decay[:, t] * h + bx[:, t]
            if mask is not None:
                h = torch.where(mask[:, t].unsqueeze(-1), new_h, h)  # freeze on pad
            else:
                h = new_h
            ys.append(cc[:, t] * h)
        y = torch.stack(ys, dim=1)
        return y * torch.sigmoid(self.gate(x))


# ------ RoPE helpers + RoPE news encoder ------
def _rope_tables(seq, dh, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device).float() / dh))
    t = torch.arange(seq, device=device).float()[:, None] * inv[None, :]
    return torch.cos(t), torch.sin(t)                 # [seq, dh/2]


def _apply_rope(x, cos, sin):  # x [., h, N, dh]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class RopeNewsEncoder(nn.Module):
    """News encoder whose token self-attention uses rotary positions (NRMS has
    no positional signal at all)."""

    def __init__(self, d, cfg):
        super().__init__()
        D, drop, heads = cfg.dim, cfg.dropout, cfg.heads
        self.h, self.dh, self.D = heads, D // heads, D
        self.word_emb = nn.Embedding(d.vocab_size, D, padding_idx=PAD)
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.o = nn.Linear(D, D)
        self.pool = AdditiveAttention(D)
        self.drop = nn.Dropout(drop)

    def forward(self, title):  # [X, L]
        X, L = title.shape
        e = self.drop(self.word_emb(title))

        def sp(t):
            return t.view(X, L, self.h, self.dh).transpose(1, 2)

        q, k, v = sp(self.q(e)), sp(self.k(e)), sp(self.v(e))
        cos, sin = _rope_tables(L, self.dh, title.device)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        s = torch.einsum("xhld,xhmd->xhlm", q, k) / (self.dh ** 0.5)
        a = torch.softmax(s, dim=-1)
        o = torch.einsum("xhlm,xhmd->xhld", a, v).transpose(1, 2).reshape(X, L, self.D)
        return self.pool(self.drop(self.o(o)))


# ===================================================================== #
#  Improvement models: shared NRMS news encoder, different user encoder  #
# ===================================================================== #
class _ContentTwoTower(nn.Module):
    """Encode history + candidates, then delegate user modeling to subclasses.

    ``build_news=False`` skips the learned word-level news encoder (used when a
    subclass plugs in frozen pretrained news embeddings instead)."""

    def __init__(self, d, cfg, news=None, build_news=True):
        super().__init__()
        self.pretrained = False
        if build_news:
            self.news = news or SelfAttnNewsEncoder(d, cfg)

    def _encode(self, b: Batch):
        if self.pretrained:                                   # frozen BGE/Jina emb
            return self.news_idx(b.hist_idx), self.news_idx(b.cand_idx)
        B, H, L = b.hist_title.shape
        C = b.cand_title.shape[1]
        hist = self.news(b.hist_title.reshape(B * H, L)).view(B, H, -1)
        cand = self.news(b.cand_title.reshape(B * C, L)).view(B, C, -1)
        return hist, cand


class NrmsDiff(_ContentTwoTower):
    def __init__(self, d, cfg):
        super().__init__(d, cfg)
        self.user = DiffAttention(cfg.dim, cfg.heads, cfg.dropout)
        self.pool = AdditiveAttention(cfg.dim)

    def score(self, b: Batch):
        hist, cand = self._encode(b)
        user = self.pool(self.user(hist, b.hist_mask), b.hist_mask)
        return torch.einsum("bd,bcd->bc", user, cand)


class NrmsMLA(_ContentTwoTower):
    def __init__(self, d, cfg):
        super().__init__(d, cfg)
        self.user = MLASelfAttn(cfg.dim, cfg.heads, cfg.latent, cfg.dropout)
        self.pool = AdditiveAttention(cfg.dim)

    def score(self, b: Batch):
        hist, cand = self._encode(b)
        user = self.pool(self.user(hist, b.hist_mask), b.hist_mask)
        return torch.einsum("bd,bcd->bc", user, cand)


class NrmsSSM(_ContentTwoTower):
    def __init__(self, d, cfg):
        super().__init__(d, cfg)
        self.user = SSMBlock(cfg.dim)
        self.pool = AdditiveAttention(cfg.dim)

    def score(self, b: Batch):
        hist, cand = self._encode(b)
        user = self.pool(self.user(hist, b.hist_mask), b.hist_mask)
        return torch.einsum("bd,bcd->bc", user, cand)


class NrmsRoPE(_ContentTwoTower):
    def __init__(self, d, cfg):
        super().__init__(d, cfg, news=RopeNewsEncoder(d, cfg))
        self.user_mha = nn.MultiheadAttention(cfg.dim, cfg.heads,
                                              dropout=cfg.dropout, batch_first=True)
        self.pool = AdditiveAttention(cfg.dim)
        self.recency = nn.Parameter(torch.zeros(d.max_hist))  # learned recency bias

    def score(self, b: Batch):
        hist, cand = self._encode(b)
        o, _ = self.user_mha(hist, hist, hist, key_padding_mask=_safe_kpm(b.hist_mask))
        # additive attention pooling with an added ALiBi-style recency bias
        w = self.pool.query(torch.tanh(self.pool.proj(o))).squeeze(-1)   # [B, H]
        w = w + self.recency[: o.size(1)][None, :]
        w = w.masked_fill(~b.hist_mask, -1e9)
        w = torch.softmax(w, dim=-1)
        user = torch.einsum("bn,bnd->bd", w, o)
        return torch.einsum("bd,bcd->bc", user, cand)


class NrmsMulti(_ContentTwoTower):
    """Multi-interest user modeling (MINS/MINER-style poly-attention): several
    interest vectors, candidate scored by its best-matching interest."""

    def __init__(self, d, cfg):
        super().__init__(d, cfg)
        self.W1 = nn.Linear(cfg.dim, cfg.dim)
        self.W2 = nn.Linear(cfg.dim, cfg.n_interest, bias=False)

    def score(self, b: Batch):
        hist, cand = self._encode(b)
        a = self.W2(torch.tanh(self.W1(hist)))              # [B, H, K]
        a = a.masked_fill(~b.hist_mask[:, :, None], -1e9)
        a = torch.softmax(a, dim=1)                          # over history
        user_k = torch.einsum("bhk,bhd->bkd", a, hist)      # [B, K, D]
        s = torch.einsum("bkd,bcd->bkc", user_k, cand)      # [B, K, C]
        return s.max(dim=1).values                           # best interest per cand


# ===================================================================== #
#  Graph fusion + contrastive learning + the combined "super" model      #
# ===================================================================== #
class GraphProp(nn.Module):
    """LightGCN propagation → collaborative user/news embeddings (reused by the
    graph-enhanced and super models)."""

    def __init__(self, d, cfg):
        super().__init__()
        D = cfg.dim
        self.n_users = d.n_users
        self.n_layers = cfg.gcn_layers
        self.user_emb = nn.Embedding(d.n_users, D)
        self.news_emb = nn.Embedding(d.n_news, D)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.news_emb.weight, std=0.1)
        self.register_buffer("adj", d.build_adjacency())

    def forward(self):
        x = torch.cat([self.user_emb.weight, self.news_emb.weight], dim=0)
        agg = x
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            agg = agg + x
        agg = agg / (self.n_layers + 1)
        return agg[: self.n_users], agg[self.n_users:]


class GraphRec(_ContentTwoTower):
    """Graph direction: NRMS content two-tower fused with LightGCN-propagated
    collaborative embeddings (real GNN message passing, unlike hybridopt's plain
    id embeddings). Score = content + learnable-weight * collaborative."""

    def __init__(self, d, cfg):
        super().__init__(d, cfg)
        self.user_pool = AdditiveAttention(cfg.dim)
        self.graph = GraphProp(d, cfg)
        self.gate = nn.Parameter(torch.zeros(()))

    def score(self, b: Batch):
        hist, cand = self._encode(b)
        cu = self.user_pool(hist, b.hist_mask)                 # content user
        content = torch.einsum("bd,bcd->bc", cu, cand)
        gu_all, gn_all = self.graph()
        collab = torch.einsum("bd,bcd->bc", gu_all[b.user_idx], gn_all[b.cand_idx])
        return content + torch.exp(self.gate) * collab


class NrmsCL(_ContentTwoTower):
    """Contrastive-learning direction: plain NRMS two-tower trained with an extra
    self-supervised InfoNCE loss (CL4SRec / SimCSE-style) — two dropout views of
    each user are pulled together, other users in the batch pushed apart."""

    def __init__(self, d, cfg):
        super().__init__(d, cfg)
        self.user_pool = AdditiveAttention(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.tau = cfg.cl_tau
        self.cl_weight = cfg.cl_weight

    def score(self, b: Batch):
        hist, cand = self._encode(b)
        user = self.user_pool(hist, b.hist_mask)
        return torch.einsum("bd,bcd->bc", user, cand)

    def extra_loss(self, b: Batch):
        hist, _ = self._encode(b)
        v1 = F.normalize(self.user_pool(self.drop(hist), b.hist_mask), dim=-1)
        v2 = F.normalize(self.user_pool(self.drop(hist), b.hist_mask), dim=-1)
        logits = (v1 @ v2.t()) / self.tau                      # [B, B]
        labels = torch.arange(v1.size(0), device=v1.device)
        return self.cl_weight * F.cross_entropy(logits, labels)


class SuperRec(_ContentTwoTower):
    """The 'combine everything' model, to stack against llmenc:
      • self-attention news encoder (NRMS)
      • DIFFERENTIAL attention to denoise the click history
      • CANDIDATE-AWARE attention over the denoised history (CAUM idea)
      • GRAPH (LightGCN) collaborative fusion
      • CONTRASTIVE self-supervised auxiliary loss
    """

    def __init__(self, d, cfg):
        pretrained = getattr(cfg, "news_encoder", "learned") == "pretrained"
        super().__init__(d, cfg, build_news=not pretrained)
        self.pretrained = pretrained
        if pretrained:                                        # BGE-M3 / Jina news emb
            self.news_idx = LLMNewsEncoder(d, cfg)
        self.hist_diff = DiffAttention(cfg.dim, cfg.heads, cfg.dropout)
        self.cand_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.graph = GraphProp(d, cfg)
        self.gate = nn.Parameter(torch.zeros(()))
        self.user_pool = AdditiveAttention(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.tau, self.cl_weight = cfg.cl_tau, cfg.cl_weight
        self.scale = cfg.dim ** 0.5

    def score(self, b: Batch):
        hist, cand = self._encode(b)
        h = self.hist_diff(hist, b.hist_mask)                  # denoised history
        att = torch.einsum("bcd,bhd->bch", self.cand_proj(cand), h) / self.scale
        att = att.masked_fill(~b.hist_mask[:, None, :], -1e9)
        att = torch.softmax(att, dim=-1)
        user_ca = torch.einsum("bch,bhd->bcd", att, h)         # candidate-aware user
        content = (user_ca * cand).sum(-1)
        gu_all, gn_all = self.graph()
        collab = torch.einsum("bd,bcd->bc", gu_all[b.user_idx], gn_all[b.cand_idx])
        return content + torch.exp(self.gate) * collab

    def extra_loss(self, b: Batch):
        hist, _ = self._encode(b)
        v1 = F.normalize(self.user_pool(self.drop(hist), b.hist_mask), dim=-1)
        v2 = F.normalize(self.user_pool(self.drop(hist), b.hist_mask), dim=-1)
        logits = (v1 @ v2.t()) / self.tau
        labels = torch.arange(v1.size(0), device=v1.device)
        return self.cl_weight * F.cross_entropy(logits, labels)


REGISTRY = {
    # --- top base models (reference) ---
    "nrms": NRMS,
    "naml": NAML,
    "fastformer": Fastformer,
    "caum": CAUM,               # strongest reproducible MIND content model
    "lightgcn": LightGCN,       # pure graph
    "llmenc": LLMEnc,           # LLM-as-encoder
    # --- improvement variants (2024-2026 attention, ablation on user encoder) ---
    "nrms_diff": NrmsDiff,      # differential attention (denoise)
    "nrms_mla": NrmsMLA,        # multi-head latent attention (low-rank KV)
    "nrms_ssm": NrmsSSM,        # Mamba-style selective SSM (linear, long history)
    "nrms_rope": NrmsRoPE,      # rotary positions + recency bias
    "nrms_multi": NrmsMulti,    # multi-interest poly-attention
    "graphrec": GraphRec,       # content + LightGCN graph fusion (GNN message passing)
    "nrms_cl": NrmsCL,          # contrastive learning (InfoNCE self-supervised aux)
    "hybridopt": HybridOpt,     # Fastformer + candidate-aware + collaborative fusion
    "supermodel": SuperRec,     # diff-attn + candidate-aware + graph + contrastive (all-in-one)
}

BASE_MODELS = ["nrms", "naml", "fastformer", "caum", "lightgcn", "llmenc"]
IMPROVED_MODELS = ["nrms_diff", "nrms_mla", "nrms_ssm", "nrms_rope", "nrms_multi",
                   "graphrec", "nrms_cl", "hybridopt", "supermodel"]
# Lean, non-diluted default: 3 strong transformers + graph + LLM-encoder + the
# consolidated super model. The rest stay available as an ablation menu.
CORE_MODELS = ["nrms", "fastformer", "caum", "lightgcn", "llmenc", "supermodel"]


def build_model(name: str, data, cfg) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'; choose from {list(REGISTRY)}")
    return REGISTRY[name](data, cfg)
