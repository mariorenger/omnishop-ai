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


class LLMEnc(nn.Module):
    """LLM-as-encoder: frozen precomputed news vectors + a light online head.

    Mirrors the production LLM pipeline (news embeddings computed offline once,
    almost no online cost). Swap ``data.llm_emb`` for real sentence-transformer
    or LLM embeddings to get a faithful LLM run.
    """

    def __init__(self, d, cfg):
        super().__init__()
        D, drop = cfg.dim, cfg.dropout
        emb = torch.from_numpy(d.llm_emb)
        self.register_buffer("llm_emb", emb)
        self.head = nn.Sequential(
            nn.Linear(emb.shape[1], D), nn.ReLU(), nn.Dropout(drop), nn.Linear(D, D)
        )
        self.user_pool = AdditiveAttention(D)

    def _news(self, idx):  # [...] -> [..., D]
        return self.head(self.llm_emb[idx])

    def score(self, b: Batch):
        hist = self._news(b.hist_idx)                         # [B, H, D]
        cand = self._news(b.cand_idx)                         # [B, C, D]
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


REGISTRY = {
    "nrms": NRMS,
    "naml": NAML,
    "fastformer": Fastformer,
    "lightgcn": LightGCN,
    "llmenc": LLMEnc,
    "hybridopt": HybridOpt,
}


def build_model(name: str, data, cfg) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'; choose from {list(REGISTRY)}")
    return REGISTRY[name](data, cfg)
