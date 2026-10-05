"""Reference news-recommendation models (NRMS / NAML / Fastformer) and a small trainer, scored by the suite's evaluator.

One ``RefNet`` class covers the three families so that a V10-style configuration (64-d, no title padding mask, ``nn.MultiheadAttention``
layout) and the published recipe (300-d GloVe, 16 heads x 16, masked, lr 1e-4) differ only in ``RefCfg`` -- that is what lets the
ablation ladder attribute the gap between V10's baselines and the literature to individual choices.

News vectors depend only on the article's own tokens, so evaluation encodes every article once (``all_news_vecs``) and a slate is
scored with ``<user vector, candidate vector>`` -- the same interface as the representation-level scorers in ``rep_eval``.
The building blocks (``AddAttn``, ``SelfAttn``, ``FastEncoder``) are reused by ``usermodel`` for learned user encoders over frozen
LLM article vectors.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0


# ------------------------------------------------------------------------------------------- building blocks
class AddAttn(nn.Module):
    """Additive attention pooling ``sum_i softmax(q . tanh(W x_i)) x_i`` over the second-to-last axis, padding masked."""

    def __init__(self, d, hidden=200):
        super().__init__()
        self.proj, self.q = nn.Linear(d, hidden), nn.Linear(hidden, 1, bias=False)

    def forward(self, x, mask=None):
        a = self.q(torch.tanh(self.proj(x))).squeeze(-1).float()
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        return torch.einsum("...n,...nd->...d", torch.softmax(a, -1).to(x.dtype), x)


class SelfAttn(nn.Module):
    """Multi-head self-attention.  NRMS layout: Q/K/V maps only (``out_proj=False``); ``out_proj=True`` adds the output projection and is
    the layout of ``torch.nn.MultiheadAttention`` that V10 uses.  ``mask`` [B, L] bool (True = real token) must keep one True per row."""

    def __init__(self, d_in, heads, head_dim, out_proj=False, attn_drop=0.0):
        super().__init__()
        D = heads * head_dim
        self.h, self.dh, self.p = heads, head_dim, attn_drop
        self.q, self.k, self.v = nn.Linear(d_in, D), nn.Linear(d_in, D), nn.Linear(d_in, D)
        self.o = nn.Linear(D, D) if out_proj else nn.Identity()

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        sp = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(self.q(x)), sp(self.k(x)), sp(self.v(x)),
                                           attn_mask=None if mask is None else mask[:, None, None, :],
                                           dropout_p=self.p if self.training else 0.0)
        return self.o(o.transpose(1, 2).reshape(B, L, self.h * self.dh))


class FastSelfAttn(nn.Module):
    """Fastformer additive attention (Wu et al. 2021), following the layout of the official code: a global query from per-head additive
    attention over the queries, an element-wise product with every key, a global key from a second additive attention, an element-wise
    product with the value (the query doubles as value), a linear transform and a residual connection to the query."""

    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.q_att, self.k_att = nn.Linear(d, heads), nn.Linear(d, heads)
        self.transform = nn.Linear(d, d)

    def _pool(self, scores, x, mask):
        B, L, d = x.shape
        w = torch.softmax((scores.float() / math.sqrt(self.dh)).masked_fill(~mask[:, :, None], -1e9), 1)
        return (w.unsqueeze(-1).to(x.dtype) * x.view(B, L, self.h, self.dh)).sum(1)                       # [B, h, dh]

    def forward(self, x, mask):
        B, L, d = x.shape
        q, k = self.q(x), self.k(x)
        gq = self._pool(self.q_att(q), q, mask)
        p = (k.view(B, L, self.h, self.dh) * gq[:, None]).reshape(B, L, d)
        gk = self._pool(self.k_att(p), p, mask)
        u = (gk[:, None] * q.view(B, L, self.h, self.dh)).reshape(B, L, d)
        return self.transform(u) + q


class FastLayer(nn.Module):
    def __init__(self, d, heads, dropout, ff_mult=4):
        super().__init__()
        self.att, self.ln1, self.ln2, self.drop = FastSelfAttn(d, heads), nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))

    def forward(self, x, mask):
        x = self.ln1(x + self.drop(self.att(x, mask)))
        return self.ln2(x + self.drop(self.ff(x)))


class FastEncoder(nn.Module):
    """Input projection + learned positions + ``layers`` Fastformer blocks (token mixing in O(L)).  ``dropout`` acts on the input
    (callers that already drop the embeddings pass 0), ``inner_dropout`` (BERT's 0.1) inside the blocks: stacking 0.2 dropouts stalls
    training at small widths (measured on the toy fixture)."""

    def __init__(self, d_in, d, heads, layers, max_len, dropout=0.0, inner_dropout=0.1):
        super().__init__()
        self.proj = nn.Identity() if d_in == d else nn.Linear(d_in, d)
        self.pos, self.ln, self.drop = nn.Embedding(max_len, d), nn.LayerNorm(d), nn.Dropout(dropout)
        self.layers = nn.ModuleList(FastLayer(d, heads, inner_dropout) for _ in range(layers))

    def forward(self, x, mask):
        L = x.shape[1]
        x = self.drop(self.ln(self.proj(x) + self.pos(torch.arange(L, device=x.device))))
        for layer in self.layers:
            x = layer(x, mask)
        return x


def safe_mask(mask):
    """A row without any real token attends to itself instead of producing NaN (such rows are zeroed afterwards)."""
    return mask | ~mask.any(-1, keepdim=True)


# ------------------------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class RefCfg:
    kind: str = "nrms"              # nrms | naml | fastformer
    text: str = "ref"               # ref: regex tokens, title-only (abstract for NAML) | v10: the 20-token title+abstract ids of data.py
    emb_dim: int = 300
    heads: int = 16
    head_dim: int = 16
    att_hidden: int = 200
    dropout: float = 0.2
    attn_drop: float = 0.0
    out_proj: bool = False          # True: torch.nn.MultiheadAttention layout (V10)
    title_mask: bool = True         # False: padding tokens take part in attention and pooling (V10)
    init: str = "glove"             # glove | random (N(0, 0.1)) | torch (nn.Embedding default N(0, 1), V10)
    dyn_neg: bool = True            # re-draw the negatives every epoch (V10: once)
    lr: float = 1e-4
    wd: float = 0.0
    bs: int = 64
    max_epochs: int = 10
    min_epochs: int = 3
    patience: int = 2
    min_delta: float = 1e-3
    ff_layers: int = 1              # V10's Fastformer is one block too; a second layer doubles the cost (the news encoder dominates)
    naml_filters: int = 400
    cat_dim: int = 100
    clip: float = 5.0               # gradient-norm clip (0 = none, as in V10)


def preset(name):
    """``v10``: V10's NRMS recipe re-implemented (harness check).  ``data``: + title-only regex tokens, padding mask, negatives re-drawn each
    epoch.  ``recipe``: the published NRMS hyper-parameters with random word vectors.  ``nrms`` / ``naml`` / ``fastformer``: + GloVe."""
    if name == "v10":
        return RefCfg(kind="nrms", text="v10", emb_dim=64, heads=2, head_dim=32, dropout=0.2, attn_drop=0.2, out_proj=True, title_mask=False,
                      init="torch", dyn_neg=False, lr=1e-3, wd=1e-5, max_epochs=12, clip=0.0)
    if name == "data":
        return replace(preset("v10"), text="ref", title_mask=True, dyn_neg=True)
    if name == "recipe":
        return RefCfg(kind="nrms", init="random")
    if name in ("nrms", "naml", "fastformer"):
        return RefCfg(kind=name, init="glove")
    raise KeyError(name)


# ------------------------------------------------------------------------------------------- the network
class RefNet(nn.Module):
    def __init__(self, cfg, tokens, vocab_size, n_cat=0, n_sub=0, emb_init=None, max_hist=50):
        super().__init__()
        self.cfg = cfg
        for k, v in tokens.items():
            self.register_buffer(k, v, persistent=False)
        D = cfg.heads * cfg.head_dim
        if cfg.out_proj and cfg.emb_dim != D:
            raise ValueError("the nn.MultiheadAttention layout needs emb_dim == heads * head_dim")
        self.emb = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=PAD)
        if emb_init is not None:
            with torch.no_grad():
                self.emb.weight.copy_(torch.as_tensor(emb_init))
                self.emb.weight[PAD].zero_()
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.kind == "nrms":
            self.news_att = SelfAttn(cfg.emb_dim, cfg.heads, cfg.head_dim, cfg.out_proj, cfg.attn_drop)
            self.news_pool = AddAttn(D, cfg.att_hidden)
            self.user_att = SelfAttn(D, cfg.heads, cfg.head_dim, cfg.out_proj, cfg.attn_drop)
            self.user_pool = AddAttn(D, cfg.att_hidden)
            self.out_dim = D
        elif cfg.kind == "fastformer":
            self.news_enc = FastEncoder(cfg.emb_dim, D, cfg.heads, cfg.ff_layers, tokens["title"].shape[1])
            self.news_pool = AddAttn(D, cfg.att_hidden)
            self.user_enc = FastEncoder(D, D, cfg.heads, cfg.ff_layers, max_hist)
            self.user_pool = AddAttn(D, cfg.att_hidden)
            self.out_dim = D
        elif cfg.kind == "naml":
            Fn = cfg.naml_filters
            self.conv_t, self.conv_b = nn.Conv1d(cfg.emb_dim, Fn, 3, padding=1), nn.Conv1d(cfg.emb_dim, Fn, 3, padding=1)
            self.pool_t, self.pool_b = AddAttn(Fn, cfg.att_hidden), AddAttn(Fn, cfg.att_hidden)
            self.cat_emb, self.sub_emb = nn.Embedding(n_cat, cfg.cat_dim, padding_idx=PAD), nn.Embedding(n_sub, cfg.cat_dim, padding_idx=PAD)
            self.cat_fc, self.sub_fc = nn.Linear(cfg.cat_dim, Fn), nn.Linear(cfg.cat_dim, Fn)
            self.view_pool, self.user_pool = AddAttn(Fn, cfg.att_hidden), AddAttn(Fn, cfg.att_hidden)
            self.out_dim = Fn
        else:
            raise ValueError(cfg.kind)

    # ---- articles
    def _mask(self, t, force=False):
        if not (self.cfg.title_mask or force):
            return None
        return safe_mask(t != PAD)

    def encode_news(self, ids):
        """[M] global article ids -> [M, out_dim]."""
        c, t = self.cfg, self.title[ids]
        if c.kind == "nrms":
            m = self._mask(t)
            return self.news_pool(self.drop(self.news_att(self.drop(self.emb(t)), m)), m)
        if c.kind == "fastformer":
            m = self._mask(t, force=True)
            return self.news_pool(self.news_enc(self.drop(self.emb(t)), m), m)
        b, mt, mb = self.body[ids], self._mask(t, True), self._mask(self.body[ids], True)
        tv = self.pool_t(self.drop(F.relu(self.conv_t(self.drop(self.emb(t)).transpose(1, 2)))).transpose(1, 2), mt)
        bv = self.pool_b(self.drop(F.relu(self.conv_b(self.drop(self.emb(b)).transpose(1, 2)))).transpose(1, 2), mb)
        cv = F.relu(self.cat_fc(self.cat_emb(self.cat[ids])))
        sv = F.relu(self.sub_fc(self.sub_emb(self.sub[ids])))
        return self.view_pool(torch.stack([tv, bv, cv, sv], 1))

    @torch.no_grad()
    def all_news_vecs(self, bs=4096, amp=False):
        """Eval-mode vectors of every article (row 0 = zeros) in float32."""
        was = self.training
        self.eval()
        N = self.title.shape[0]
        out = torch.zeros(N, self.out_dim, device=self.title.device)
        for s in range(0, N, bs):
            ids = torch.arange(s, min(s + bs, N), device=out.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out[s:s + len(ids)] = self.encode_news(ids).float()
        out[PAD] = 0
        self.train(was)
        return out

    # ---- readers
    def user_vec(self, hv, hmask):
        """[B, H, d] history vectors + [B, H] mask -> [B, d]; a reader without history gets the zero vector (all scores tie)."""
        c, safe = self.cfg, safe_mask(hmask)
        if c.kind == "nrms":
            u = self.user_pool(self.user_att(hv, safe), safe)
        elif c.kind == "fastformer":
            u = self.user_pool(self.user_enc(hv, safe), safe)
        else:
            u = self.user_pool(hv, safe)
        return u * hmask.any(-1, keepdim=True).to(u.dtype)

    def forward(self, hist, hmask, cand):
        """Training scores [B, C]: every distinct article of the batch is encoded once."""
        B, H = hist.shape
        C = cand.shape[1]
        uniq, inv = torch.unique(torch.cat([hist.reshape(-1), cand.reshape(-1)]), return_inverse=True)
        vecs = self.encode_news(uniq)
        hv, cv = vecs[inv[:B * H]].view(B, H, -1), vecs[inv[B * H:]].view(B, C, -1)
        return torch.einsum("bd,bcd->bc", self.user_vec(hv, hmask), cv)


def build_net(cfg, device, seed, text=None, data=None, max_hist=50):
    """``cfg.text == 'v10'`` reads V10's ``data.news_title`` (20 tokens, OOV -> PAD); otherwise the ``RefText`` tables."""
    torch.manual_seed(seed)
    if cfg.text == "v10":
        net = RefNet(cfg, {"title": torch.from_numpy(data.news_title)}, data.vocab_size, max_hist=max_hist)
    else:
        init = None if cfg.init == "torch" else text.embedding(cfg.emb_dim, seed, cfg.init)
        net = RefNet(cfg, text.tokens("cpu"), text.vocab_size, text.n_cat, text.n_sub, init, max_hist)
    return net.to(device)


def n_params(net):
    return int(sum(p.numel() for p in net.parameters()))


def make_scorer(net, amp=False):
    """Slate scorer for ``rep_eval.evaluate`` / ``Suite.run`` (encodes the catalogue once, in eval mode)."""
    V = net.all_news_vecs(amp=amp)

    def f(b):
        return torch.einsum("bd,bcd->bc", net.user_vec(V[b["hist"]], b["hist_mask"]), V[b["cand"]])
    return f


# ------------------------------------------------------------------------------------------- training
def train_ref(net, trainset, val_fn, cfg, seed=0, device="cpu", deadline=None, log=print, fixed_epochs=None):
    """Adam + cross-entropy over (1 positive, ``n_neg`` negatives).  Early stopping exactly as V10 / ``usermodel``: validation nDCG@10 per
    epoch, patience ``cfg.patience`` after ``cfg.min_epochs``, improvement must exceed ``cfg.min_delta``; the best epoch is restored.
    ``val_fn(net) -> float``.  Stops (after finishing a validation pass) once ``deadline`` has passed.
    ``fixed_epochs=k``: no validation and no early stopping -- exactly ``k`` epochs (or fewer if the deadline passes) and the final weights are
    kept; used to refit on train+validation for the epoch count a validated run selected (``val_fn`` is not called and may be None)."""
    amp = device == "cuda"
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, best_epoch, stale, hist, steps, t0 = None, None, 0, 0, [], 0, time.time()
    over = lambda: deadline is not None and time.time() > deadline
    for epoch in range(1, (fixed_epochs or cfg.max_epochs) + 1):
        if best_state is not None and over():
            log(f"      [ref] deadline reached after epoch {epoch - 1}")
            break
        net.train()
        te, tot, nb, cut = time.time(), torch.zeros((), device=device), 0, False
        for H, M, C in trainset.batches(cfg.bs, epoch, seed, device):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = net(H, M, C)
            loss = F.cross_entropy(logits.float(), torch.zeros(len(H), dtype=torch.long, device=device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
            scaler.step(opt)
            scaler.update()
            tot += loss.detach()
            nb += 1
            steps += 1
            if nb % 100 == 0:
                if not torch.isfinite(tot):
                    raise FloatingPointError(f"non-finite loss in epoch {epoch}, step {nb}")
                if over() and best_state is not None:
                    cut = True
                    break
        train_s = time.time() - te
        if not torch.isfinite(tot):
            raise FloatingPointError(f"non-finite loss in epoch {epoch}")
        net.eval()
        loss_m = float(tot) / max(nb, 1)
        if fixed_epochs:
            best, best_epoch = 0.0, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            hist.append({"epoch": epoch, "loss": loss_m, "val": float("nan"), "train_s": train_s, "partial": cut})
            log(f"      [ref] refit epoch {epoch:02d}/{fixed_epochs} loss={loss_m:.4f} ({train_s:.0f}s train, {nb} steps){' PARTIAL' if cut else ''}")
            if cut:
                break
            continue
        val = float(val_fn(net))
        hist.append({"epoch": epoch, "loss": loss_m, "val": val, "train_s": train_s, "partial": cut})
        improved = best is None or val > best + cfg.min_delta
        if improved:
            best, best_epoch, stale = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif epoch >= cfg.min_epochs:
            stale += 1
        log(f"      [ref] epoch {epoch:02d} loss={loss_m:.4f} val nDCG@10={val:.4f} ({train_s:.0f}s train, {nb} steps){' PARTIAL' if cut else ''}")
        if cut or (epoch >= cfg.min_epochs and stale >= cfg.patience):
            break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"best_val": float(best), "best_epoch": best_epoch, "epochs": len(hist), "steps": steps, "train_s": time.time() - t0,
                 "history": hist, "params": n_params(net)}
