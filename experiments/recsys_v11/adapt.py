"""Contrastive adaptation of text embeddings to news recommendation.

Hypothesis (LLM2Rec, KDD'25 / LC-Rec / CoLLM line): a frozen text embedding knows what an
article is *about*, not which articles readers consume together.  We adapt it with a
co-click objective and ask whether the gain **transfers to articles never seen in
training** -- the case that matters for news.

Training signal (V10 ``train_core`` only, so nothing from validation/test leaks):
  anchor   = one of the reader's last ``k_anchor`` history articles
  positive = an article the reader then clicked
  hard neg = articles shown to the same reader in that impression but skipped
             (impression-aware negatives; MIND provides them for free)
  + in-batch negatives, with false-negative masking (same article id).

Adapters (all start as the identity => step 0 == the frozen control by construction):
  lin / mlp     residual head on precomputed vectors            (TAM-style, no encoder pass)
  lora / dora   low-rank adaptation of the encoder, in the loop (LoRA; DoRA = weight-decomposed)
Options: ``mrl`` Matryoshka loss over nested prefixes (MRL, NeurIPS'22; used in recsys by
fMRLRec/SMEC) and ``hardneg`` impression negatives.
"""
from __future__ import annotations

import math
import re
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import fix_peft_torchao, load_hf


# ------------------------------------------------------------------- pairs
def build_pairs(data, k_anchor=3, max_pairs=200_000, seed=0):
    """(anchor, positive, hard-negatives) from ``data.train_core``."""
    rng = np.random.default_rng(seed)
    A, P, N = [], [], []
    seen = set()
    for _user, hist, pos, negs in data.train_core:
        if not len(hist):
            continue
        for a in hist[-k_anchor:]:
            key = (a, pos)
            if key in seen or a == pos:
                continue
            seen.add(key)
            A.append(a); P.append(pos); N.append(negs)
    A, P, N = np.array(A, np.int64), np.array(P, np.int64), np.array(N, np.int64)
    if len(A) > max_pairs:
        keep = rng.choice(len(A), max_pairs, replace=False)
        A, P, N = A[keep], P[keep], N[keep]
    return A, P, N


# -------------------------------------------------------------------- loss
def mrl_dims(d, scales=(64, 128, 256)):
    return [m for m in scales if m < d] + [d]


def contrastive_loss(a, p, p_idx, negs=None, tau=0.05, dims=None):
    """InfoNCE(anchor -> positive) with in-batch negatives (+ optional hard negatives),
    averaged over Matryoshka prefixes.  ``p_idx`` masks in-batch false negatives."""
    dims = dims or [a.shape[-1]]
    same = p_idx[:, None] == p_idx[None, :]
    same.fill_diagonal_(False)
    tgt = torch.arange(len(a), device=a.device)
    total = 0.0
    for m in dims:
        am, pm = F.normalize(a[:, :m], dim=-1), F.normalize(p[:, :m], dim=-1)
        logits = (am @ pm.T / tau).masked_fill(same, float("-inf"))
        if negs is not None:
            nm = F.normalize(negs[:, :, :m], dim=-1)
            logits = torch.cat([logits, torch.einsum("bd,bkd->bk", am, nm) / tau], dim=1)
        total = total + F.cross_entropy(logits, tgt)
    return total / len(dims)


# ------------------------------------------------------------------- heads
class ResLinear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.w.weight)

    def forward(self, x):
        return F.normalize(x + self.w(x), dim=-1)


class ResMLP(nn.Module):
    def __init__(self, d, h=None):
        super().__init__()
        self.a, self.b = nn.Linear(d, h or d), nn.Linear(h or d, d)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return F.normalize(x + self.b(F.gelu(self.a(x))), dim=-1)


HEADS = {"lin": ResLinear, "mlp": ResMLP}


def train_head(E0, pairs, kind="lin", mrl=False, hardneg=False, epochs=3, bs=1024, lr=1e-3,
               tau=0.05, seed=0, device="cpu", val_fn=None, deadline=None, log=print):
    """Train a residual head on precomputed vectors.  ``val_fn(E) -> float`` (higher is
    better) selects the epoch; epoch 0 (= frozen control) competes, so the returned
    embedding is never worse than frozen on validation."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    E = torch.from_numpy(E0).to(device)
    head = HEADS[kind](E.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    A, P, N = pairs
    dims = mrl_dims(E.shape[1]) if mrl else None
    best = (val_fn(E0) if val_fn else -math.inf, E0.copy(), 0)
    for ep in range(1, epochs + 1):
        if deadline is not None and time.time() > deadline and ep > 1:
            break
        perm = rng.permutation(len(A))
        head.train()
        tot, nb = 0.0, 0
        for s in range(0, len(A), bs):
            ix = perm[s:s + bs]
            a, p = head(E[A[ix]]), head(E[P[ix]])
            n = head(E[N[ix]]) if hardneg else None
            loss = contrastive_loss(a, p, torch.from_numpy(P[ix]).to(device), n, tau, dims)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        head.eval()
        with torch.no_grad():
            Ea = head(E).cpu().numpy()
        score = val_fn(Ea) if val_fn else float(ep)          # no validation signal: keep the last epoch
        log(f"      [{kind}{'+mrl' if mrl else ''}{'+hn' if hardneg else ''}] epoch {ep} loss={tot / max(nb, 1):.4f} val={score:.4f}")
        if score > best[0]:
            best = (score, Ea, ep)
    return best[1], {"best_epoch": best[2], "val": float(best[0])}


# ----------------------------------------------------------- encoder in loop
POOLING_BY_NAME = (("qwen3-embedding", "last"), ("bge", "cls"), ("e5", "mean"), ("gte", "mean"), ("minilm", "mean"))


def guess_pooling(name):
    low = name.lower()
    for key, mode in POOLING_BY_NAME:
        if key in low:
            return mode
    return "mean"


def params_b(name):
    """Parameter count in billions read from a model name (``Qwen/Qwen3-Embedding-4B`` -> 4.0, ``...-0.6B`` -> 0.6); 0 when the name carries none."""
    m = re.findall(r"(\d+(?:\.\d+)?)[bB](?![A-Za-z0-9])", name.split("/")[-1])
    return float(m[-1]) if m else 0.0


def resolve_dtype(name, enc_dtype="auto", device="cpu"):
    """Weight dtype of a frozen encoder.  ``fp32``: fp32 weights, fp16 autocast on a GPU (the suite's default); ``fp16``: the linear layers in fp16,
    embeddings and norms in fp32, same autocast -- the same arithmetic as ``fp32`` (autocast casts the weights of every matmul to fp16 anyway) at half
    the weight memory: a 4B encoder is 16 GB in fp32, more than a 16 GB T4 holds next to its activations; ``auto``: ``fp16`` for >= 3B parameters
    on a GPU, otherwise ``fp32``."""
    gpu = str(device).startswith("cuda")
    if enc_dtype == "fp16" or (enc_dtype == "auto" and gpu and params_b(name) >= 3.0):
        return torch.float16 if gpu else torch.float32          # fp16 matmuls on a CPU are slow and not what the GPU computes: keep fp32 there
    if enc_dtype not in ("auto", "fp32", "fp16"):
        raise ValueError(f"--enc-dtype {enc_dtype!r}: choose auto, fp32 or fp16")
    return torch.float32


def keep_fp32_residual(model):
    """Cast the embedding tables and every normalisation layer of a half-precision model back to fp32 (the linear layers stay fp16).  Under fp16 autocast the
    hidden states then stay fp32 between the matmuls -- what autocast computes for a model kept in fp32 -- so large activations cannot overflow fp16 in the
    residual stream.  Returns the number of modules cast."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Embedding) or "norm" in type(m).__name__.lower():
            m.float()
            n += 1
    return n


class HFEncoder:
    """Plain Hugging Face encoder with explicit pooling (so LoRA/DoRA training and
    inference share one code path and no ST-version drift)."""

    def __init__(self, name, pooling=None, max_len=128, device="cpu", trust_remote_code=False, prefix="",
                 dtype=torch.float32):
        from transformers import AutoModel, AutoTokenizer
        self.name, self.max_len, self.device, self.prefix = name, max_len, device, prefix
        self.pooling = pooling or guess_pooling(name)
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        if self.pooling == "last":
            self.tok.padding_side = "left"
        self.model = load_hf(AutoModel, name, dtype, trust_remote_code).to(device)
        gpu = str(device).startswith("cuda")
        if gpu and dtype == torch.float16:                             # fp16 linear layers: keep embeddings and norms (the residual stream) in fp32, as autocast does with fp32 weights
            keep_fp32_residual(self.model)
        self.amp = gpu and dtype in (torch.float32, torch.float16)   # fp16 matmuls under autocast, fp32 everywhere else

    def _pool(self, out, attn):
        h = out.last_hidden_state
        if self.pooling == "cls":
            return h[:, 0]
        if self.pooling == "last":
            return h[:, -1]
        m = attn.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def embed(self, texts):
        batch = self.tok([self.prefix + t for t in texts], padding=True, truncation=True,
                         max_length=self.max_len, return_tensors="pt").to(self.device)
        kw = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        if self.pooling == "last":
            kw["use_cache"] = False                                  # decoder-only embedders: no KV cache needed
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(**kw)
        return self._pool(out, batch["attention_mask"]).float()

    @torch.no_grad()
    def encode(self, texts, bs=128):
        was = self.model.training
        self.model.eval()
        order = np.argsort([len(t) for t in texts])
        vecs = np.zeros((len(texts), self.model.config.hidden_size), np.float32)
        for s in range(0, len(texts), bs):
            ix = order[s:s + bs]
            vecs[ix] = F.normalize(embed_batch(self, [texts[i] for i in ix]), dim=-1).cpu().numpy()
        self.model.train(was)
        return vecs

    def attach_peft(self, mode="lora", r=16, dropout=0.05):
        from peft import LoraConfig, get_peft_model
        fix_peft_torchao()
        names = {n.split(".")[-1] for n, _ in self.model.named_modules()}
        targets = ["q_proj", "v_proj"] if "q_proj" in names else ["query", "value"]
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=dropout, target_modules=targets,
                         use_dora=(mode == "dora"), bias="none")
        self.model = get_peft_model(self.model, cfg)
        try:                                                  # activations of 192 texts x 256 tokens do not fit a 16 GB card otherwise
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            pass
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def embed_batch(enc, texts):
    """``enc.embed(texts)`` that halves the batch while the GPU runs out of memory (the longest texts of a large encoder); every half is embedded as a
    batch of its own, so only the padding of that batch changes."""
    try:
        return enc.embed(texts)
    except torch.cuda.OutOfMemoryError:
        if len(texts) <= 1:
            raise
        torch.cuda.empty_cache()
        h = len(texts) // 2
        return torch.cat([embed_batch(enc, texts[:h]), embed_batch(enc, texts[h:])])


def encode_multi(name, texts, max_len, prefix, dtype, bs, devices, log=print):
    """``HFEncoder.encode`` spread over several GPUs: the same batches (texts sorted by length, consecutive slices of ``bs``), batch ``k`` on
    ``devices[k % n]``, one encoder per device in its own thread (grad mode and autocast are per thread; a tokenizer is not shared between threads).
    Every batch is computed exactly as on one GPU, so the vectors are the same; only the wall-clock time is divided."""
    encs = []
    for d in devices:
        encs.append(HFEncoder(name, max_len=max_len, device=d, prefix=prefix, dtype=dtype))
        encs[-1].model.eval()
    order = np.argsort([len(t) for t in texts])
    starts = list(range(0, len(texts), bs))
    vecs = np.zeros((len(texts), encs[0].model.config.hidden_size), np.float32)
    errors = []

    def work(i):
        try:
            with torch.no_grad():
                for s in starts[i::len(encs)]:
                    ix = order[s:s + bs]
                    vecs[ix] = F.normalize(embed_batch(encs[i], [texts[j] for j in ix]), dim=-1).cpu().numpy()
        except BaseException as e:                       # re-raised in the calling thread
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(encs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    del encs
    if errors:
        raise errors[0]
    return vecs


def check_equivalence(enc, st_encode, texts, tol=0.99):
    """Guard against pooling/prefix mistakes: HFEncoder must agree with the reference
    (Sentence-Transformers) embeddings of the same model."""
    a, b = enc.encode(texts), st_encode(texts)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    return float(cos.min()), float(cos.mean()), bool(cos.min() >= tol)


def fit_encoder_batch(enc, texts, pairs, bs, hardneg, params, log=print):
    """Dry-run one forward+backward on the pairs with the longest texts; halve the batch while the GPU is out of memory."""
    A, P, N = pairs
    L = np.array([len(texts[a]) + len(texts[p]) for a, p in zip(A, P)])
    while bs >= 4:
        ix = np.argsort(-L)[:bs]
        ids = np.concatenate([A[ix], P[ix]] + ([N[ix].ravel()] if hardneg else []))
        uniq = np.unique(ids)
        try:
            enc.model.train()
            z = enc.embed([texts[i] for i in uniq])
            z.pow(2).mean().backward()
            for p in params:
                p.grad = None
            return bs
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            for p in params:
                p.grad = None
            if enc.device == "cuda":
                torch.cuda.empty_cache()
            bs //= 2
            log(f"      [encoder] out of memory on the longest pairs -> batch size {bs}")
    raise RuntimeError("cannot fit a batch of 4 pairs on the device")


def train_encoder(enc, texts, pairs, E0, mode="lora", mrl=False, hardneg=True, epochs=1, bs=32,
                  lr=2e-4, tau=0.05, seed=0, r=16, val_ids=None, val_fn=None, max_steps=None, deadline=None, log=print):
    """Contrastive LoRA/DoRA training with the encoder in the loop.

    After every epoch only the articles needed for validation are re-encoded (``val_ids``);
    epoch 0 (= the frozen embedding) competes, and the adapter weights of the best epoch are
    restored before one final pass over *all* articles.  Returns ``(E_final, info)``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_train = enc.attach_peft(mode, r)
    log(f"      [{mode} r={r}] trainable params: {n_train:,}")
    named = [(n, p) for n, p in enc.model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=enc.amp)
    A, P, N = pairs
    bs = fit_encoder_batch(enc, texts, pairs, bs, hardneg, params, log)
    dims = mrl_dims(E0.shape[1]) if mrl else None
    best_score, best_ep, best_state = (val_fn(E0) if val_fn else -math.inf), 0, None
    step, t0 = 0, time.time()
    for ep in range(1, epochs + 1):
        enc.model.train()
        perm = rng.permutation(len(A))
        tot, nb = 0.0, 0
        for s in range(0, len(A), bs):
            ix = perm[s:s + bs]
            ids = np.concatenate([A[ix], P[ix]] + ([N[ix].ravel()] if hardneg else []))
            uniq, inv = np.unique(ids, return_inverse=True)
            z = enc.embed([texts[i] for i in uniq])
            inv = torch.from_numpy(inv).to(z.device)
            b = len(ix)
            a, p = z[inv[:b]], z[inv[b:2 * b]]
            n = z[inv[2 * b:]].view(b, -1, z.shape[-1]) if hardneg else None
            loss = contrastive_loss(a, p, torch.from_numpy(P[ix]).to(z.device), n, tau, dims)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); nb += 1; step += 1
            if step % 100 == 0:
                log(f"        step {step} loss={tot / nb:.4f} ({time.time() - t0:.0f}s)")
            if max_steps and step >= max_steps:
                break
            if deadline is not None and time.time() > deadline:
                log(f"      [{mode}] deadline reached at step {step}")
                break
        if val_fn is not None and val_ids is not None:
            Ea = E0.copy()
            Ea[val_ids] = enc.encode([texts[i] for i in val_ids])
            score = val_fn(Ea)
        else:
            score = float(ep)                                              # no validation signal: keep the last epoch
        log(f"      [{mode}] epoch {ep} loss={tot / max(nb, 1):.4f} val={score:.4f}")
        if score > best_score:
            best_score, best_ep = score, ep
            best_state = {n: p.detach().clone() for n, p in named}
        if (max_steps and step >= max_steps) or (deadline is not None and time.time() > deadline):
            break
    info = {"best_epoch": best_ep, "val": float(best_score), "steps": step, "trainable": n_train}
    if best_ep == 0:
        return E0.copy(), info                                             # adaptation did not beat frozen on validation
    with torch.no_grad():
        for n, p in named:
            p.copy_(best_state[n])
    E = E0.copy()
    E[1:] = enc.encode(texts[1:])
    return E, info
