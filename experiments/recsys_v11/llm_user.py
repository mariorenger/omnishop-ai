"""H5: a fine-tuned LLM as the *user encoder* (history-as-text), with causal / bidirectional / soft attention masks.

The reader's last K headlines become one text; a (decoder-only) LLM with LoRA/DoRA maps it to a vector that is
trained, contrastively, to be close to the embedding of the article the reader clicked next (frozen document
tower, in-batch + impression hard negatives).  Compared with H1 (the same model, zero-shot) this isolates what
*learning to read a history* adds.

Attention-mask study (arXiv 2602.10622, "How Do Decoder-Only LLMs Perceive Users?", Ant Group, 2/2026): a decoder-only LLM
reads the history causally, so early headlines never see later ones.  Three variants are trained with the same recipe:

  causal  the model's native mask                                     (last-token pooling)
  bidir   full bidirectional attention                                (LLM2Vec-style; 4D mask built here)
  soft    future positions are down-weighted by an additive ``log(lambda)`` bias, lambda ramped 0 -> 1 during
          the first half of training (linear causal -> bidirectional scheduler); evaluated at lambda = 1.

``soft`` is only the *scheduler* ingredient of that paper.  Its Gradient-Guided Soft Masking (a gradient-based
pre-warmup computed from a frozen left-tower encoder) and its hybrid mask are NOT implemented here.

Pooling is the last token in all three, so the mask is the only thing that differs.  Whether the loaded model
honours a custom 4D mask is checked at run time (``probe_masks``); a variant whose probe fails is skipped.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import adapt
from common import load_hf, log

MODES = ("causal", "bidir", "soft")


def build_mask(attn, mode, lam, dtype):
    """Additive 4D attention mask ``[B,1,T,T]`` for left-padded batches (``attn``: [B,T] 1 = real token).

    Padding keys are always masked; padding *queries* may attend to themselves so that no row is fully masked
    (their outputs are never used).  ``soft``: future keys stay visible but get ``log(lam)`` added."""
    B, T = attn.shape
    dev = attn.device
    valid = attn.bool()[:, None, None, :]                                      # [B,1,1,T] key is a real token
    tril = torch.ones(T, T, dtype=torch.bool, device=dev).tril()[None, None]   # key j <= query i
    eye = torch.eye(T, dtype=torch.bool, device=dev)[None, None]
    if mode == "causal":
        allow = (tril & valid) | eye
    else:
        allow = valid.expand(B, 1, T, T) | eye
    bias = torch.zeros(B, 1, T, T, dtype=torch.float32, device=dev)
    if mode == "soft" and lam < 1.0:
        bias = bias + torch.where(tril, 0.0, math.log(max(lam, 1e-6)))
    neg = torch.finfo(dtype).min
    return torch.where(allow, bias, torch.full_like(bias, neg)).to(dtype)


def peft_targets(model):
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    if "q_proj" in names:
        return [t for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if t in names]
    return [t for t in ("query", "key", "value") if t in names]


class UserTower:
    """LoRA/DoRA-adapted language model that maps a history text to one vector."""

    def __init__(self, name, mode="causal", prefix="", max_len=256, device="cpu", dtype=torch.float32,
                 trust_remote_code=False, grad_ckpt=True):
        from transformers import AutoModel, AutoTokenizer
        self.name, self.prefix, self.max_len, self.device = name, prefix, max_len, device
        self.pooling = adapt.guess_pooling(name)
        self.decoder = self.pooling == "last"
        self.mode = mode if self.decoder else "native"
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        if self.decoder:
            self.tok.padding_side = "left"
        self.model = load_hf(AutoModel, name, dtype, trust_remote_code).to(device)
        self.mask_dtype = self.model.dtype
        self.amp = device == "cuda" and dtype == torch.float32          # fp32 master weights + fp16 autocast
        self.grad_ckpt = grad_ckpt
        self.lam = 1.0
        self.peft = False

    # -- adaptation
    def attach_peft(self, kind="lora", r=16, dropout=0.05):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=dropout, target_modules=peft_targets(self.model),
                         use_dora=(kind == "dora"), bias="none")
        self.model = get_peft_model(self.model, cfg)
        self.peft = True
        if self.grad_ckpt:
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception:                                     # some wrappers expose it on the base model only
                self.model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def trainable(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    # -- forward
    def _tokenize(self, texts):
        return self.tok([self.prefix + t for t in texts], padding=True, truncation=True, max_length=self.max_len,
                        return_tensors="pt").to(self.device)

    def _hidden(self, batch, mode, lam):
        ids, attn = batch["input_ids"], batch["attention_mask"]
        kw = {"input_ids": ids}
        if "token_type_ids" in batch:
            kw["token_type_ids"] = batch["token_type_ids"]
        if self.decoder and mode != "causal":
            kw["attention_mask"] = build_mask(attn, mode, lam, self.mask_dtype)
        else:
            kw["attention_mask"] = attn
        if self.decoder:
            kw["use_cache"] = False
        return self.model(**kw).last_hidden_state

    def embed(self, texts, mode=None, lam=None):
        batch = self._tokenize(texts)
        mode = mode or self.mode
        lam = self.lam if lam is None else lam
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            h = self._hidden(batch, mode, lam)
        if self.pooling == "cls":
            z = h[:, 0]
        elif self.pooling == "last":
            z = h[:, -1]
        else:
            m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
            z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return z.float()

    @torch.no_grad()
    def encode(self, texts, bs=64, mode=None, lam=1.0, deadline=None):
        """Normalised vectors, sorted by length to keep padding small.  Un-encoded rows (deadline hit) stay zero."""
        was = self.model.training
        self.model.eval()
        order = np.argsort([len(t) for t in texts])
        out = None
        for s in range(0, len(texts), bs):
            if deadline is not None and time.time() > deadline:
                log("      [tower] deadline reached while encoding - remaining queries left at zero")
                break
            ix = order[s:s + bs]
            z = F.normalize(self.embed([texts[i] for i in ix], mode, lam), dim=-1).cpu().numpy()
            if out is None:
                out = np.zeros((len(texts), z.shape[1]), np.float32)
            out[ix] = z
        self.model.train(was)
        return out if out is not None else np.zeros((len(texts), 1), np.float32)


@torch.no_grad()
def probe_masks(tower, texts, tol=5e-3):
    """Does the loaded model honour a custom 4D attention mask?  Returns ``(ok, report)``.

    1. our 4D causal mask reproduces the model's native 2D-mask path (so building the mask by hand is correct);
    2. the bidirectional mask changes the output (it is not silently ignored);
    3. ``soft`` at lambda=1 equals ``bidir`` and at lambda->0 equals ``causal``."""
    was = tower.model.training
    tower.model.eval()
    batch = tower._tokenize(texts)

    def pooled(mode, lam=1.0, custom_causal=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=tower.amp):
            if custom_causal:
                ids, attn = batch["input_ids"], batch["attention_mask"]
                h = tower.model(input_ids=ids, attention_mask=build_mask(attn, "causal", 1.0, tower.mask_dtype),
                                use_cache=False).last_hidden_state
            else:
                h = tower._hidden(batch, mode, lam)
        return F.normalize(h[:, -1].float(), dim=-1)

    rep = {}
    try:
        native = pooled("causal")
        rep["causal_4d_vs_native"] = float((pooled("causal", custom_causal=True) - native).abs().max())
        bidir = pooled("bidir")
        rep["bidir_vs_causal"] = float((bidir - native).abs().max())
        rep["soft1_vs_bidir"] = float((pooled("soft", 1.0) - bidir).abs().max())
        rep["soft0_vs_causal"] = float((pooled("soft", 1e-6) - native).abs().max())
    except Exception as e:                                          # the model rejects 4D masks altogether
        tower.model.train(was)
        return False, {"error": repr(e)[:200]}
    tower.model.train(was)
    ok = (rep["causal_4d_vs_native"] < tol and rep["bidir_vs_causal"] > 1e-6
          and rep["soft1_vs_bidir"] < tol and rep["soft0_vs_causal"] < tol)
    return ok, rep


def fit_batch_size(tower, texts, bs, params, log=log):
    """Dry-run forward+backward on the longest queries; halve the batch while the GPU runs out of memory
    (a failure 30 minutes into training would cost the whole variant)."""
    longest = sorted(texts, key=len)[-bs:]
    while bs >= 2:
        try:
            tower.model.train()
            z = tower.embed(longest[:bs])
            z.float().pow(2).mean().backward()
            for p in params:
                p.grad = None
            return bs
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            for p in params:
                p.grad = None
            if tower.device == "cuda":
                torch.cuda.empty_cache()
            bs //= 2
            log(f"      [tower] out of memory on the longest queries -> batch size {bs}")
    raise RuntimeError("cannot fit even a batch of 2 queries on the device")


def train_user_tower(tower, query_fn, rows, E_doc, kind="lora", steps=1000, bs=16, lr=1e-4, tau=0.05, r=16,
                     seed=0, eval_fn=None, eval_at=(0.5, 1.0), ramp=0.5, warmup=50, deadline=None, log=log):
    """Contrastive fine-tuning of the user tower.

    ``query_fn(hist) -> str``: history ids -> query text;  ``rows``: train_core ``(user, hist, pos, negs)``;
    ``E_doc``: frozen document vectors [n_news, d] (torch, on the device).  ``eval_fn(tower) -> float`` (higher is better,
    typically validation nDCG@10) is called at the fractions ``eval_at`` of training and may leave anything it computed
    in ``tower.eval_payload``; the weights (and payload) of the best evaluation are restored/returned.  Returns ``info``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_train = tower.attach_peft(kind, r)
    log(f"      [{tower.mode}/{kind} r={r}] trainable params: {n_train:,}")
    named = tower.trainable()
    params = [p for _, p in named]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=tower.amp)
    usable = [i for i, rw in enumerate(rows) if len(rw[1])]
    order = rng.permutation(usable)
    order = np.concatenate([order] * math.ceil(steps * bs / max(len(order), 1)))[:steps * bs]
    sample = [query_fn(rows[i][1]) for i in order[:min(len(order), 4000)]]
    bs = fit_batch_size(tower, sample, bs, params, log)
    order = np.concatenate([order] * math.ceil(steps * bs / max(len(order), 1)))[:steps * bs]
    marks = sorted({max(1, int(round(f * steps))) for f in eval_at})
    best = (-math.inf, None, 0, None)
    tot, nb, first, t0, done, losses, bad = 0.0, 0, None, time.time(), 0, [], 0
    dims = None
    tower.model.train()
    for step in range(1, steps + 1):
        if deadline is not None and time.time() > deadline:
            log(f"      [tower] deadline reached at step {step - 1}/{steps}")
            break
        tower.lam = min(1.0, step / max(ramp * steps, 1)) if tower.mode == "soft" else 1.0
        ix = order[(step - 1) * bs:step * bs]
        sel = [rows[i] for i in ix]
        z = tower.embed([query_fn(rw[1]) for rw in sel])
        pos = torch.as_tensor([rw[2] for rw in sel], device=z.device)
        negs = torch.as_tensor(np.stack([np.asarray(rw[3], np.int64) for rw in sel]), device=z.device)
        loss = adapt.contrastive_loss(z, E_doc[pos], pos, E_doc[negs], tau, dims)
        if not torch.isfinite(loss):                                   # fp16 overflow in the LLM: skip the batch, abort if persistent
            bad += 1
            if bad > 5:
                raise FloatingPointError(f"non-finite loss in {bad} steps (fp16 overflow?) - try a smaller learning rate or fp32")
            opt.zero_grad()
            continue
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item(); nb += 1; done = step
        losses.append(loss.item())
        first = losses[0]
        if step % 50 == 0:
            log(f"        step {step}/{steps} loss={tot / nb:.4f} lam={tower.lam:.2f} ({time.time() - t0:.0f}s)")
            tot, nb = 0.0, 0
        if eval_fn is not None and step in marks:
            tower.lam = 1.0
            score = eval_fn(tower)
            log(f"      [{tower.mode}] step {step}: val={score:.4f}")
            if score > best[0]:
                best = (score, {n: p.detach().clone() for n, p in named}, step, getattr(tower, "eval_payload", None))
            tower.model.train()
    if eval_fn is not None and done not in marks:                  # stopped early by the deadline: evaluate what we have
        tower.lam = 1.0
        score = eval_fn(tower)
        log(f"      [{tower.mode}] step {done}: val={score:.4f}")
        if score > best[0]:
            best = (score, {n: p.detach().clone() for n, p in named}, done, getattr(tower, "eval_payload", None))
    if best[1] is not None:
        with torch.no_grad():
            for n, p in named:
                p.copy_(best[1][n])
    tower.lam = 1.0
    tower.model.eval()
    return {"best_step": best[2], "val": float(best[0]) if best[1] is not None else float("nan"), "steps": done, "trainable": n_train,
            "first_loss": first, "train_s": time.time() - t0, "losses": losses, "payload": best[3]}
