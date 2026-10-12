"""Unfreeze the top layers of the article encoder under the learned reader -- PLM-NR's recipe, made cheap enough for one T4 (stage ``finetune``).

Why this and not more stacking: in the measured v16b run the stacker is saturated (``stack_llm`` = the 4-weight linear fusion, ``stack_meta`` moves +0.0008 while the
content-only fusion moves +0.0105) and the frozen-vector readers are the best components (concat 0.4223 > BGE-large 0.4152 > Qwen3 0.4050 nDCG@10).  What is left is the
content / personalisation model itself, which the literature trains end to end: PLM-NR (Wu et al., SIGIR'21) fine-tunes only the LAST TWO transformer layers of the
pre-trained model because tuning all of them differs very little (search-result summary, PDF not opened), and a frozen 175B LM is beaten by a tuned 125M one on MIND
(arXiv 2305.11700, same caveat).  Our own earlier evidence: the LoRA tower of v14 matched zero-shot pooling after 0.5% of the data with validation still rising.

How it is made cheap.  The lower layers are a fixed function of the text, so their output (``[articles, L, d]`` float16: 20 GB for 104k articles at L=96, d=1024) is
computed ONCE and kept on disk (read with ``pread``, never memory-mapped, so it does not count as resident memory); a training step runs only the top layers, on the unique
CANDIDATES of the batch (gradient).  The history side reads a table of article vectors that is refreshed with the current top layers every ``--ft-refresh`` steps
(stop-gradient: the user side sees the encoder as it was a few hundred steps ago -- the SpeedyFeed idea of reusing intermediate encodings).  Training starts from the
reader that has been trained on the frozen table, so the comparison is ``same model, top layers unfrozen`` (``ft_base`` -> ``ft_reader``): any difference is due to
unfreezing, not to a different training budget.  Checkpoints are chosen on the validation mean re-balanced to the test split's cold share (82% of its clicks are on new
articles; memorising popular articles would raise the plain validation mean), the test split is touched once per row, and a gate (``--ft-min-gain``) says whether the
direction is worth a full run.

Limits (also in the report): only the candidate side receives gradient; the cache truncates texts at ``--ft-maxlen`` tokens (the frozen vectors used 256); one seed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import adapt
import common
import usermodel as UM
from common import log


# ================================================================================================ the pieces
MASK = {"mode": "auto"}                                                           # "auto": the library's own builder when it has one; "additive": the classic extended mask


def attention_mask_for(config, valid, hidden):
    """Padding mask in the format the encoder layers of the INSTALLED transformers want (``layer(h, attention_mask=m)`` is the same call in 4.x and 5.x, the mask is
    not): 5.x builds a 4-D boolean mask with ``create_bidirectional_mask`` (``None`` when nothing is padded), 4.x an additive one.  The stage checks the result against the
    full encoder and falls back to the additive mask if the library's builder does not fit its layers."""
    valid = valid.to(torch.long)
    if MASK["mode"] == "auto":
        try:
            from transformers.masking_utils import create_bidirectional_mask
        except ImportError:
            pass
        else:
            return create_bidirectional_mask(config=config, inputs_embeds=hidden, attention_mask=valid)
    return (1.0 - valid[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min


def encoder_layers(model):
    """The ``nn.ModuleList`` of transformer layers of a BERT-like model (BERT, RoBERTa, XLM-R, MPNet, ...), else ``None``."""
    layers = getattr(getattr(model, "encoder", None), "layer", None)
    return layers if isinstance(layers, nn.ModuleList) else None


def autocast_on(device):
    """Is autocast active for ``device``?  (``torch.is_autocast_enabled`` takes a device type from torch 2.4 on and none before.)"""
    try:
        return bool(torch.is_autocast_enabled(torch.device(device).type))
    except TypeError:
        return bool(torch.is_autocast_enabled())


class TopEncoder(nn.Module):
    """The last layers of a BERT-like encoder + pooling + L2 normalisation: hidden states ``[B, L, d]`` entering the first of them and their lengths -> unit vectors ``[B, d]``."""

    def __init__(self, layers, config, pooling="cls"):
        super().__init__()
        self.layers, self.config, self.pooling = nn.ModuleList(layers), config, pooling

    def forward(self, x, lens):
        if not autocast_on(x.device):                                            # a float16 cache read without autocast (CPU): the weights are float32
            x = x.to(next(self.parameters()).dtype)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lens[:, None]
        mask = attention_mask_for(self.config, valid, x)
        h = x
        for layer in self.layers:
            out = layer(h, attention_mask=mask)
            h = out[0] if isinstance(out, tuple) else out
        if self.pooling == "cls":
            v = h[:, 0]
        else:                                                                    # "mean": the masked mean of the tokens
            m = valid.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(v.float(), dim=-1)


class FTReader(UM.LLMEncCA):
    """``LLMEncCA`` whose scoring can start from raw article vectors (so the candidate vectors can come from the trainable encoder).  ``score`` is unchanged."""

    def score_raw(self, h_raw, hist_mask, c_raw):
        h, c = self.head(h_raw), self.head(c_raw)
        att = torch.einsum("bcd,bhd->bch", self.cand_proj(c), h) / self.scale
        att = torch.softmax(att.masked_fill(~hist_mask[:, None, :], -1e9), -1)
        return (torch.einsum("bch,bhd->bcd", att, h) * c).sum(-1)


# ================================================================================================ the cache of the lower layers
class HiddenCache:
    """Hidden states of every article at the cut, ``[n, L, d]`` in one flat file, plus the true token counts.  Rows are read with ``pread`` (random access for a batch,
    one contiguous read for a refresh pass): the file is never mapped, so it does not show up in the process's resident memory."""

    FILE, LENS, META = "hidden.bin", "lens.npy", "meta.json"

    def __init__(self, directory, n, L, d, dtype=np.float16, tag=None):
        self.dir, self.n, self.L, self.d, self.dtype, self.tag = directory, int(n), int(L), int(d), np.dtype(dtype), dict(tag or {})
        self.row_bytes = self.L * self.d * self.dtype.itemsize
        self.path = os.path.join(directory, self.FILE)
        self.lens, self._fd = None, None

    @property
    def nbytes(self):
        return self.n * self.row_bytes

    def _meta(self):
        return {"n": self.n, "L": self.L, "d": self.d, "dtype": self.dtype.name, **self.tag}

    def is_complete(self):
        """A finished cache of exactly this model / texts / cut (a half-written one, or one made for other settings, is rebuilt)."""
        try:
            with open(os.path.join(self.dir, self.META), encoding="utf-8") as f:
                m = json.load(f)
            return (bool(m.get("complete")) and all(m.get(k) == v for k, v in self._meta().items()) and os.path.getsize(self.path) == self.nbytes
                    and os.path.exists(os.path.join(self.dir, self.LENS)))
        except (OSError, ValueError):
            return False

    def open(self):
        self.lens = np.load(os.path.join(self.dir, self.LENS)).astype(np.int64)
        self._fd = os.open(self.path, os.O_RDONLY)
        return self

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _read_into(self, view, offset):
        got = os.preadv(self._fd, [view], offset) if hasattr(os, "preadv") else None
        if got is None:
            data = os.pread(self._fd, len(view), offset)
            view[:len(data)] = data
            got = len(data)
        if got != len(view):
            raise OSError(f"short read from {self.path}: {got} of {len(view)} bytes at {offset}")

    def rows(self, ids):
        """``[len(ids), L, d]`` for article rows ``ids`` (0-based)."""
        out = np.empty((len(ids), self.L, self.d), self.dtype)
        for k, i in enumerate(ids):
            self._read_into(memoryview(out[k]).cast("B"), int(i) * self.row_bytes)
        return out

    def chunk(self, a, b):
        """Rows ``a .. b-1`` in one contiguous read."""
        out = np.empty((b - a, self.L, self.d), self.dtype)
        self._read_into(memoryview(out).cast("B"), a * self.row_bytes)
        return out

    def build(self, model, tok, texts, prefix, cut, device, amp=True, bs=128, logf=log):
        """Run the lower ``cut`` layers over every text once (shortest texts first, so little padding) and write the rows at their article offsets.  Returns seconds."""
        t0 = time.time()
        os.makedirs(self.dir, exist_ok=True)
        meta_path = os.path.join(self.dir, self.META)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        os.ftruncate(fd, self.nbytes)
        lens = np.zeros(self.n, np.int16)
        order = np.argsort([len(t) for t in texts], kind="stable")
        layers = encoder_layers(model)
        model.encoder.layer = nn.ModuleList(list(layers)[:cut])               # weights are shared; the full stack is put back below
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                for s in range(0, self.n, bs):
                    ix = order[s:s + bs]
                    batch = tok([prefix + texts[i] for i in ix], padding=True, truncation=True, max_length=self.L, return_tensors="pt").to(device)
                    kw = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp and str(device).startswith("cuda")):
                        h = model(**kw).last_hidden_state
                    h = h.to(torch.float32).cpu().numpy().astype(self.dtype)
                    ln = batch["attention_mask"].sum(1).cpu().numpy()
                    block = np.zeros((len(ix), self.L, self.d), self.dtype)
                    block[:, :h.shape[1]] = h
                    for k, i in enumerate(ix):
                        block[k, ln[k]:] = 0                                   # nothing from padding positions
                        os.pwrite(fd, block[k].tobytes(), int(i) * self.row_bytes)
                    lens[ix] = ln
                    if (s // bs) % 100 == 0 and s:
                        logf(f"    hidden states: {s:,}/{self.n:,} articles ({time.time() - t0:.0f}s)")
        finally:
            model.encoder.layer = layers
            model.train(was)
            os.close(fd)
        np.save(os.path.join(self.dir, self.LENS), lens)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({**self._meta(), "complete": True}, f)
        return time.time() - t0


@torch.no_grad()
def refresh_table(top, cache, device, n_news, bs=256, amp=True):
    """Article vectors of ALL articles from the current top layers -> ``[n_news, d]`` float32 on ``device`` (row 0 = padding = zeros); the next chunk is read from
    disk while the current one is on the GPU."""
    was = top.training
    top.eval()
    table = torch.zeros(n_news, cache.d, device=device)
    spans = [(a, min(a + bs, cache.n)) for a in range(0, cache.n, bs)]
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(cache.chunk, *spans[0]) if spans else None
        for k, (a, b) in enumerate(spans):
            x = fut.result()
            fut = ex.submit(cache.chunk, *spans[k + 1]) if k + 1 < len(spans) else None
            lens = torch.from_numpy(cache.lens[a:b]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and str(device).startswith("cuda")):
                v = top(torch.from_numpy(x).to(device), lens)
            table[1 + a:1 + b] = v
    top.train(was)
    return table


def check_equivalence(model, tok, top, cache, texts, prefix, pooling, device, amp=True, n=48):
    """The cached path (hidden states at the cut -> top layers) must reproduce the full encoder on articles that fit in ``L`` tokens: (min, mean) cosine."""
    short = np.nonzero(cache.lens < cache.L)[0]
    ids = short[np.linspace(0, len(short) - 1, min(n, len(short))).astype(int)] if len(short) else np.zeros(0, int)
    if not len(ids):
        return 0.0, 0.0
    was = model.training
    model.eval()
    with torch.no_grad():
        batch = tok([prefix + texts[i] for i in ids], padding=True, truncation=True, max_length=cache.L, return_tensors="pt").to(device)
        kw = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and str(device).startswith("cuda")):
            h = model(**kw).last_hidden_state
        if pooling == "cls":
            full = h[:, 0]
        else:
            m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
            full = (h * m).sum(1) / m.sum(1).clamp(min=1)
        full = F.normalize(full.float(), dim=-1)
        lens = torch.from_numpy(cache.lens[ids]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and str(device).startswith("cuda")):
            cached = top(torch.from_numpy(cache.rows(ids)).to(device), lens)
    model.train(was)
    cos = (full * cached).sum(1)
    return float(cos.min()), float(cos.mean())


# ================================================================================================ the stage
def unsupported(args):
    """Why this run cannot unfreeze layers (``None`` if it can)."""
    if str(args.encoder).startswith("concat:"):
        return "the composite encoder 'concat:A+B' has no layers of its own"
    if getattr(args, "news_emb", None):
        return "the article vectors come from a file (--news-emb): there is no encoder to unfreeze"
    pooling = adapt.guess_pooling(args.encoder)
    if pooling not in ("cls", "mean"):
        return f"'{args.encoder}' pools with '{pooling}' (decoder-only): only BERT-like encoders with CLS / mean pooling are supported"
    return None


def ft_signature(args):
    """What the finetune stage depends on besides the data and the reader settings: its own arguments (a changed one re-runs this stage alone)."""
    a = {k: v for k, v in sorted(vars(args).items()) if k.startswith("ft_") and k != "ft_cache_dir"}
    return hashlib.sha1(json.dumps(a, sort_keys=True, default=str).encode()).hexdigest()


def gate_verdict(gain_w, gain, min_gain):
    """PASS: the cold-weighted validation mean rose by at least ``min_gain`` and the plain one did not fall; INCONCLUSIVE: it rose by at least a third of it;
    FAIL: otherwise.  Decided on validation only -- the test split is reported, never used to decide."""
    if gain_w >= min_gain and gain > -0.001:
        return "PASS"
    if gain_w >= min_gain / 3:
        return "INCONCLUSIVE"
    return "FAIL"


def stage_finetune(S, args):
    from stages_ext import short                                                 # lazy: stages_ext imports this module's siblings
    k_layers = int(args.ft_layers)
    log(f"\n== FT  unfreeze the top {k_layers} layer(s) of {short(args.encoder)} under the learned reader (PLM-NR recipe; the lower layers' hidden states are cached on disk) ==")
    why = unsupported(args)
    if why:
        log(f"  not run: {why}")
        return False
    dev = S.device
    amp = str(dev).startswith("cuda")
    L, n_art = int(args.ft_maxlen), S.data.n_news - 1
    texts = list(S.meta["text"])
    assert len(texts) == n_art, "article order mismatch between loader and metadata"
    enc = adapt.HFEncoder(args.encoder, max_len=L, device=dev, prefix=args.doc_prefix)
    layers = encoder_layers(enc.model)
    if layers is None:
        log(f"  not run: '{args.encoder}' has no encoder.layer list to cut")
        return False
    n_layers = len(layers)
    k_layers = max(1, min(k_layers, n_layers))
    cut = n_layers - k_layers
    d = enc.model.config.hidden_size
    sha = hashlib.sha1("\n".join(texts[:2000] + texts[-2000:]).encode("utf-8")).hexdigest()[:16]
    cdir = args.ft_cache_dir or os.path.join(S.cache_dir, "ft_hidden")
    cdir = os.path.join(cdir, f"{short(args.encoder)}_cut{cut}_L{L}_n{n_art}_{sha}")
    cache = HiddenCache(cdir, n_art, L, d, np.float16 if amp else np.float32, {"model": args.encoder, "cut": cut, "prefix": args.doc_prefix, "texts": sha})
    t_stage = time.time()
    if cache.is_complete():
        log(f"  hidden states of the lower {cut} layers: reused from {cdir}")
    else:
        os.makedirs(cdir, exist_ok=True)
        free = shutil.disk_usage(cdir).free
        if free < 1.1 * cache.nbytes:
            log(f"  not run: the cache needs {cache.nbytes / 1e9:.1f} GB and {cdir} has {free / 1e9:.1f} GB free")
            return False
        secs = cache.build(enc.model, enc.tok, texts, args.doc_prefix, cut, dev, amp=amp, bs=128)
        log(f"  hidden states of the lower {cut} layers cached: {n_art:,} x {L} x {d} {cache.dtype.name} = {cache.nbytes / 1e9:.1f} GB in {secs:.0f}s ({cdir})")
    cache.open()
    top = TopEncoder([copy.deepcopy(x) for x in list(layers)[cut:]], enc.model.config, enc.pooling).to(dev)
    cmin, cmean = check_equivalence(enc.model, enc.tok, top, cache, texts, args.doc_prefix, enc.pooling, dev, amp=amp)
    if cmin < 0.99 and MASK["mode"] == "auto":                                   # the library's mask builder may not match its own layers: the classic additive mask
        MASK["mode"] = "additive"
        cmin, cmean = check_equivalence(enc.model, enc.tok, top, cache, texts, args.doc_prefix, enc.pooling, dev, amp=amp)
        log(f"  retried with the additive attention mask: cosine min {cmin:.4f} mean {cmean:.4f}")
        if cmin < 0.99:
            MASK["mode"] = "auto"
    cut_frac = float((cache.lens >= L).mean())
    log(f"  cached path vs the full encoder on articles that fit in {L} tokens: cosine min {cmin:.4f} mean {cmean:.4f}; {cut_frac:.1%} of the articles are truncated at {L} tokens")
    S.note_check("finetune: cached top layers == full encoder", cmin >= 0.99, f"cos min {cmin:.4f} mean {cmean:.4f} ({'fp16' if amp else 'fp32'} cache)")
    if cmin < 0.99:
        log("  not run: the cached path does not reproduce the encoder (a transformers version difference in the layer call?) -- training on it would be meaningless")
        cache.close()
        return False
    del enc, layers
    if amp:
        torch.cuda.empty_cache()

    pack = S.store.get("um_pack") or UM.pack_train(S.data.train_core)
    S.store["um_pack"] = pack
    H_np, M_np, C_np = pack
    val_w_all = S.val_weights()

    def val_scores(reader, E_np):
        res = S.run(UM.scorer(reader, E_np), "val")
        return res["mean"]["ndcg@10"], common.weighted_mean(res["arr"]["ndcg@10"], val_w_all[res["ids"]])

    # ---- phase 0: the reader on the FROZEN top layers = the common starting point of both arms
    T0 = refresh_table(top, cache, dev, S.data.n_news, amp=amp)
    T0_np = T0.cpu().numpy()
    t0 = time.time()
    base, info0 = UM.train_user_model(T0_np, pack, lambda m: S.run(UM.scorer(m), "val")["mean"]["ndcg@10"], seed=0, max_epochs=args.um_epochs, device=dev,
                                      dim=getattr(args, "um_dim", 64), deadline=S.deadline(0.2), log=lambda *a: None, kind="ca")
    reader0 = FTReader(T0_np, dim=getattr(args, "um_dim", 64), device=dev)
    reader0.load_state_dict(base.state_dict())
    reader0.eval()
    del base
    v0, vw0 = val_scores(reader0, T0_np)
    log(f"  phase 0: reader on the frozen top layers: {info0['epochs']} epochs, val nDCG@10={v0:.4f} (cold-weighted {vw0:.4f}) ({time.time() - t0:.0f}s)")
    start = {"step": 0, "val": v0, "val_w": vw0, "top": copy.deepcopy(top.state_dict()), "reader": copy.deepcopy(reader0.state_dict())}

    # ---- phase 1: two arms from that start, the same samples, optimiser and validation schedule.  The reader of phase 0 is often NOT converged (it stops at the epoch cap),
    # so training it further helps by itself; only the difference between the arms is due to unfreezing.
    steps = int(args.ft_steps)
    arm_b = _run_arm(S, args, True, top, start, T0, T0_np, cache, pack, val_scores, steps, amp, deadline=S.deadline(cap_s=float(args.ft_minutes) * 60), label="unfrozen")
    arm_a = _run_arm(S, args, False, top, start, T0, T0_np, cache, pack, val_scores, arm_b["steps_run"], amp, deadline=None, label="frozen (control)")

    # ---- the two best checkpoints, each evaluated once on validation and test
    reader_a = FTReader(T0_np, dim=getattr(args, "um_dim", 64), device=dev)
    reader_a.load_state_dict(arm_a["best"]["reader"])
    reader_a.eval()
    res_va, _ = S.run_rec(UM.scorer(reader_a, T0_np), "val")
    res_ta, _ = S.run_rec(UM.scorer(reader_a, T0_np), "test")
    ref_name = "um_frozen" if "um_frozen" in S.cache else "ft_base"
    S.record("ft_base", "FT-control", {"layers": k_layers, "frozen": True, "max_len": L, "steps": arm_a["best"]["step"], "reader": "ca"}, res_va, res_ta, ref_name,
             {"epochs": [info0["epochs"]], "seeds": 1, "best_step": arm_a["best"]["step"], "steps_run": arm_a["steps_run"]})
    top.load_state_dict(arm_b["best"]["top"])
    reader_b = FTReader(T0_np, dim=getattr(args, "um_dim", 64), device=dev)
    reader_b.load_state_dict(arm_b["best"]["reader"])
    reader_b.eval()
    T = refresh_table(top, cache, dev, S.data.n_news, amp=amp)
    T_np = T.cpu().numpy()
    res_v, vr = S.run_rec(UM.scorer(reader_b, T_np), "val")
    res_t, tr = S.run_rec(UM.scorer(reader_b, T_np), "test")
    row = S.record("ft_reader", "H12-finetune", {"layers": k_layers, "max_len": L, "steps": arm_b["best"]["step"], "batch": int(args.ft_batch), "lr": float(args.ft_lr), "reader": "ca"},
                   res_v, res_t, "ft_base", {"seeds": 1, "best_step": arm_b["best"]["step"], "steps_run": arm_b["steps_run"], "stopped": arm_b["stop"] or "step budget"})
    S.export_scores("ft_reader", vr, tr, {"family": "H12-finetune", "seeds": 1, "kind": "reader"})
    np.save(os.path.join(S.work, "ft_vectors.npy"), T_np[1:].astype(np.float16))
    wv = lambda res: (res["mean"]["ndcg@10"], common.weighted_mean(res["arr"]["ndcg@10"], val_w_all[res["ids"]]))
    (va, vwa), (vb, vwb) = wv(res_va), wv(res_v)
    gain, gain_w = vb - va, vwb - vwa
    d = row.get("d_ndcg@10")
    verdict = gate_verdict(gain_w, gain, float(args.ft_min_gain))
    gate = {"verdict": verdict, "layers": k_layers, "max_len": L, "min_gain": float(args.ft_min_gain), "val_start": v0, "val_w_start": vw0,
            "val_frozen": va, "val_w_frozen": vwa, "val_best": vb, "val_w_best": vwb, "gain": gain, "gain_w": gain_w,
            "best_step": arm_b["best"]["step"], "steps_run": arm_b["steps_run"], "stopped": arm_b["stop"] or "step budget", "test_delta": list(d) if d else None,
            "cos_min": cmin, "truncated_frac": cut_frac, "seconds": time.time() - t_stage, "curve": arm_b["curve"], "curve_frozen": arm_a["curve"]}
    S.kv["ft_gate"] = gate
    write_gate_report(S, gate, row)
    log(f"  GATE {verdict}: cold-weighted validation, unfrozen {vwb:.4f} vs the same reader trained as long with the encoder frozen {vwa:.4f} ({gain_w:+.4f}; needs {args.ft_min_gain:+.4f}); "
        f"plain {vb:.4f} vs {va:.4f} ({gain:+.4f})" + (f" | test Δ nDCG@10 {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" if d else ""))
    cache.close()
    del top, reader0, reader_a, reader_b, T, T0
    if amp:
        torch.cuda.empty_cache()
    return True


def _run_arm(S, args, unfreeze, top, start, T0, T0_np, cache, pack, val_scores, steps, amp, deadline, label):
    """One arm of phase 1: ``unfreeze`` trains the top layers on the candidate vectors (the history side reads a table refreshed every ``--ft-refresh`` steps), the control
    trains the reader alone on the frozen table.  Both start from ``start``, draw the same samples in the same order and keep the best validation checkpoint."""
    dev = S.device
    top.load_state_dict(start["top"])
    reader = FTReader(T0_np, dim=getattr(args, "um_dim", 64), device=dev)
    reader.load_state_dict(start["reader"])
    B = int(args.ft_batch)
    refresh_every = max(int(args.ft_refresh), 1)
    eval_every = max(int(args.ft_eval_every) // refresh_every, 1) * refresh_every
    warm = min(200, max(steps // 20, 1))
    H_np, M_np, C_np = pack
    H, M = torch.from_numpy(H_np).to(dev), torch.from_numpy(M_np).to(dev)
    C = torch.from_numpy(C_np).to(dev)
    groups = [{"params": list(reader.parameters()), "lr": float(args.ft_head_lr), "weight_decay": 1e-5}]
    if unfreeze:
        groups.insert(0, {"params": list(top.parameters()), "lr": float(args.ft_lr), "weight_decay": 0.01})
    opt = torch.optim.AdamW(groups)
    lr0 = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=amp and unfreeze)
    target = torch.zeros(B, dtype=torch.long, device=dev)
    rng = np.random.default_rng(1)
    N = len(H_np)
    state = {"perm": rng.permutation(N), "ptr": 0}

    def next_ix():
        if state["ptr"] + B > N:
            state["perm"], state["ptr"] = rng.permutation(N), 0
        ix = state["perm"][state["ptr"]:state["ptr"] + B]
        state["ptr"] += B
        return ix

    def load(ix):                                                                # runs in the IO thread: the unique candidates of the batch and their hidden states
        c = C_np[ix]
        uniq, inv = np.unique(c, return_inverse=True)
        return ix, uniq, inv.reshape(c.shape), (cache.rows(uniq - 1) if unfreeze else None)

    T = T0
    best = {k: start[k] for k in ("step", "val", "val_w")} | {"top": start["top"], "reader": start["reader"]}
    curve = [{"step": 0, "val": start["val"], "val_w": start["val_w"], "seconds": 0.0}]
    t_train, losses, last_gain_step, stop, step, t_refresh = time.time(), [], 0, "", 0, 0.0
    top.train(unfreeze)
    reader.train()
    log(f"  phase 1, {label}: up to {steps:,} steps x {B} samples; " + (f"top layers lr {args.ft_lr:g}, " if unfreeze else "encoder frozen, ")
        + f"reader lr {args.ft_head_lr:g}; " + (f"history table refreshed every {refresh_every}, " if unfreeze else "") + f"validated every {eval_every} steps"
        + (f"; stops after {args.ft_patience} validations without progress or {args.ft_minutes:g} minutes" if unfreeze else "; the same number of steps as the unfrozen arm"))
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(load, next_ix())
        for step in range(1, steps + 1):
            ix, uniq, inv, x = fut.result()
            fut = ex.submit(load, next_ix()) if step < steps else None
            for g, base_lr in zip(opt.param_groups, lr0):
                g["lr"] = base_lr * min(1.0, step / warm)
            ixt = torch.from_numpy(ix).to(dev)
            if unfreeze:
                lens = torch.from_numpy(cache.lens[uniq - 1]).to(dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    ev = top(torch.from_numpy(x).to(dev), lens)                   # [U, d] float32: the candidates, WITH gradient
                c_raw = ev[torch.from_numpy(inv).to(dev)]                          # [B, 5, d]
            else:
                c_raw = T0[C[ixt]]
            h_raw = T[H[ixt]]                                                      # [B, 50, d]: the history from the (refreshed) table, no gradient
            logits = reader.score_raw(h_raw, M[ixt], c_raw)
            loss = F.cross_entropy(logits, target[:len(ix)])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            if not np.isfinite(losses[-1]):
                stop = "non-finite loss"
                break
            if step % refresh_every == 0 or step == steps:
                if unfreeze:
                    t_r = time.time()
                    T = refresh_table(top, cache, dev, S.data.n_news, amp=amp)
                    t_refresh += time.time() - t_r
                    if step == refresh_every:
                        log(f"    one refresh of the history table takes {time.time() - t_r:.0f}s")
                    if t_refresh > 0.35 * (time.time() - t_train) and refresh_every < 4096:      # a slow disk: refresh less often rather than spend the budget reading
                        refresh_every *= 2
                        eval_every = -(-eval_every // refresh_every) * refresh_every
                        log(f"    refreshing takes {t_refresh / (time.time() - t_train):.0%} of the time -> every {refresh_every} steps from now on (validated every {eval_every})")
                if step % eval_every == 0 or step == steps:
                    reader.eval()
                    v, vw = val_scores(reader, T.cpu().numpy())
                    reader.train()
                    curve.append({"step": step, "val": v, "val_w": vw, "seconds": time.time() - t_train})
                    improved = vw > best["val_w"]
                    if improved:
                        best = {"step": step, "val": v, "val_w": vw, "top": copy.deepcopy(top.state_dict()), "reader": copy.deepcopy(reader.state_dict())}
                        last_gain_step = step
                    log(f"    [{label}] step {step:6d}: loss {np.mean(losses[-eval_every:]):.4f} | val nDCG@10 {v:.4f} (cold-weighted {vw:.4f}, {vw - start['val_w']:+.4f} vs the start)"
                        f"{' *' if improved else ''} ({time.time() - t_train:.0f}s)")
                    if unfreeze and (step - last_gain_step) >= int(args.ft_patience) * eval_every:
                        stop = f"no progress for {args.ft_patience} validations"
                        break
            if deadline is not None and time.time() > deadline:
                stop = "time limit"
                break
    log(f"  [{label}] ended at step {step:,}{f' ({stop})' if stop else ''}; best validation at step {best['step']:,} (cold-weighted {best['val_w']:.4f})")
    reader.eval()
    return {"best": best, "curve": curve, "steps_run": int(step), "stop": stop, "seconds": time.time() - t_train}


def write_gate_report(S, gate, row):
    """``ft_gate.md`` in the run's directory: the verdict, what it was decided on and the validation curves of both arms (the notebook adds it to the summary)."""
    d = gate["test_delta"]
    lines = [f"# Fine-tuning the top {gate['layers']} layer(s) under the learned reader: gate {gate['verdict']}", "",
             f"Decided on VALIDATION only, on the mean re-balanced to the test split's cold share: the unfrozen arm's best checkpoint {gate['val_w_best']:.4f} against the same reader "
             f"trained for as many steps with the encoder frozen {gate['val_w_frozen']:.4f} = {gate['gain_w']:+.4f} (the gate needs {gate['min_gain']:+.4f}); plain mean "
             f"{gate['val_best']:.4f} against {gate['val_frozen']:.4f} = {gate['gain']:+.4f}.  Both arms start from the reader trained on the frozen layers "
             f"({gate['val_w_start']:.4f} cold-weighted, {gate['val_start']:.4f} plain), draw the same samples and validate on the same schedule.  The unfrozen arm's best checkpoint is "
             f"at step {gate['best_step']:,} of {gate['steps_run']:,} run ({gate['stopped']}); {gate['seconds'] / 60:.0f} minutes in the stage.", "",
             "Test, touched once per row (`ft_reader` against `ft_base`; paired over the common impressions): "
             + (f"Δ nDCG@10 {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" if d else "not available") + ".  "
             f"`ft_reader` nDCG@10 {row['ndcg@10']:.4f}, AUC {row['auc']:.4f}, cold AUC {row.get('cold_auc', float('nan')):.4f}.", "",
             f"Cached path vs the full encoder: cosine min {gate['cos_min']:.4f}; {gate['truncated_frac']:.1%} of the articles are cut at {gate['max_len']} tokens "
             f"(the frozen vectors of the other rows read 256).", "",
             "| step | unfrozen: val nDCG@10 | cold-weighted | frozen control: val nDCG@10 | cold-weighted | seconds (unfrozen) |", "|---|---|---|---|---|---|"]
    fro = {c["step"]: c for c in gate["curve_frozen"]}
    for c in gate["curve"]:
        f = fro.get(c["step"])
        lines.append(f"| {c['step']:,} | {c['val']:.4f} | {c['val_w']:.4f} | " + (f"{f['val']:.4f} | {f['val_w']:.4f}" if f else "- | -") + f" | {c['seconds']:.0f} |")
    lines += ["", "Reading it: PASS = unfreezing helped by more than the noise of a single reader (two seeds of a frozen reader differ by 0.005 - 0.007 nDCG@10 on test) -> worth a full-scale "
              "run (more steps, more layers, both sides with gradient); INCONCLUSIVE = a gain below the threshold, look at the curves (the unfrozen one still rising at the end -> train "
              "longer); FAIL = no gain -> stop here and keep the frozen readers.", "",
              "Limits: one seed; only the candidate side receives gradient (the history side is a table refreshed every N steps); both checkpoints are selected on validation, so the "
              "validation numbers of both rows are slightly optimistic, like those of every early-stopped reader."]
    with open(os.path.join(S.work, "ft_gate.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
