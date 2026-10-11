"""Helpers shared by the suite and its stages."""
from __future__ import annotations

import os
import re
import time

import numpy as np
import torch

import rep_eval as R

TASK = "Given the headlines a reader recently clicked, retrieve the news article they would most likely read next"


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ wall-clock budget
class Clock:
    """One wall-clock budget shared by every ``suite.main`` call of a notebook.  Long loops receive
    ``deadline`` (absolute ``time.time()``) and stop gracefully, so a slow GPU degrades the run instead of killing it."""

    def __init__(self):
        self.t0, self.budget = time.time(), None

    def start(self, hours):
        """The first call of a notebook fixes the budget; later ``suite.main`` calls share it."""
        if self.budget is None and hours:
            self.t0, self.budget = time.time(), float(hours) * 3600.0

    def reset(self):
        self.t0, self.budget = time.time(), None

    @property
    def deadline(self):
        return None if self.budget is None else self.t0 + self.budget

    def left(self):
        return float("inf") if self.budget is None else self.t0 + self.budget - time.time()

    def elapsed(self):
        return time.time() - self.t0


CLOCK = Clock()


def ram_total_gb():
    """Memory this process may really use: the machine's RAM, or the container's cgroup limit when that is lower (psutil reports the host's)."""
    total = 0.0
    try:
        import psutil
        total = psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and 0 < int(v) / 1e9 < (total or 1e9):
                total = int(v) / 1e9
        except Exception:
            pass
    return total


class PeakRAM:
    """Peak resident memory of this process since ``reset()``, sampled by a daemon thread (a few times per second).

    Up to v16 the suite printed the resident memory *at the end* of a stage.  A transient inside the stage never showed in it: the 17 GB (d=1024) to 34 GB
    (d=2048) of ``semgraph.Memory.knn_view`` was invisible in the log of the BGE-large run, and the "peak 16.2 GB" quoted for v15 was such a snapshot."""

    def __init__(self, every=0.25, alarm=0.7):
        self.every, self.alarm, self.peak, self._thread, self._proc = every, alarm, 0.0, None, None
        self.label, self._warned, self._limit = "", 0.0, 0.0

    def _rss(self):
        return self._proc.memory_info().rss if self._proc is not None else 0.0

    def start(self):
        if self._thread is not None:
            return
        try:
            import psutil
            import threading
            self._proc = psutil.Process()
        except Exception:
            return
        self.peak, self._limit = self._rss(), ram_total_gb() * 1e9
        main_id = threading.main_thread().ident

        def loop():
            while True:
                try:
                    rss = self._rss()
                    self.peak = max(self.peak, rss)
                    self._warn(rss, main_id)
                except Exception:
                    return
                time.sleep(self.every)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def _warn(self, rss, main_id):
        """Close to the memory this process may use, say where the main thread is -- once per 5% step.  An out-of-memory kill leaves no traceback, so the line
        printed here is the only trace of what was allocating (v16 died without one)."""
        limit = self._limit or ram_total_gb() * 1e9
        if not limit or rss < self.alarm * limit or rss < self._warned + 0.05 * limit:
            return
        self._warned = rss
        import sys
        import traceback
        frame = sys._current_frames().get(main_id)
        where = "".join(traceback.format_stack(frame, limit=6)[-4:]).rstrip() if frame is not None else "?"
        log(f"!! RAM {rss / 1e9:.1f} GB = {rss / limit:.0%} of the {limit / 1e9:.0f} GB this process can use (stage '{self.label}'); the main thread is at:\n{where}")

    def reset(self, label=""):
        self.start()
        self.peak, self.label, self._warned = self._rss(), label, 0.0

    def gb(self):
        return max(self.peak, self._rss()) / 1e9


PEAK = PeakRAM()


def lifetime_peak_gb():
    """Highest resident memory this process has ever had (the kernel's own high-water mark: it also catches a transient too short to be sampled)."""
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss                       # KiB on Linux, bytes on macOS
        return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9
    except Exception:
        return 0.0


def mem_note():
    """One short string with this process's resident memory (now, peak of the current stage, peak of the whole run), the machine's, and the GPU's peak
    allocation (empty when nothing can be measured)."""
    out = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        lim = ram_total_gb()
        out.append(f"RAM process {psutil.Process().memory_info().rss / 1e9:.1f} GB (stage peak {PEAK.gb():.1f}, run peak {lifetime_peak_gb():.1f}), "
                   f"machine {vm.used / 1e9:.1f}/{vm.total / 1e9:.0f} GB" + (f", container limit {lim:.0f} GB" if lim and lim < vm.total / 1e9 - 0.5 else ""))
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            out.append(f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    except Exception:
        pass
    return "; ".join(out)


def time_left():
    return CLOCK.left()


def out_of_time(margin=0.0):
    return CLOCK.left() < margin


# ------------------------------------------------------------------ prompts / pooling of results
def query_prefix(name, fmt="auto", instruct=True):
    low = name.lower()
    if fmt == "auto":
        fmt = "qwen3" if "qwen3" in low else "bge" if "bge" in low else "e5" if "e5" in low else "plain"
    if fmt == "qwen3":
        return f"Instruct: {TASK}\nQuery:" if instruct else ""
    if fmt == "bge":
        return "Represent this sentence for searching relevant passages: " if instruct else ""
    if fmt == "e5":
        return "query: "
    return ""


def avg_results(rs):
    """Average several seeds of the same variant (per-impression arrays and slice values)."""
    if len(rs) == 1:
        return rs[0]
    out = dict(rs[0])
    out["arr"] = {k: np.nanmean([r["arr"][k] for r in rs], axis=0) for k in R.METRICS}
    out["pos_auc"] = np.mean([r["pos_auc"] for r in rs], axis=0)
    m = {"auc": float(np.nanmean(out["arr"]["auc"])), **{k: float(np.mean(out["arr"][k])) for k in R.METRICS[1:]}}
    if "cold" in out:
        c = out["cold"]
        m["cold_auc"] = float(out["pos_auc"][c].mean()) if c.any() else float("nan")
        m["warm_auc"] = float(out["pos_auc"][~c].mean()) if (~c).any() else float("nan")
    out["mean"] = m
    return out


# ------------------------------------------------------------------ Hugging Face loading
def load_hf(auto_cls, name, dtype=None, trust_remote_code=False):
    """``from_pretrained`` that copes with the ``torch_dtype`` -> ``dtype`` rename between transformers versions
    and always ends up in the requested dtype."""
    kw = {"trust_remote_code": trust_remote_code}
    if dtype is None:
        return auto_cls.from_pretrained(name, **kw)
    try:
        model = auto_cls.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        model = auto_cls.from_pretrained(name, torch_dtype=dtype, **kw)
    return model if model.dtype == dtype else model.to(dtype)


def fix_peft_torchao():
    """Kaggle images ship torchao 0.10 while recent peft raises ``ImportError`` for torchao < 0.16 the moment a LoRA adapter is injected
    (``dispatch_torchao`` -> ``is_torchao_available``).  We never use torchao, so make peft treat it as absent.  Returns True if patched."""
    try:
        import peft.import_utils as iu
        iu.is_torchao_available()
        return False                                          # absent or recent enough: nothing to do
    except ImportError:
        pass
    except Exception:
        return False
    import importlib
    off = lambda *a, **k: False
    iu.is_torchao_available = off
    if hasattr(iu, "is_torchao_ge_v0_18_0"):
        iu.is_torchao_ge_v0_18_0 = off
    for mod in ("peft.tuners.lora.torchao", "peft.utils.quantization_utils"):
        try:
            m = importlib.import_module(mod)
            for name in ("is_torchao_available", "is_torchao_ge_v0_18_0"):
                if hasattr(m, name):
                    setattr(m, name, off)
        except Exception:
            pass
    log("      [peft] incompatible torchao detected -> treating it as absent (we do not use it)")
    return True


def decoder_only(name):
    """Last-token-pooled embedding models are decoder-only LLMs (Qwen3-Embedding, e5-mistral, ...)."""
    import adapt
    return adapt.guess_pooling(name) == "last"


# ------------------------------------------------------------------ evaluation subsets
def subset_evalset(rows, ids, device="cpu"):
    """EvalSet over ``rows[ids]`` that remembers the original positions (``.ids``) so that results over
    different subsets of the same split can be paired impression by impression."""
    ids = np.asarray(ids, np.int64)
    es = R.EvalSet([rows[i] for i in ids], device=device)
    es.ids = ids
    return es


def random_subset(n_total, n, seed=0):
    """Nested random subsets: ``random_subset(N, a) ⊂ random_subset(N, b)`` for a <= b (same seed)."""
    n = min(int(n), n_total)
    return np.sort(np.random.default_rng(seed).permutation(n_total)[:n])


# ------------------------------------------------------------------ score fusion and caching
def zscore(x, mask):
    """Standardise scores within each impression over its real candidates (same as V10 ``_znorm``)."""
    m = mask.float()
    n = m.sum(-1, keepdim=True).clamp(min=1)
    mu = (x * m).sum(-1, keepdim=True) / n
    sd = (((x - mu) * m) ** 2).sum(-1, keepdim=True).div(n).sqrt()
    return (x - mu) / (sd + 1e-6)


def fuse(parts):
    """Weighted sum of per-impression z-scored scorers: ``parts = [(scorer, weight), ...]``."""
    def f(b):
        return sum(w * zscore(s(b).float(), b["cand_mask"]) for s, w in parts)
    return f


def pop_scorer(b):
    """V10's online smoothed log-CTR of every candidate (counted only from strictly earlier impressions)."""
    return b["pop"]


class ScoreCache:
    """Raw per-candidate scores of an expensive scorer (SID language model, LLM tower, LLM judge), computed once
    per EvalSet so that fusion-weight grids are free."""

    def __init__(self, evalset, score_fn):
        self.n, self.rows = evalset.n, [None] * evalset.n
        for b in evalset:
            s = score_fn(b).float().cpu().numpy()
            cm = b["cand_mask"].cpu().numpy()
            for k, r in enumerate(b["rows"]):
                self.rows[r] = s[k, :int(cm[k].sum())].copy()

    def scorer(self, device, shift=None):
        """``shift``: optional callable ``(batch) -> [B,C]`` tensor subtracted from the cached scores."""
        def f(b):
            C = b["cand"].shape[1]
            out = np.zeros((len(b["rows"]), C), np.float32)
            for k, r in enumerate(b["rows"]):
                v = self.rows[r]
                out[k, :len(v)] = v
            out = torch.from_numpy(out).to(device)
            return out if shift is None else out - shift(b)
        return f


# ------------------------------------------------------------------ raw scores of a scorer, kept for the supermodel
class ScoreRecorder:
    """Wrap a scorer so that the raw per-candidate scores of an evaluation pass are kept without a second pass:
    ``rec = ScoreRecorder(scorer, evalset.n)``; evaluate with ``rec`` in place of the scorer; ``rec.ragged(result["ids"])`` afterwards."""

    def __init__(self, scorer, n):
        self.f, self.rows = scorer, [None] * n

    def __call__(self, b):
        s = self.f(b)
        arr = s.detach().float().cpu().numpy()
        cm = b["cand_mask"].cpu().numpy()
        for k, r in enumerate(b["rows"]):
            self.rows[r] = arr[k, :int(cm[k].sum())].astype(np.float32, copy=True)
        return s

    def ragged(self, ids):
        if any(r is None for r in self.rows):
            raise ValueError("not every impression was scored")
        return make_ragged(ids, self.rows)


def make_ragged(ids, rows):
    """``rows``: one 1-d float array per impression (candidate order of the data) -> ``{"ids", "ptr", "flat"}`` (CSR-like, float32)."""
    lens = np.fromiter((len(r) for r in rows), np.int64, len(rows))
    flat = np.concatenate(rows).astype(np.float32) if len(rows) else np.zeros(0, np.float32)
    return {"ids": np.asarray(ids, np.int64), "ptr": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), "flat": flat}


def ragged_scorer(rag, ids, device="cpu"):
    """Scorer for ``rep_eval.evaluate`` that replays stored scores.  ``rag`` holds the scores of the impressions ``rag["ids"]`` (strictly increasing);
    the EvalSet being evaluated has the impressions ``ids`` (original positions, in the EvalSet's own row order) -- ``arange(n)`` for a full split."""
    ids = np.asarray(ids, np.int64)
    ix = np.searchsorted(rag["ids"], ids)
    if len(ix) and (ix.max() >= len(rag["ids"]) or not np.array_equal(rag["ids"][ix], ids)):
        raise ValueError("the stored scores do not cover every impression of the EvalSet")
    ptr, flat = rag["ptr"], rag["flat"]

    def f(b):
        r = ix[b["rows"]]
        start, n = ptr[r], ptr[r + 1] - ptr[r]
        C = b["cand"].shape[1]
        col = np.arange(C)[None, :]
        keep = col < n[:, None]
        out = np.where(keep, flat[np.minimum(start[:, None] + col, max(len(flat) - 1, 0))], 0.0).astype(np.float32)
        return torch.from_numpy(out).to(device)
    return f


def ragged_rows(rag):
    return [rag["flat"][rag["ptr"][i]:rag["ptr"][i + 1]] for i in range(len(rag["ptr"]) - 1)]


def ragged_zscore(rag):
    """Standardise every impression's scores (mean 0, sd 1 over its candidates; the same formula as ``zscore``)."""
    ptr, flat = rag["ptr"], rag["flat"].astype(np.float64)
    lens = np.diff(ptr)
    seg = np.repeat(np.arange(len(lens)), lens)
    n = np.maximum(lens, 1)
    mu = np.bincount(seg, weights=flat, minlength=len(lens)) / n
    sd = np.sqrt(np.bincount(seg, weights=(flat - mu[seg]) ** 2, minlength=len(lens)) / n)
    return {"ids": rag["ids"], "ptr": ptr, "flat": ((flat - mu[seg]) / (sd[seg] + 1e-6)).astype(np.float32)}


def ragged_mean_z(rags):
    """Seed ensemble: the mean of the per-impression z-scored scores of several runs of the same model (all over the same impressions)."""
    if len(rags) == 1:
        return rags[0]
    for r in rags[1:]:
        if not (np.array_equal(r["ids"], rags[0]["ids"]) and np.array_equal(r["ptr"], rags[0]["ptr"])):
            raise ValueError("runs scored different impressions / candidates")
    zs = [ragged_zscore(r)["flat"] for r in rags]
    return {"ids": rags[0]["ids"], "ptr": rags[0]["ptr"], "flat": np.mean(zs, axis=0).astype(np.float32)}


def cold_weights(rows, seen, target=0.8):
    """Per-impression weights that re-balance the validation split towards the test split's mix of cold and warm clicks.

    Measured on MIND-large (v14 / v15): the validation day is warmer than the test day (82% of the test clicks are on articles no training row showed),
    so a selection made on the plain validation mean favours models that are strong on warm articles.  ``seen``: bool over article ids (observed by the
    training rows); the cold share of an impression is the share of its clicked articles with ``seen`` False; ``target``: the cold share of the clicks
    the weighted mean should reflect.  The weights are w = (target / c) * cold + ((1 - target) / (1 - c)) * (1 - cold), with c the cold share of all
    the clicks of ``rows`` -- so a weighted mean of a per-impression metric is the mean it would have if the split had ``target`` cold clicks (clicks of
    one kind are weighted alike).  Impressions without a click weigh 1 (their nDCG is 0 whatever the scores).  ``target`` outside (0, 1), or a split with
    only cold / only warm clicks, gives all-ones weights.  Returns ``(w [n], info)``; ``info`` = {"cold_frac": c, "target": target, "n": ...}."""
    seen = np.asarray(seen, bool)
    n = len(rows)
    cold_share = np.full(n, np.nan)
    n_pos = n_cold = 0
    for i, r in enumerate(rows):
        y = np.asarray(r[4])
        if y.size and (y > 0.5).any():
            pos = np.asarray(r[3], np.int64)[y > 0.5]
            c = ~seen[pos]
            cold_share[i] = float(c.mean())
            n_pos += len(pos)
            n_cold += int(c.sum())
    cfrac = n_cold / n_pos if n_pos else float("nan")
    info = {"cold_frac": cfrac, "target": target, "n": n}
    if not (target and 0.0 < target < 1.0) or not (0.0 < cfrac < 1.0):
        return np.ones(n), info
    w = (target / cfrac) * cold_share + ((1.0 - target) / (1.0 - cfrac)) * (1.0 - cold_share)
    w[np.isnan(cold_share)] = 1.0
    return w, info


def weighted_mean(values, weights=None):
    """Mean of ``values`` weighted by ``weights`` (plain mean when ``weights`` is None); 0 total weight gives nan."""
    v = np.asarray(values, np.float64)
    if weights is None:
        return float(np.mean(v)) if v.size else float("nan")
    w = np.asarray(weights, np.float64)
    tot = float(w.sum())
    return float((v * w).sum() / tot) if tot > 0 else float("nan")


# Raw per-candidate scores that stages keep on disk.  SUPER_RE: the zero-shot scorers the supermodel's linear fusion may pick from (unchanged since v14);
# AUX_RE: further scorers only the stacker (stacker.py) reads -- kept out of the fusion because, measured on MIND-large, the validation split rewards
# them (RAG / GraphRAG, plain mean-pool) more than the colder test split does.
SUPER_RE = re.compile(r"^(pool_lse|pool_km5|lora\+hn|txt_(t|cta|ctae|views)_lse|hist\d+_lse|ut_causal(@\d+)?)$")
AUX_RE = re.compile(r"^(frozen|pool_(recency|maxsim|topk|km\d+)|txt_(t|cta|ctae|views)|hist\d+_mean|(entity|knn)_rag|community_graphrag)$")
EXPORT_RE = re.compile(SUPER_RE.pattern[:-2] + "|" + AUX_RE.pattern[2:])


def export_wanted(name):
    """Rows whose raw scores are kept (the controls ``txt_ctr*`` are deliberately not among them)."""
    return bool(EXPORT_RE.match(name))


def default_kind(name):
    """``zs`` / ``tower`` components are what the supermodel fuses; ``aux`` ones are inputs of the stacker only."""
    if name.startswith("ut_"):
        return "tower"
    return "zs" if SUPER_RE.match(name) else "aux"


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.+@-]", "_", name)


def save_scores(dirpath, name, val, test, meta=None):
    """One ``.npz`` per component: ragged scores of the validation and test impressions it was evaluated on."""
    import json
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, _safe(name) + ".npz")
    arrays = {f"{sp}_{k}": rag[k] for sp, rag in (("val", val), ("test", test)) for k in ("ids", "ptr", "flat")}
    np.savez(path + ".tmp.npz", name=np.array(name), meta=np.array(json.dumps(meta or {}, default=str)), **arrays)
    os.replace(path + ".tmp.npz", path)
    return path


def load_scores(path):
    import json
    z = np.load(path, allow_pickle=False)
    out = {"name": str(z["name"]), "meta": json.loads(str(z["meta"]))}
    for sp in ("val", "test"):
        out[sp] = {k: z[f"{sp}_{k}"] for k in ("ids", "ptr", "flat")}
    return out


# ------------------------------------------------------------------ history-as-text queries
class HistQueries:
    """Unique last-K histories of validation/test rows, their query texts, and a scorer
    ``score = <query vector, candidate vector>`` (history-as-text user representation)."""

    def __init__(self, data, titles, K):
        self.titles, self.K = titles, K
        self.keys, self.rowq = {}, {}
        for sp, rows in (("val", data.validation), ("test", data.test)):
            rq = []
            for r in rows:
                h = tuple(r[2][-K:])
                rq.append(self.keys.setdefault(h, len(self.keys)) if h else -1)
            self.rowq[sp] = np.array(rq, np.int64)
        self.uniq = sorted(self.keys, key=self.keys.get)

    def needed(self, split, ids=None):
        """Indices of the unique queries used by the rows ``ids`` (default: every row) of a split."""
        rq = self.rowq[split] if ids is None else self.rowq[split][np.asarray(ids)]
        return np.unique(rq[rq >= 0])

    def texts(self, qidx=None):
        qidx = range(len(self.uniq)) if qidx is None else qidx
        return ["; ".join(self.titles[i] for i in self.uniq[q]) for q in qidx]

    def alloc(self, d):
        return np.zeros((max(len(self.uniq), 1), d), np.float32)

    def scorer(self, Q, E, split, device, ids=None):
        """``Q``: [n_unique, d] numpy; ``E``: candidate matrix tensor on ``device``;
        ``ids``: original row ids when scoring a subset EvalSet."""
        Qt = torch.from_numpy(np.concatenate([np.asarray(Q, np.float32), np.zeros((1, Q.shape[1]), np.float32)])).to(device)
        rq = self.rowq[split].copy()
        rq[rq < 0] = len(Q)                              # empty history -> zero query (ties like the mean-pool control)
        rq_t = torch.from_numpy(rq).to(device)
        ids_t = None if ids is None else torch.from_numpy(np.asarray(ids)).to(device)

        def f(b):
            r = torch.as_tensor(b["rows"], device=device)
            r = r if ids_t is None else ids_t[r]
            return torch.einsum("bd,bcd->bc", Qt[rq_t[r]], E[b["cand"]])
        return f


def atomic_write(path, data, mode="w"):
    tmp = path + ".tmp"
    with open(tmp, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
        f.write(data)
    os.replace(tmp, path)
