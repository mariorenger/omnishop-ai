"""Reference-recipe data for the NRMS / NAML / Fastformer baselines on the *exact* V10 split.

Why this module exists.  V10's trained baselines are small and cheap on purpose: 64-d random word vectors, the first 20
whitespace tokens of title+abstract with rare words mapped to PAD, 4 negatives drawn once for all epochs, lr 1e-3.  They score
~0.04 below published MIND-small numbers on every metric, so "beats V10's Fastformer" says little.  Here the same model families
are trained with the usual published recipe (title-only regex tokens, GloVe-300 initialisation, negatives re-drawn every epoch,
lr 1e-4) on the same train_core / validation / MINDsmall_dev split and scored by the same evaluator.

Nothing here decides *which* impressions are used on its own: the training impressions are re-derived exactly as ``data.py`` does it
and asserted against the ``MindData`` object (same day split, same number of impressions, same number of training samples).
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import zipfile
from collections import Counter
from datetime import datetime

import numpy as np
import torch

PAD, UNK = 0, 1
_WORD = re.compile(r"[\w]+|[.,!?;|]")


# ------------------------------------------------------------------------------------------- text
def tokenize(s):
    """Microsoft Recommenders' ``word_tokenize``: lower-cased words, punctuation marks as tokens of their own."""
    return _WORD.findall(s.lower()) if isinstance(s, str) else []


def build_vocab(token_lists, min_freq=2, keep=None):
    """0 = PAD, 1 = UNK, then every word seen at least ``min_freq`` times or listed in ``keep`` (words with a pre-trained vector)."""
    cnt = Counter()
    for t in token_lists:
        cnt.update(t)
    keep = keep or ()
    words = sorted((w for w, c in cnt.items() if c >= min_freq or w in keep), key=lambda w: (-cnt[w], w))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode_tokens(token_lists, vocab, length):
    """[n_articles + 1, length] int64; row 0 is the PAD article, tokens are left-aligned."""
    out = np.zeros((len(token_lists) + 1, length), np.int64)
    for i, toks in enumerate(token_lists, 1):
        ids = [vocab.get(w, UNK) for w in toks[:length]]
        out[i, :len(ids)] = ids
    return out


def label_ids(values):
    """Category-like strings -> ids (0 = PAD article), and the table size."""
    m = {v: i for i, v in enumerate(sorted(set(values)), 1)}
    return np.array([0] + [m[v] for v in values], np.int64), len(m) + 1


# ------------------------------------------------------------------------------------------- GloVe
def _from_txt(fh, dim, want):
    words, vecs = [], []
    for line in fh:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "ignore")
        w, _, rest = line.partition(" ")
        if w not in want:
            continue
        v = rest.split()
        if len(v) == dim:                                           # multi-token entries of the 840B file are skipped
            words.append(w)
            vecs.append(np.asarray(v, np.float32))
    if not words:
        raise ValueError("no overlap with the corpus vocabulary")
    return words, np.stack(vecs)


def _glove_local(path, dim, want):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(f".{dim}d.txt"))
            with io.TextIOWrapper(z.open(name), encoding="utf-8") as fh:
                return _from_txt(fh, dim, want)
    with open(path, encoding="utf-8") as fh:
        return _from_txt(fh, dim, want)


def _from_st_dir(root, dim, want):
    """sentence-transformers ``WordEmbeddings`` folder: a JSON holding the vocabulary list and a torch file with the [V, dim] matrix."""
    folder = os.path.join(root, "0_WordEmbeddings")
    folder = folder if os.path.isdir(folder) else root
    vocab = None
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".json"):
            with open(os.path.join(folder, fn), encoding="utf-8") as f:
                j = json.load(f)
            if isinstance(j, dict) and isinstance(j.get("vocab"), list):
                vocab = j["vocab"]
                break
    mat = None
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        if fn.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(p)
        elif fn.endswith((".bin", ".pt")):
            sd = torch.load(p, map_location="cpu")
        else:
            continue
        cands = [v for v in sd.values() if getattr(v, "ndim", 0) == 2 and v.shape[1] == dim]
        if cands:
            mat = max(cands, key=lambda v: v.shape[0]).float().numpy()
            break
    if vocab is None or mat is None or len(vocab) != mat.shape[0]:
        raise ValueError(f"unrecognised word-embedding folder {folder} (vocab={None if vocab is None else len(vocab)}, "
                         f"matrix={None if mat is None else mat.shape})")
    idx = {w: i for i, w in enumerate(vocab)}
    words = [w for w in want if w in idx]
    if not words:
        raise ValueError("no overlap with the corpus vocabulary")
    return words, mat[[idx[w] for w in words]].copy()


def _glove_hf(dim, want):
    from huggingface_hub import snapshot_download
    root = snapshot_download(f"sentence-transformers/average_word_embeddings_glove.6B.{dim}d", allow_patterns=["0_WordEmbeddings/*"])
    return _from_st_dir(root, dim, want)


def _glove_gensim(dim, want):
    import gensim.downloader as api
    kv = api.load(f"glove-wiki-gigaword-{dim}")
    words = [w for w in want if w in kv.key_to_index]
    if not words:
        raise ValueError("no overlap with the corpus vocabulary")
    return words, np.stack([np.asarray(kv[w], np.float32) for w in words])


def find_local_glove(dim):
    out = []
    for pat in (f"/kaggle/input/**/glove*{dim}d*.txt", f"/kaggle/input/**/glove*{dim}d*.zip", "/kaggle/input/**/glove*6B*.zip"):
        out += sorted(glob.glob(pat, recursive=True))
    return list(dict.fromkeys(out))


def load_glove(want, dim=300, source=None, log=print):
    """GloVe vectors restricted to ``want`` (the corpus words) -> ``(words, matrix, label)``; ``(None, None, '')`` if every source fails,
    in which case the models start from random vectors and say so.  Order: ``source`` / ``$V11_GLOVE`` (txt, gz-less zip), a GloVe file
    attached to the notebook, the Hugging Face copy of GloVe-6B, gensim's downloader."""
    attempts = []
    src = source or os.environ.get("V11_GLOVE")
    if src:
        attempts.append((f"file:{src}", lambda: _glove_local(src, dim, want)))
    for p in find_local_glove(dim):
        attempts.append((f"file:{p}", lambda p=p: _glove_local(p, dim, want)))
    attempts.append(("huggingface:sentence-transformers/average_word_embeddings_glove.6B", lambda: _glove_hf(dim, want)))
    attempts.append(("gensim:glove-wiki-gigaword", lambda: _glove_gensim(dim, want)))
    for label, fn in attempts:
        try:
            words, mat = fn()
            if mat.ndim != 2 or mat.shape[1] != dim or not np.isfinite(mat).all():
                raise ValueError("malformed matrix")
            log(f"  GloVe-{dim}d <- {label}: {len(words):,} of {len(want):,} corpus words have a vector")
            return words, mat.astype(np.float32), label
        except Exception as e:                                       # a missing source is normal; try the next one
            log(f"  GloVe source {label} unavailable: {repr(e)[:110]}")
    log("  !! no GloVe source worked -> word vectors start from random (the rows are labelled '_noglove')")
    return None, None, ""


class RefText:
    """Token tables of the reference recipe (row 0 = PAD article): title (NRMS / Fastformer / NAML), abstract (NAML), category ids."""

    def __init__(self, meta, title_len=30, body_len=50, min_freq=2, glove_dim=300, use_glove=True, glove_source=None, log=print):
        titles = [tokenize(t) for t in meta["titles"]]
        bodies = [tokenize(a) for a in meta["abstracts"]]
        want = set()
        for t in titles + bodies:
            want.update(t)
        self.glove_words = self.glove_mat = None
        self.glove_label = ""
        if use_glove:
            self.glove_words, self.glove_mat, self.glove_label = load_glove(want, glove_dim, glove_source, log)
        self.glove_dim = glove_dim
        self.vocab = build_vocab(titles + bodies, min_freq, keep=set(self.glove_words) if self.glove_words else None)
        self.title = encode_tokens(titles, self.vocab, title_len)
        self.body = encode_tokens(bodies, self.vocab, body_len)
        self.cat, self.n_cat = label_ids(meta["cats"])
        self.sub, self.n_sub = label_ids(meta.get("subcats") or ["-"] * len(meta["cats"]))
        gset = set(self.glove_words) if self.glove_words else set()
        real = self.title[1:][self.title[1:] != PAD]
        self.stats = {"vocab": len(self.vocab), "glove": self.glove_label or "none", "glove_vocab_hits": sum(1 for w in self.vocab if w in gset),
                      "title_tokens": int(real.size), "unk_title_share": float((real == UNK).mean()) if real.size else 0.0}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def embedding(self, dim, seed=0, init="glove"):
        """Initial word-embedding matrix [V, dim]: GloVe rows where available, otherwise N(0, s) with s matched to GloVe (0.1 without it);
        the PAD row is zero.  ``init='random'`` ignores GloVe."""
        rng = np.random.default_rng(seed)
        use = init == "glove" and self.glove_mat is not None and dim == self.glove_mat.shape[1]
        std = float(self.glove_mat.std()) if use else 0.1
        M = rng.normal(0.0, std, (len(self.vocab), dim)).astype(np.float32)
        M[PAD] = 0.0
        if use:
            gi = {w: i for i, w in enumerate(self.glove_words)}
            for w, i in self.vocab.items():
                j = gi.get(w)
                if j is not None:
                    M[i] = self.glove_mat[j]
        return M

    def tokens(self, device):
        t = lambda a: torch.from_numpy(a).to(device)
        return {"title": t(self.title), "body": t(self.body), "cat": t(self.cat), "sub": t(self.sub)}


# ------------------------------------------------------------------------------------------- behaviours / split
_TIME_FORMATS = ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S")


def _time_of(value):
    value = value.strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"unparseable impression time: {value!r}")


def read_behaviors(path, nid_map):
    """Same filtering as ``MindData.from_mind``: ``(user, history, candidates, labels, day)`` per impression, global article ids."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            _impr, user_raw, timestamp, history_raw, impressions = p[:5]
            hist = [nid_map[n] for n in history_raw.split() if n in nid_map] if history_raw else []
            cands, labs = [], []
            for item in impressions.split():
                nid, lab = item.rsplit("-", 1)
                if nid in nid_map:
                    cands.append(nid_map[nid])
                    labs.append(int(lab))
            out.append((user_raw, hist, cands, labs, _time_of(timestamp).date().isoformat()))
    return out


def split_train_core(rows, dev_ratio=0.1):
    """The chronological split of ``MindData.from_mind``: the last days holding at least ``dev_ratio`` of the impressions are validation."""
    days = sorted({r[4] for r in rows})
    target = max(1, int(np.ceil(len(rows) * dev_ratio)))
    val_days, count = [], 0
    for day in reversed(days):
        val_days.append(day)
        count += sum(r[4] == day for r in rows)
        if count >= target:
            break
    vd = set(val_days)
    return [r for r in rows if r[4] not in vd], [r for r in rows if r[4] in vd], sorted(val_days)


def read_train_core(train_dir, ids, data, dev_ratio=0.1):
    """Training impressions (with every candidate and label) re-derived from ``behaviors.tsv`` and checked against ``MindData``."""
    nid_map = {nid: i + 1 for i, nid in enumerate(ids)}
    core, val, val_days = split_train_core(read_behaviors(os.path.join(train_dir, "behaviors.tsv"), nid_map), dev_ratio)
    if len(core) != data.train_core_impressions or len(val) != data.validation_impressions or val_days != list(data.validation_days):
        raise ValueError(f"re-derived split differs from MindData: {len(core)}/{len(val)} impressions, validation days {val_days} vs "
                         f"{data.train_core_impressions}/{data.validation_impressions}, {list(data.validation_days)}")
    return core


def check_train_alignment(ts, data):
    """The re-derived samples must be V10's: same clicked article and history in the same order, every negative V10 drew lies in the
    impression's pool (catches an article-id mapping that differs from ``MindData`` while the counts still agree)."""
    if ts.n_samples != len(data.train_core):
        raise ValueError(f"{ts.n_samples} re-derived samples vs {len(data.train_core)} in MindData.train_core")
    H = ts.hist.shape[1]
    for k, (_u, hist, pos, negs) in enumerate(data.train_core):
        i = int(ts.samp_imp[k])
        h = list(hist)[-H:]
        pool = ts.neg_flat[ts.neg_off[i]:ts.neg_off[i] + ts.neg_cnt[i]]
        if (pos != ts.samp_pos[k] or not np.array_equal(ts.hist[i, :len(h)], h) or ts.hist[i, len(h):].any()
                or not set(int(x) for x in negs) <= set(pool.tolist())):
            raise ValueError(f"re-derived training sample {k} differs from MindData.train_core[{k}] (article-id mapping mismatch?)")


def subsample(items, frac, seed=0):
    """Random subset (original order kept); ``frac >= 1`` keeps everything."""
    if frac is None or frac >= 1:
        return items
    n = max(1, int(round(len(items) * frac)))
    return [items[i] for i in np.sort(np.random.default_rng(seed).permutation(len(items))[:n])]


# ------------------------------------------------------------------------------------------- training samples
class _Samples:
    def _to(self, device):
        if getattr(self, "_dev", None) != device:
            self._hist_t = torch.from_numpy(self.hist).to(device)
            self._dev = device
        return self._hist_t


class TrainSet(_Samples):
    """V10's training samples (one per clicked article of an impression that also has a non-clicked one) but with the *whole* pool of
    impression negatives kept, so ``n_neg`` of them are re-drawn every epoch (V10 draws them once).  ``replace`` follows V10:
    sampling without replacement unless the pool is smaller than ``n_neg``."""

    dynamic = True

    def __init__(self, imps, max_hist=50, n_neg=4):
        keep = [r for r in imps if any(l == 1 for l in r[3]) and any(l == 0 for l in r[3])]
        self.n_neg, self.n_imp = n_neg, len(keep)
        self.hist = np.zeros((len(keep), max_hist), np.int32)
        neg_flat, samp_imp, samp_pos = [], [], []
        self.neg_off = np.zeros(len(keep), np.int64)
        self.neg_cnt = np.zeros(len(keep), np.int32)
        for i, (_u, h, c, y, _d) in enumerate(keep):
            h = list(h)[-max_hist:]
            self.hist[i, :len(h)] = h
            neg = [x for x, l in zip(c, y) if l == 0]
            self.neg_off[i], self.neg_cnt[i] = len(neg_flat), len(neg)
            neg_flat.extend(neg)
            for x, l in zip(c, y):
                if l == 1:
                    samp_imp.append(i)
                    samp_pos.append(x)
        self.neg_flat = np.asarray(neg_flat, np.int32)
        self.samp_imp = np.asarray(samp_imp, np.int64)
        self.samp_pos = np.asarray(samp_pos, np.int32)
        self.n_samples = len(self.samp_imp)

    def draw(self, rng):
        """[n_samples, n_neg] negatives for one epoch."""
        K = self.n_neg
        out = np.empty((self.n_samples, K), np.int32)
        for s in range(self.n_samples):
            i = self.samp_imp[s]
            o, c = self.neg_off[i], self.neg_cnt[i]
            out[s] = rng.choice(self.neg_flat[o:o + c], size=K, replace=bool(c < K))
        return out

    def batches(self, bs, epoch, seed, device):
        rng = np.random.default_rng([seed, epoch])
        negs = self.draw(rng)
        perm = rng.permutation(self.n_samples)
        hist_t = self._to(device)
        for s in range(0, self.n_samples, bs):
            ix = perm[s:s + bs]
            H = hist_t[torch.from_numpy(self.samp_imp[ix]).to(device)].long()
            C = torch.from_numpy(np.concatenate([self.samp_pos[ix, None], negs[ix]], 1).astype(np.int64)).to(device)
            yield H, H > 0, C


class StaticTrainSet(_Samples):
    """V10's own ``train_core`` rows ``(user, hist, pos, negs)``: the negatives drawn at build time stay fixed for every epoch."""

    dynamic = False

    def __init__(self, rows, max_hist=50):
        self.n_samples = self.n_imp = len(rows)
        self.hist = np.zeros((len(rows), max_hist), np.int32)
        self.cands = np.zeros((len(rows), 1 + len(rows[0][3])), np.int64)
        for i, (_u, h, pos, negs) in enumerate(rows):
            h = list(h)[-max_hist:]
            self.hist[i, :len(h)] = h
            self.cands[i] = [pos] + list(negs)

    def batches(self, bs, epoch, seed, device):
        rng = np.random.default_rng([seed, epoch])
        perm = rng.permutation(self.n_samples)
        hist_t = self._to(device)
        for s in range(0, self.n_samples, bs):
            ix = perm[s:s + bs]
            H = hist_t[torch.from_numpy(ix).to(device)].long()
            yield H, H > 0, torch.from_numpy(self.cands[ix]).to(device)
