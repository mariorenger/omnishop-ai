"""A memory-lean twin of ``MindData.from_mind`` for MIND-large (2.2M training + 0.4M dev impressions).

``data.py`` is V10's loader, vendored word for word, and it keeps what a Kaggle machine cannot hold at that size: one Python tuple of
float lists per impression for the popularity feature, a datetime and a date string per impression, and a ``(user, article)`` tuple for every
history item (``_edges``, which only the graph models of V10 read; ~58M tuples on MIND-large).  This version computes **exactly the same split,
the same online popularity and the same negative samples** (``selftest.py`` compares the two loaders field by field on the fixture), but

  * keeps ``pop`` as one float64 numpy array per impression,
  * keeps the day as an ordinal and the timestamp as an integer key (parsed once per distinct string),
  * does not build ``_edges`` (left empty),
  * returns the sampled negatives as plain ints.

Use it through ``suite.py --lean-data 1``; MIND-small stays on the verbatim loader.

It also builds what the learned ranker (``stacker.py``) needs and ``data.py`` does not have:

  * ``data.aux_validation`` / ``data.aux_test``: one float32 array ``[n_candidates, 5]`` per impression -- hours since the article first appeared in
    any earlier impression's candidate list (0 if it never did), ``log1p`` of the number of earlier impressions that showed it, and ``log1p`` of
    the number of impressions that showed it in the last 1 / 6 / 24 completed clock hours (``exposure_features_lean``).  All **label-free and
    causal** (only earlier impressions, ties in time scored before any of them is counted, as for the popularity feature);
  * ``data.item_stats``: per article, how often the training rows (``train_core``) showed it and how often it was clicked there (``train_pos`` /
    ``train_shown``: what is known before the validation days) and the same counts over ``train_core`` + the validation days (``trainval_pos`` /
    ``trainval_shown``: what is known before the test day -- the stacker reads the first for validation impressions and the second for test ones, so both see
    the labelled history up to the day before).
"""
from __future__ import annotations

import os
import time
from collections import Counter
from datetime import datetime

import numpy as np

from data import PAD, MindData

_FORMATS = ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S")


def online_log_ctr_lean(cands, labs, tkeys, n_news):
    """Smoothed log-CTR of every candidate from impressions strictly earlier in time (ties scored before any of them is counted):
    the recipe of ``data._online_log_ctr`` with counters in numpy arrays.  Returns one float64 array per impression."""
    clicks, shown = np.zeros(n_news, np.float64), np.zeros(n_news, np.float64)
    tk = np.asarray(tkeys, np.int64)
    order = np.argsort(tk, kind="stable")
    ts = tk[order]
    cut = np.flatnonzero(np.diff(ts)) + 1
    starts, ends = np.concatenate([[0], cut]), np.concatenate([cut, [len(order)]])
    out = [None] * len(cands)
    for s, e in zip(starts, ends):
        grp = order[s:e]
        for k in grp:
            c = np.asarray(cands[k], np.int64)
            out[k] = np.log((clicks[c] + 1.0) / (shown[c] + 20.0))
        for k in grp:
            c = np.asarray(cands[k], np.int64)
            np.add.at(shown, c, 1.0)
            np.add.at(clicks, c, np.asarray(labs[k], np.float64))
    return out


EXPO_WINDOWS = (1, 6, 24)                                                  # hours (completed hour buckets before the impression's own)


def exposure_features_lean(cands, tkeys, n_news, want):
    """Label-free, causal exposure features of the impressions flagged in ``want`` (bool over impressions).  For every candidate:

      0  hours since the article first appeared in any EARLIER impression (0 if never);
      1  ``log1p`` of the number of earlier impressions that showed it;
      2-4  ``log1p`` of the number of impressions that showed it in the last 1 / 6 / 24 COMPLETED clock hours before the impression's own hour
           (``EXPO_WINDOWS``; the hour of the impression itself is never counted, so the lag is between 0 and 1 hour).

    Impressions with the same time stamp are scored before any of them is counted (columns 0-1).  Returns a list (``None`` for impressions not
    wanted) of float32 arrays ``[len(candidates), 5]``."""
    tk = np.asarray(tkeys, np.int64)
    hour = (tk - tk.min()) // 3600
    nb = int(hour.max()) + 1
    per_hour = np.zeros((nb + 1, n_news), np.int32)                           # cum[b] = shows in the hours before hour b
    by_hour = [[] for _ in range(nb)]
    for k, h in enumerate(hour):
        by_hour[h].append(k)
    for h in range(nb):
        if by_hour[h]:
            per_hour[h + 1] = np.bincount(np.concatenate([np.asarray(cands[k], np.int64) for k in by_hour[h]]), minlength=n_news)
    cum = np.cumsum(per_hour, axis=0, dtype=np.int32)
    shown = np.zeros(n_news, np.float64)
    first = np.full(n_news, -1, np.int64)
    order = np.argsort(tk, kind="stable")
    ts = tk[order]
    cut = np.flatnonzero(np.diff(ts)) + 1
    starts, ends = np.concatenate([[0], cut]), np.concatenate([cut, [len(order)]])
    out = [None] * len(cands)
    for s_, e_ in zip(starts, ends):
        grp = order[s_:e_]
        now = int(ts[s_])
        for k in grp:
            if want[k]:
                c = np.asarray(cands[k], np.int64)
                f = first[c]
                age = np.where(f >= 0, (now - f) / 3600.0, 0.0)
                b = int(hour[k])
                win = [np.log1p((cum[b] - cum[max(b - w, 0)])[c]) for w in EXPO_WINDOWS]
                out[k] = np.stack([age, np.log1p(shown[c])] + win, 1).astype(np.float32)
        for k in grp:
            c = np.asarray(cands[k], np.int64)
            np.add.at(shown, c, 1.0)
            new = c[first[c] < 0]
            first[new] = now
    return out


def item_stats_lean(cands, labs, idx, n_news, chunk=200_000):
    """Per-article counts over the impressions ``idx``: how often it was shown and how often it was clicked."""
    shown, pos = np.zeros(n_news, np.float64), np.zeros(n_news, np.float64)
    idx = np.asarray(idx, np.int64)
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        c = np.concatenate([np.asarray(cands[i], np.int64) for i in part]) if len(part) else np.zeros(0, np.int64)
        y = np.concatenate([np.asarray(labs[i], np.float64) for i in part]) if len(part) else np.zeros(0)
        shown += np.bincount(c, minlength=n_news)
        pos += np.bincount(c, weights=y, minlength=n_news)
    return pos.astype(np.float32), shown.astype(np.float32)


def from_mind_lean(train_dir, dev_dir, title_len=20, max_hist=50, n_neg=4, min_word_freq=3, llm_dim=64, llm_embeddings=None, dev_ratio=0.1, seed=0, log=print, aux=True):
    rng = np.random.default_rng(seed)

    def read_news(path):
        out = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 5:
                    out[p[0]] = (p[1], p[3], p[4])
        return out

    news_rows = read_news(os.path.join(train_dir, "news.tsv"))
    if dev_dir and os.path.exists(os.path.join(dev_dir, "news.tsv")):
        news_rows.update(read_news(os.path.join(dev_dir, "news.tsv")))

    wf = Counter()
    for _, title, abstract in news_rows.values():
        wf.update((title + " " + abstract).lower().split())
    vocab = {"<pad>": PAD}
    for word, count in wf.items():
        if count >= min_word_freq:
            vocab[word] = len(vocab)
    cats = sorted({cat for cat, _, _ in news_rows.values()})
    cat_map = {cat: i + 1 for i, cat in enumerate(cats)}
    nid_map = {"<pad>": PAD}
    for nid in news_rows:
        nid_map[nid] = len(nid_map)
    n_news = len(nid_map)
    news_title = np.zeros((n_news, title_len), np.int64)
    news_cat = np.zeros(n_news, np.int64)
    for nid, (cat, title, abstract) in news_rows.items():
        words = (title + " " + abstract).lower().split()[:title_len]
        gi = nid_map[nid]
        news_title[gi, :len(words)] = [vocab.get(w, PAD) for w in words]
        news_cat[gi] = cat_map[cat]

    if llm_embeddings is None:
        llm_emb = np.zeros((n_news, llm_dim), np.float32)
    else:
        if not llm_embeddings:
            raise ValueError("empty BGE embedding map")
        dims = {np.asarray(v).reshape(-1).size for v in llm_embeddings.values()}
        if len(dims) != 1:
            raise ValueError("inconsistent embedding dimensions")
        dim = dims.pop()
        llm_emb = np.zeros((n_news, dim), np.float32)
        for nid, gi in nid_map.items():
            if nid != "<pad>" and nid in llm_embeddings:
                llm_emb[gi] = np.asarray(llm_embeddings[nid], np.float32).reshape(-1)
        coverage = float((np.linalg.norm(llm_emb[1:], axis=1) > 0).mean())
        if coverage < 0.999 or not np.isfinite(llm_emb).all():
            raise ValueError(f"invalid BGE coverage={coverage:.4%}")
        print(f"  embedding coverage: {coverage:.3%} (PAD excluded)")
    llm_emb = (llm_emb / (np.linalg.norm(llm_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    uid_map = {}
    tcache = {}

    def time_of(value):
        """-> (day ordinal, integer timestamp key); parsed once per distinct string."""
        hit = tcache.get(value)
        if hit is not None:
            return hit
        v = value.strip()
        for fmt in _FORMATS:
            try:
                t = datetime.strptime(v, fmt)
                break
            except ValueError:
                t = None
        if t is None:
            raise ValueError(f"unparseable impression time: {value!r}")
        tcache[value] = (t.toordinal(), t.toordinal() * 86400 + t.hour * 3600 + t.minute * 60 + t.second)
        return tcache[value]

    def read_behaviors(path):
        ids, users, hists, cands, labs, days, tkeys = [], [], [], [], [], [], []
        t0 = time.time()
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                impr_id, user_raw, timestamp, history_raw, impressions = p[:5]
                hists.append([nid_map[x] for x in history_raw.split() if x in nid_map] if history_raw else [])
                c, y = [], []
                for item in impressions.split():
                    nid, lab = item.rsplit("-", 1)
                    if nid in nid_map:
                        c.append(nid_map[nid])
                        y.append(int(lab))
                cands.append(c)
                labs.append(y)
                if user_raw not in uid_map:
                    uid_map[user_raw] = len(uid_map)
                users.append(uid_map[user_raw])
                ids.append(str(impr_id))
                day, key = time_of(timestamp)
                days.append(day)
                tkeys.append(key)
                if n % 500_000 == 0:
                    log(f"    {os.path.basename(os.path.dirname(path))}/behaviors.tsv: {n:,} lines ({time.time() - t0:.0f}s)")
        return ids, users, hists, cands, labs, days, tkeys

    tr = read_behaviors(os.path.join(train_dir, "behaviors.tsv"))
    if not dev_dir or os.path.abspath(dev_dir) == os.path.abspath(train_dir):
        raise ValueError("V10 requires separate MINDsmall_train and MINDsmall_dev")
    te = read_behaviors(os.path.join(dev_dir, "behaviors.tsv"))
    n_tr, n_te = len(tr[0]), len(te[0])
    if not n_tr or not n_te:
        raise ValueError("MIND behaviors files are empty")
    t0 = time.time()
    pop = online_log_ctr_lean(tr[3] + te[3], tr[4] + te[4], tr[6] + te[6], n_news)
    log(f"    online popularity of {n_tr + n_te:,} impressions ({time.time() - t0:.0f}s)")

    days = sorted(set(tr[5]))
    target = max(1, int(np.ceil(n_tr * dev_ratio)))
    per_day = Counter(tr[5])
    val_days, count = [], 0
    for day in reversed(days):
        val_days.append(day)
        count += per_day[day]
        if count >= target:
            break
    vset = set(val_days)
    train_days = [d for d in days if d not in vset]
    if not train_days:
        raise ValueError("day split leaves no train_core day")
    core_idx = [i for i in range(n_tr) if tr[5][i] not in vset]
    val_idx = [i for i in range(n_tr) if tr[5][i] in vset]
    if max(tr[5][i] for i in core_idx) >= min(tr[5][i] for i in val_idx):
        raise AssertionError("chronological train/validation ordering violated")
    iso = lambda o: datetime.fromordinal(o).date().isoformat()
    validation = [(tr[0][i], tr[1][i], tr[2][i], tr[3][i], tr[4][i], pop[i]) for i in val_idx]
    test = [(te[0][i], te[1][i], te[2][i], te[3][i], te[4][i], pop[n_tr + i]) for i in range(n_te)]
    test_days = sorted({iso(d) for d in te[5]})
    for i in core_idx:                                                  # the training rows' popularity is never read again: free it before the samples are built
        pop[i] = None
    aux_val = aux_test = item_stats = None
    if aux:                                                             # label-free exposure features of the validation / test slates + training-row item counts
        t0 = time.time()
        want = np.zeros(n_tr + n_te, bool)
        want[val_idx] = True
        want[n_tr:] = True
        feats = exposure_features_lean(tr[3] + te[3], tr[6] + te[6], n_news, want)
        aux_val, aux_test = [feats[i] for i in val_idx], feats[n_tr:]
        del feats
        pos_cnt, shown_cnt = item_stats_lean(tr[3], tr[4], core_idx, n_news)
        pos_tv, shown_tv = item_stats_lean(tr[3], tr[4], core_idx + val_idx, n_news)             # everything labelled before the test day: the same "yesterday included" as for validation
        item_stats = {"train_pos": pos_cnt, "train_shown": shown_cnt, "trainval_pos": pos_tv, "trainval_shown": shown_tv}
        log(f"    exposure features and item counts ({time.time() - t0:.0f}s)")

    train_core = []
    for i in core_idx:
        user, hist, cands, labs = tr[1][i], tr[2][i], tr[3][i], tr[4][i]
        pos = [c for c, lab in zip(cands, labs) if lab == 1]
        neg = [c for c, lab in zip(cands, labs) if lab == 0]
        if not pos or not neg:
            continue
        neg_arr = np.asarray(neg, np.int64)
        for p_ in pos:
            train_core.append((user, hist, p_, rng.choice(neg_arr, size=n_neg, replace=len(neg) < n_neg).tolist()))
    val_days_iso, train_days_iso = sorted(iso(d) for d in val_days), [iso(d) for d in train_days]
    split_info = {
        "train_days": train_days_iso, "validation_days": val_days_iso, "test_days": test_days,
        "train_day_min": min(train_days_iso), "train_day_max": max(train_days_iso),
        "validation_day_min": min(val_days_iso), "validation_day_max": max(val_days_iso),
        "target_validation_ratio": float(dev_ratio), "actual_validation_ratio": len(validation) / n_tr,
        "raw_train_impressions": n_tr, "train_core_impressions": len(core_idx), "validation_impressions": len(validation), "test_impressions": len(test),
    }
    print(f"  chronological split: {len(core_idx):,} train_core / {len(validation):,} validation / {len(test):,} official test impressions")
    data = MindData(news_title, news_cat, n_news, len(uid_map), len(vocab), len(cat_map) + 1, title_len, max_hist, llm_emb, train_core, validation, test,
                    len(core_idx), len(validation), len(test), train_days_iso, val_days_iso, test_days, split_info, [])
    if aux:
        data.aux_validation, data.aux_test, data.item_stats = aux_val, aux_test, item_stats
    return data
