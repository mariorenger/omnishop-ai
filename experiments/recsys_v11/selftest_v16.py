"""Tests (39)-(49) of the v16 additions: the cold-weighted validation mean, the windowed click-through and the impression times of the loader, the entity-overlap
features, the stacker's folds / forward holdout / new-article weights / coarsening, a planted case that shows the machinery exposes same-day memorisation, the
supermodel's weighted selection, the stack-only restart, the release of a finished run and the composite encoder.  Called by ``selftest.main()``.

    (39)  common.cold_weights: re-balances the cold / warm clicks to the target; degenerate inputs give ones; weighted_mean
    (40)  data_lean: windowed click-through == a plain-python recomputation from the raw files; hand-made case; strictly causal (labels of the impression's own hour and
          later never reach it); impression times; cap_data cuts the new per-impression arrays with the same indices as the rows
    (41)  featscore: entity tables (IDF, rarest kept) and entity-overlap features == the plain-python definition; padding is 0
    (42)  stacker helpers: time folds / buffer / early-stopping part, forward split, new-article weights, coarsening (idempotent, z-columns follow), column swap round trip,
          configuration columns; the builder's new columns equal an independent recomputation
    (43)  planted same-day memorisation: user folds are optimistic about a model that memorises article-level clicks, time folds + the new-article slice are not
    (44)  supermodel: weighted nDCG@10 == the manual weighted mean; the cold-weighted objective picks the component that is better on cold clicks
    (45)  restart: changed stack settings (or --restack) re-run only the stack stage; other changed settings start fresh
    (46)  Suite.release(): the data is freed, the records / per-impression results survive and the cross-run tables still work
    (47)  composite encoder 'concat:A+B': unit vectors, dot product = mean of the parts' dot products, served from the cache, errors
    (48)  Memory.knn_view: equal to the one-shot definition and bounded by a block (the one-shot gather killed the MIND-large run on 2048-d vectors)
    (49)  common.PEAK / lifetime_peak_gb / mem_note: a transient allocation is seen in the stage peak; the container limit is read
"""
from __future__ import annotations

import gc
import os
import shutil
import time

import numpy as np
import torch

import common
import data_lean
import featscore as FS
import metrics
import stacker as ST
import suite
import supermodel as SM
from selftest_v14 import ok, quiet, raises, zrow
from selftest_v15 import TS_FORMAT, _read_behaviors, _splits  # noqa: F401


# ------------------------------------------------------------------------------------------------ (39)
def test_cold_weights():
    seen = np.array([False, True, True, False, False, True])
    mk = lambda i, cands, labs: (i, 0, [], cands, labs, None)
    rows = [mk(0, [1, 3], [1, 0]), mk(1, [3, 4], [1, 0]), mk(2, [1, 2, 3], [1, 0, 1]), mk(3, [4, 5], [0, 0])]          # clicks: 1 warm | 3 cold | 1 warm + 3 cold | none
    w, info = common.cold_weights(rows, seen, 0.8)
    assert info["cold_frac"] == 0.5 and info["target"] == 0.8 and info["n"] == 4
    assert np.allclose(w, [0.4, 1.6, 1.0, 1.0]), w                                              # (0.8 / 0.5) * cold share + (0.2 / 0.5) * warm share; no click -> 1
    # single-click impressions: the weighted cold share of the clicks IS the target
    rng = np.random.default_rng(0)
    cold_art, warm_art = 3, 1
    pick = rng.random(500) < 0.3
    single = [mk(i, [cold_art if c else warm_art, 4], [1, 0]) for i, c in enumerate(pick)]
    w, info = common.cold_weights(single, seen, 0.9)
    cold_clicks = np.array([r[3][0] == cold_art for r in single])
    assert abs(info["cold_frac"] - pick.mean()) < 1e-12 and abs((w * cold_clicks).sum() / w.sum() - 0.9) < 1e-9
    m = rng.random(500)
    assert abs(common.weighted_mean(m, w) - (w * m).sum() / w.sum()) < 1e-12 and common.weighted_mean(m) == float(m.mean()) and np.isnan(common.weighted_mean([1.0], [0.0]))
    # degenerate inputs: no re-weighting
    for tgt, rr in ((0.0, single), (1.0, single), (None, single), (0.8, [mk(0, [3, 4], [1, 0])]), (0.8, [mk(0, [1, 2], [1, 0])]), (0.8, [mk(0, [1, 2], [0, 0])])):
        ww, _ = common.cold_weights(rr, seen, tgt)
        assert (ww == 1).all() and len(ww) == len(rr), (tgt, ww)
    ok("(39) cold_weights: weighted cold share of the clicks == the target (single-click impressions), no click -> weight 1, degenerate inputs -> ones; weighted_mean")


# ------------------------------------------------------------------------------------------------ (40)
def test_popwin_and_times(S, mind_train, mind_dev):
    L = np.log
    cands = [[1, 2], [1, 3], [1, 2, 4], [2], [1, 3]]
    labs = [[1, 0], [0, 1], [1, 1, 0], [0], [1, 0]]
    tkeys = [0, 100, 2 * 3600 + 5, 3 * 3600, 3 * 3600]                                       # hours 0, 0, 2, 3, 3
    out = data_lean.popwin_features_lean(cands, labs, tkeys, 5, np.ones(5, bool))

    def row(c1, s1, c6, s6, c24, s24):
        return [L((c1 + 1) / (s1 + 20)), L((c6 + 1) / (s6 + 20)), L((c24 + 1) / (s24 + 20)), np.log1p(c1), np.log1p(c6), np.log1p(c24)]
    want = [
        [row(0, 0, 0, 0, 0, 0)] * 2,                                                        # nothing is complete before hour 0
        [row(0, 0, 0, 0, 0, 0)] * 2,
        [row(0, 0, 1, 2, 1, 2), row(0, 0, 0, 1, 0, 1), row(0, 0, 0, 0, 0, 0)],             # hour 2: hour 1 is empty, hours 0..1 hold impressions 0 and 1
        [row(1, 1, 1, 2, 1, 2)],                                                            # hour 3: hour 2 (1 / 1 for article 2); hours 0..2 (1 click / 2 shown)
        [row(1, 1, 2, 3, 2, 3), row(0, 0, 1, 1, 1, 1)],                                     # impression 3 (same hour) is NOT counted for impression 4
    ]
    for k in range(5):
        assert out[k].dtype == np.float32 and out[k].shape == (len(cands[k]), 6), (k, out[k].shape)
        assert np.allclose(out[k], np.array(want[k]), atol=1e-5), (k, out[k], want[k])
    sub = data_lean.popwin_features_lean(cands, labs, tkeys, 5, np.array([False, True, False, False, True]))
    assert sub[0] is None and sub[2] is None and sub[3] is None and np.allclose(sub[1], out[1]) and np.allclose(sub[4], out[4])
    # strictly causal: flipping every label from the impression's own hour on leaves its features unchanged; flipping an earlier hour moves them
    hours = [0, 0, 2, 3, 3]
    for q in range(5):
        flipped = [[1 - y for y in labs[k]] if hours[k] >= hours[q] else labs[k] for k in range(5)]
        again = data_lean.popwin_features_lean(cands, flipped, tkeys, 5, np.ones(5, bool))
        assert np.array_equal(again[q], out[q]), f"impression {q}: its features depend on labels of its own hour or later"
    early = [[1 - y for y in labs[k]] if hours[k] == 0 else labs[k] for k in range(5)]
    assert not np.array_equal(data_lean.popwin_features_lean(cands, early, tkeys, 5, np.ones(5, bool))[3], out[3]), "labels of an earlier hour must reach later impressions"

    # the loader against a plain-python recomputation from the raw files
    emb = dict(zip(S.meta["ids"], S.E0[1:]))
    d = data_lean.from_mind_lean(mind_train, mind_dev, llm_embeddings=emb, log=lambda *a: None)
    d0 = data_lean.from_mind_lean(mind_train, mind_dev, llm_embeddings=emb, log=lambda *a: None, aux=False)
    assert not hasattr(d0, "aux_pop_validation") and hasattr(d0, "time_validation") and hasattr(d, "aux_pop_test"), "times always, windowed click-through only with aux"
    tr, te = _read_behaviors(os.path.join(mind_train, "behaviors.tsv")), _read_behaviors(os.path.join(mind_dev, "behaviors.tsv"))
    allr = tr + te
    t0 = min(r["t"] for r in allr)
    hour = [(r["t"] - t0) // 3600 for r in allr]
    by_art = {}
    for r, h in zip(allr, hour):
        for c, y in zip(r["cands"], r["labels"]):
            by_art.setdefault(c, []).append((h, y))

    def reference(q):
        b, res = hour[q], []
        for c in allr[q]["cands"]:
            ctr, clk = [], []
            for w in data_lean.POP_WINDOWS:
                inw = [y for h, y in by_art[c] if b - w <= h < b]
                ctr.append(L((sum(inw) + 1.0) / (len(inw) + 20.0)))
                clk.append(np.log1p(sum(inw)))
            res.append(ctr + clk)
        return np.array(res, np.float32)
    pos_of = {r["id"]: i for i, r in enumerate(tr)}
    bad = checked = nonzero = 0
    for r, a in zip(d.validation, d.aux_pop_validation):
        bad += int(not np.allclose(reference(pos_of[r[0]]), a, atol=1e-4))
        nonzero += int((a[:, 3:] > 0).any())
        checked += 1
    for i, a in enumerate(d.aux_pop_test):
        bad += int(not np.allclose(reference(len(tr) + i), a, atol=1e-4))
        nonzero += int((a[:, 3:] > 0).any())
        checked += 1
    assert bad == 0 and checked == len(d.validation) + len(d.test) and nonzero > 50, (bad, checked, nonzero)
    # impression times: the order of the rows, one per impression
    vt = np.array([r["t"] for r in tr if r["id"] in {x[0] for x in d.validation}])
    assert len(d.time_validation) == len(d.validation) and np.array_equal(d.time_validation, vt) and np.array_equal(d.time_test, [r["t"] for r in te])
    # cap_data cuts the new per-impression arrays with the very indices of the rows
    import argparse
    args = argparse.Namespace(val_max=70, test_max=90, train_sub=0)
    full_val, full_test = list(d.validation), list(d.test)
    keep = {k: getattr(d, k) for k in ("aux_validation", "aux_pop_validation", "time_validation", "aux_test", "aux_pop_test", "time_test")}
    quiet(suite.cap_data, d, args)
    iv = np.array([[r[0] for r in full_val].index(r[0]) for r in d.validation])
    it = np.array([[r[0] for r in full_test].index(r[0]) for r in d.test])
    assert len(iv) == 70 and len(it) == 90 and np.all(np.diff(iv) > 0)
    for name, ix in (("validation", iv), ("test", it)):
        assert np.array_equal(getattr(d, "time_" + name), keep["time_" + name][ix]) and isinstance(getattr(d, "time_" + name), np.ndarray)
        for pre in ("aux_", "aux_pop_"):
            got, ref_ = getattr(d, pre + name), keep[pre + name]
            assert len(got) == len(ix) and all(np.array_equal(got[k], ref_[i]) for k, i in enumerate(ix)), (pre, name)
    ok(f"(40) windowed click-through == a plain-python recomputation ({checked} impressions, {nonzero} with clicks in a window); hand-made case; causal (own hour and later never reach "
       "an impression, earlier hours do); impression times; cap_data cuts the new arrays with the rows' indices")


# ------------------------------------------------------------------------------------------------ (41)
def test_entity_features():
    ents = [["A", "B"], ["B"], [], ["C", "A", "B", "D"]]
    idx, idf = FS.entity_tables(ents, 5, emax=2)                                              # article i has global id i + 1; row 0 is the padding
    assert idx.shape == (5, 2) and idf.shape == (5,) and idf[0] == 0 and (idx[0] == 0).all() and (idx[3] == 0).all()
    want = np.log(5.0 / (np.array([2, 3, 1, 1]) + 1.0)) + 1.0                                  # df: A 2, B 3, C 1, D 1 over 4 articles
    assert np.allclose(idf[1:], want, atol=1e-6)
    assert idx[1].tolist() == [1, 2] and idx[2].tolist() == [2, 0], "A, B and then B alone (ids are in order of first sight)"
    assert idx[4].tolist() == [3, 4], "an article with more entities than emax keeps the rarest (C, D) -- ties broken by id"
    rng = np.random.default_rng(1)
    n_news = 80
    ents = [[f"Q{e}" for e in sorted(set(rng.integers(0, 12, rng.integers(0, 7)).tolist()))] for _ in range(n_news - 1)]
    idx, idf = FS.entity_tables(ents, n_news, emax=4)
    kept = {a: [int(x) for x in idx[a] if x > 0] for a in range(n_news)}
    idf_of = {j: float(idf[j]) for j in range(1, len(idf))}
    B, H, C = 8, 9, 7
    hist, hm, cand, cm = np.zeros((B, H), np.int64), np.zeros((B, H), bool), np.zeros((B, C), np.int64), np.zeros((B, C), bool)
    rows = []
    for k in range(B):
        h = rng.integers(1, n_news, rng.integers(0, H + 1)).tolist()
        c = rng.integers(1, n_news, rng.integers(1, C + 1)).tolist()
        hist[k, :len(h)], hm[k, :len(h)], cand[k, :len(c)], cm[k, :len(c)] = h, True, c, True
        rows.append((h, c))
    b = {"hist": torch.from_numpy(hist), "hist_mask": torch.from_numpy(hm), "cand": torch.from_numpy(cand), "cand_mask": torch.from_numpy(cm)}
    f = FS.batch_entity_features(b, torch.from_numpy(idx), torch.from_numpy(idf))
    assert set(f) == set(FS.ENT_NAMES) and all(v.shape == (B, C) and v.dtype == torch.float32 for v in f.values())
    worst = 0.0
    for k, (h, c) in enumerate(rows):
        ref = FS.reference_entity_features(h, c, kept, idf_of)
        for nm in FS.ENT_NAMES:
            got = f[nm][k].numpy()
            worst = max(worst, float(np.abs(got[:len(c)] - ref[nm]).max()))
            assert (got[len(c):] == 0).all(), f"{nm}: padded candidates must be 0"
    assert worst < 1e-5 and float(f["ent_ov"].sum()) > 0 and float(f["ent_ov3"].sum()) > 0 and bool((f["ent_ov3"] <= f["ent_ov"] + 1e-6).all())
    # hand-made: history 1 then 2; candidate 3 shares B with article 1 / 2, candidate 4 shares nothing
    ents = [["A", "B"], ["B"], ["B", "C"], ["D"]]
    idx, idf = FS.entity_tables(ents, 5, emax=4)
    b = {"hist": torch.tensor([[1, 2, 0]]), "hist_mask": torch.tensor([[True, True, False]]), "cand": torch.tensor([[3, 4, 0]]), "cand_mask": torch.tensor([[True, True, False]])}
    h = {k: v[0].numpy() for k, v in FS.batch_entity_features(b, torch.from_numpy(idx), torch.from_numpy(idf)).items()}
    assert np.allclose(h["ent_n"], [2, 1, 0]) and np.allclose(h["ent_frac"], [0.5, 0, 0]) and h["ent_ov"][0] > 0 and h["ent_ov"][1] == 0 and h["ent_ov3"][0] == h["ent_ov"][0]
    empty = FS.batch_entity_features(dict(b, hist_mask=torch.zeros(1, 3, dtype=torch.bool)), torch.from_numpy(idx), torch.from_numpy(idf))
    assert all((v == 0).all() for k, v in empty.items() if k != "ent_n"), "no history: no overlap"
    ok(f"(41) entity tables (IDF, rarest kept) and overlap features == the plain-python definition (max |diff| {worst:.1e}); hand-made case; no history -> 0; padding -> 0")


# ------------------------------------------------------------------------------------------------ (42)
def test_stack_helpers(S_stack):
    rng = np.random.default_rng(0)
    # ---- time folds
    times = rng.integers(0, 86400, 4000)
    blk = ST.time_folds(times, 4)
    assert set(blk) == {0, 1, 2, 3} and np.bincount(blk).min() == 1000 and all(times[blk == k].max() <= times[blk == k + 1].min() for k in range(3))
    assert np.array_equal(blk, ST.time_folds(times, 4)) and ST.buffer_seconds(times, 30) == min(1800.0, 0.02 * (times.max() - times.min())) and ST.buffer_seconds([5], 30) == 0.0
    usable = rng.random(4000) < 0.85
    buf = ST.buffer_seconds(times, 30)
    for k in range(4):
        tr, es, held = ST.time_fold_split(k, blk, times, usable, buf)
        assert np.array_equal(held, np.flatnonzero(blk == k)) and not (set(tr) & set(es)) and not (set(tr) & set(held)) and not (set(es) & set(held))
        lo, hi = times[held].min(), times[held].max()
        for part in (tr, es):
            assert usable[part].all() and ((times[part] <= lo - buf) | (times[part] >= hi + buf)).all(), "the buffer around the held-out block is left out"
        gap = lambda ix: np.where(times[ix] < lo, lo - times[ix], times[ix] - hi)
        assert gap(es).max() <= gap(tr).min() + 1e-9, "early stopping uses the impressions nearest in time to the block"
        assert abs(len(es) - 0.12 * (len(es) + len(tr))) <= 1
    tr, es, held = ST.time_fold_split(0, ST.time_folds(times[:30], 4), times[:30], np.ones(30, bool), 1e9)         # a buffer that eats everything: halves of the pool
    assert len(tr) + len(es) == 0
    # ---- forward split
    tr, es, ev = ST.forward_split(times, usable, buf)
    c1, c2 = np.quantile(times, [0.6, 0.7])
    assert (times[tr] < c1 - buf).all() and ((times[es] >= c1) & (times[es] < c2 - buf)).all() and (times[ev] >= c2).all() and usable[tr].all() and usable[es].all()
    assert not (set(tr) & set(es)) and abs(len(ev) / 4000 - 0.3) < 0.02
    tiny = ST.forward_split(np.arange(10), np.ones(10, bool), 0.0)
    assert sum(len(x) for x in tiny) == 10 and tiny[0].max() < tiny[1].min() < tiny[2].min(), "tiny data: three consecutive parts of the order"
    # ---- new-article weights: a hand-made split (articles 1..6)
    rows = [(0, 0, [], [1, 2], [1, 0], None), (1, 0, [], [3, 4], [0, 1], None), (2, 0, [], [2, 5, 6], [1, 1, 0], None), (3, 0, [], [1, 3], [0, 0], None), (4, 0, [], [5, 6], [0, 1], None)]
    sp = ST.Split(rows)
    shown = ST.shown_in(sp, [0, 1], 7)
    assert shown.tolist() == [False, True, True, True, True, False, False]
    nf = ST.new_fraction(sp, [2, 3, 4, 0], shown)
    assert np.allclose(nf, [0.5, 0.0, 1.0, 0.0]), nf                                           # impression 2: click on 2 (seen) and 5 (new); 3: no click; 4: 6 is new; 0: 1 is seen
    assert ST.restrict_new(sp, np.arange(5), shown, min_n=1, min_frac=0.0).tolist() == [2, 4] and ST.restrict_new(sp, np.arange(5), shown, min_n=10).tolist() == [0, 1, 2, 3, 4]
    assert ST.restrict_new(sp, np.zeros(0, np.int64), shown).size == 0
    res = {"ids": np.array([2, 3, 4]), "arr": {"ndcg@10": np.array([1.0, 0.0, 0.5])}, "mean": {"ndcg@10": 0.5}}
    w_new = np.array([0, 0, 0.5, 0.0, 1.0])
    nd, ws = ST.new_metric(res, w_new)
    assert abs(nd - (0.5 * 1.0 + 1.0 * 0.5) / 1.5) < 1e-12 and ws == 1.5
    assert ST.new_metric(res, np.zeros(5)) == (0.5, 0.0), "no new-article click: the plain mean"
    # ---- coarsening
    names = ["a", "m_log_shown", "m_ctr", "m_age_h", "m_log_prior_shown", "m_expo_6h", "m_ctr_z", "m_age_z", "m_expo_z", "m_expo6_z", "pop", "pop_z", "ctx_n_cand", "um:um_prior_x"]
    ptr = np.arange(0, 301, 3)
    X = rng.random((300, len(names))).astype(np.float32) * 5
    X0 = X.copy()
    Xc = ST.coarsen(X.copy(), names, ptr)
    j = {n: i for i, n in enumerate(names)}
    assert np.array_equal(Xc[:, j["a"]], X0[:, j["a"]]) and np.array_equal(Xc[:, j["ctx_n_cand"]], X0[:, j["ctx_n_cand"]]), "columns outside COARSE / REL_SRC stay"
    for nm, (kind, step) in ST.COARSE.items():
        if nm in j:
            v = np.log1p(X0[:, j[nm]].astype(np.float64)) if kind == "log1p" else X0[:, j[nm]].astype(np.float64)
            q = Xc[:, j[nm]] / step
            assert np.allclose(q, np.round(q), atol=1e-4) and np.abs(Xc[:, j[nm]] - v).max() <= step / 2 + 1e-5, nm
    z = ST.make_seg_z(ptr)
    for zn, src in ST.REL_SRC.items():
        if zn in j and src in j:
            assert np.allclose(Xc[:, j[zn]], z(Xc[:, j[src]]), atol=1e-5), zn
    assert len(np.unique(Xc[:, j["m_age_h"]])) < len(np.unique(X0[:, j["m_age_h"]])), "coarsening merges values"
    sw_X = X0.copy()
    sw = ST.ColumnSwap(sw_X, names, ptr)
    sw.set(sw_X, "lo")
    assert np.array_equal(sw_X, Xc)
    sw.set(sw_X, "hi")
    assert np.array_equal(sw_X, X0), "hi -> lo -> hi restores the table"
    sw.set(sw_X, "lo"); sw.set(sw_X, "lo")
    assert np.array_equal(sw_X, Xc)
    # ---- configuration columns (on the real builder of a finished stack run)
    val, test = _splits(S_stack.data)
    comps = ST.load_components([S_stack.work], val, test)
    bld = ST.Builder(S_stack, comps, ST.GROUPS)
    m = bld.prefix(("scores", "ctx", "meta"))
    c_v15 = ST.config_columns(bld, m, ST.CONFIGS["v15"])
    c_clean = ST.config_columns(bld, m, ST.CONFIGS["clean"])
    c_nop = ST.config_columns(bld, m, ST.CONFIGS["noprior_d5"])
    assert np.array_equal(c_v15, np.arange(m)) and "ctx_n_cand" in bld.names and [bld.names[i] for i in set(c_v15) - set(c_clean)] == ["ctx_n_cand"]
    gone = {bld.names[i] for i in set(c_clean) - set(c_nop)}
    assert gone and all(":um_prior" in n for n in gone), gone
    # ---- the builder's new columns
    n = val.n
    X = bld.build("val", val, 0, n)
    assert bld.ent is not None and bld.has_popw and X.shape[1] == bld.n_features
    j0 = int(np.flatnonzero(bld.group_of == "ctx")[0]) + len(FS.NAMES)
    assert bld.names[j0:j0 + 4] == ["ctx_" + k for k in FS.ENT_NAMES]
    idx, idf = FS.entity_tables(S_stack.meta["ents"], S_stack.data.n_news)
    kept = {a: [int(x) for x in idx[a] if x > 0] for a in range(S_stack.data.n_news)}
    idf_of = {k: float(idf[k]) for k in range(1, len(idf))}
    worst = 0.0
    for i in range(0, n, 11):
        row = val.rows[i]
        ref = FS.reference_entity_features([int(x) for x in list(row[2])[-50:]], [int(x) for x in row[3]], kept, idf_of)
        got = X[val.ptr[i]:val.ptr[i + 1], j0:j0 + 4]
        worst = max(worst, float(np.abs(got - np.stack([ref[k] for k in FS.ENT_NAMES], 1)).max()))
    assert worst < 1e-4, worst
    jp = int(np.flatnonzero(bld.group_of == "pop")[0])
    assert bld.names[jp:jp + 9] == ["pop", "pop_z", *ST.POPW_NAMES]
    pw = np.concatenate(S_stack.data.aux_pop_validation)
    assert np.array_equal(X[:, jp + 2:jp + 8], pw) and np.allclose(X[:, jp + 8], common.ragged_zscore({"ids": np.arange(n), "ptr": val.ptr, "flat": pw[:, 1]})["flat"], atol=1e-4)
    raises(ValueError, bld.build, "val", ST.Split(S_stack.data.validation, S_stack.data.aux_validation), 0, 5)        # windowed click-through asked for, none on the split
    ok(f"(42) stacker helpers: time folds (blocks, buffer, early stopping nearest in time), forward split, new-article weights, coarsening (rounding, z-columns follow), "
       f"column swap round trip, configuration columns; builder entity columns == the plain-python definition (max |diff| {worst:.1e}), windowed click-through columns == the loader's")


# ------------------------------------------------------------------------------------------------ (43)
def _planted_rows(rng, with_signal=True):
    """Articles live for 6 hours of a day; each has a hidden click-through; impressions list 8 live articles.  ``sig`` is a noisy content proxy of the hidden rate
    (generalises to new articles); ``fp`` is the article's id as a number (can only memorise)."""
    n_art, life, day = 500, 6 * 3600, 86400
    start = rng.uniform(-life, day, n_art)                                                   # live articles at every time of the day (about a fifth of them)
    logit = rng.normal(-1.4, 1.5, n_art)
    rows, feats, times = [], [], []
    for i in range(6000):
        t = rng.uniform(0, day)
        alive = np.flatnonzero((start <= t) & (t < start + life))
        c = rng.choice(alive, 8, replace=False)
        y = (rng.random(8) < 1 / (1 + np.exp(-logit[c]))).astype(np.float32)
        rows.append((i, int(rng.integers(0, 300)), [], (c + 1).tolist(), y.tolist(), None))
        feats.append(np.stack([logit[c] * 0.5 + rng.normal(0, 1.5, 8), (c + 1).astype(np.float64)], 1))
        times.append(int(t))
    return rows, np.concatenate(feats).astype(np.float32), np.array(times, np.int64), n_art + 1


def test_planted_memorisation():
    rng = np.random.default_rng(3)
    rows, X, times, n_news = _planted_rows(rng)
    sp = ST.Split(rows, times=times)
    n = sp.n
    y = sp.labels(0, n)
    usable = np.array([float(np.sum(r[4])) > 0 for r in rows]) & (sp.lens > 1)
    buf = ST.buffer_seconds(times, 30)
    be = ST.make_backend("xgboost", "cpu")
    base = {"eta": 0.1, "rounds": 200, "patience": 20, "seed": 0, "depth": 7, "mcw": 5}
    clicked = np.flatnonzero(usable)                                                       # impressions with a click: the mean the new-article mean is comparable with

    def ndcg(flat, imp, weights=None):
        v = np.array([metrics.ndcg_score(np.asarray(rows[i][4]), flat[sp.ptr[i]:sp.ptr[i + 1]].astype(np.float32), 10) for i in imp])
        return common.weighted_mean(v, weights)

    def oof(cv, cols):
        specs, _, w_new = ST.make_folds(cv, 4, 0, sp, usable, times, n_news, buf)
        flat = np.zeros(sp.N, np.float32)
        for tr, es, held in specs:
            fm = ST.fit_cols(be, X, y, sp, tr, es, cols, "binary:logistic", base)
            r = ST.expand(sp.ptr, sp.lens, held)
            flat[r] = ST.predict_rows(be, fm, X, cols, r)
        return ndcg(flat, clicked), ndcg(flat, np.arange(n), w_new)

    u_all, _ = oof("user", [0, 1])
    t_all, t_new = oof("time", [0, 1])
    u_all0, _ = oof("user", [0])
    t_all0, t_new0 = oof("time", [0])
    # with the fingerprint column the user folds are optimistic: the model has seen the same article's clicks of the same hours from other users; time folds remove most of it
    # and the new-article slice of the time folds is what the test day looks like (no article of it was ever shown to the model)
    assert u_all - t_new > 0.03 and u_all - t_all > 0.02 and t_new < t_all + 0.005, (u_all, t_all, t_new)
    # without it (only the content proxy) the folds agree -- so the gap above is memorisation, not a property of the folds
    assert abs(u_all0 - t_new0) < 0.02 and abs(u_all0 - t_all0) < 0.02, (u_all0, t_all0, t_new0)
    assert u_all - t_new > 3 * abs(u_all0 - t_new0) + 0.02, (u_all - t_new, u_all0 - t_new0)
    ok(f"(43) planted same-day memorisation: with the article-id column user-fold nDCG@10 {u_all:.3f} vs {t_new:.3f} on new-article clicks of time folds (all impressions {t_all:.3f}); "
       f"without it {u_all0:.3f} vs {t_new0:.3f} -- the new-article slice of time folds exposes what user folds hide")


# ------------------------------------------------------------------------------------------------ (44)
def test_weighted_fusion(S):
    rows = S.data.validation
    n = len(rows)
    rng = np.random.default_rng(5)
    seen = np.zeros(S.data.n_news, bool)
    seen[1:S.data.n_news // 2] = True
    w, info = common.cold_weights(rows, seen, 0.9)
    assert 0 < info["cold_frac"] < 1 and w.min() > 0 and not np.allclose(w, 1)
    cold_pos = np.array([float(np.mean([not seen[c] for c, y in zip(r[3], r[4]) if y > 0.5])) if any(y > 0.5 for y in r[4]) else np.nan for r in rows])

    def comp(name, strength_cold, strength_warm):
        rr = []
        for i, r in enumerate(rows):
            s = strength_cold if (np.nan_to_num(cold_pos[i]) > 0.5) else strength_warm
            rr.append((s * np.asarray(r[4], np.float64) + rng.standard_normal(len(r[3]))).astype(np.float32))
        rag = common.make_ragged(np.arange(n), rr)
        return SM.Comp("t", name, "zs", rag, rag, {})
    warm_good, cold_good = comp("warm_good", 0.4, 2.5), comp("cold_good", 2.5, 0.4)
    comps = [warm_good, cold_good]
    plain = SM.Fusion(rows, comps, "val", np.arange(n), "cpu", es=S.es["val"])
    ones = SM.Fusion(rows, comps, "val", np.arange(n), "cpu", es=S.es["val"], imp_w=np.ones(n))
    wf = SM.Fusion(rows, comps, "val", np.arange(n), "cpu", es=S.es["val"], imp_w=w)
    for vec in ([1, 0], [0, 1], [1, 0.5]):
        assert abs(plain.ndcg10(vec) - ones.ndcg10(vec)) < 1e-6, "all-ones weights == the plain mean"
        imps = [(np.array(r[4]), sum(wi * zrow(common.ragged_rows(c.val)[i]) for wi, c in zip(vec, comps))) for i, r in enumerate(rows)]
        per = np.array([metrics.ndcg_score(a, b.astype(np.float32), 10) for a, b in imps])
        assert abs(wf.ndcg10(vec) - float((per * w).sum() / w.sum())) < 1e-5, (vec, wf.ndcg10(vec))
    sub = np.arange(0, n, 2)
    ws = SM.Fusion(rows, comps, "val", sub, "cpu", imp_w=w)
    per_all = np.array([metrics.ndcg_score(np.array(rows[i][4]), zrow(common.ragged_rows(warm_good.val)[i]).astype(np.float32), 10) for i in sub])
    assert abs(ws.ndcg10([1, 0]) - float((per_all * w[sub]).sum() / w[sub].sum())) < 1e-5, "weights follow the impressions of a subset"
    # the plain validation mean and the cold-weighted one disagree about which component is better iff the split's mix differs from the target
    p0, p1 = plain.ndcg10([1, 0]), plain.ndcg10([0, 1])
    w0, w1 = wf.ndcg10([1, 0]), wf.ndcg10([0, 1])
    assert (w1 - w0) > (p1 - p0) + 1e-3, (p0, p1, w0, w1)             # re-weighting towards cold clicks favours the component that is good on cold ones
    ok(f"(44) supermodel Fusion: weighted nDCG@10 == the manual weighted mean (also on a subset); all-ones weights == plain; the cold-weighted objective moves towards the "
       f"component that is better on cold clicks (cold_good - warm_good: plain {p1 - p0:+.4f} -> weighted {w1 - w0:+.4f})")


# ------------------------------------------------------------------------------------------------ (45) / (46)
def test_restack_and_release(tmp, mind, enc_dir, cache_dir, run_args):
    work = os.path.join(tmp, "outRestack")
    shutil.rmtree(work, ignore_errors=True)
    common_args = [*mind, "--work", work, "--encoder", enc_dir, "--cache-dir", cache_dir, "--lean-data", "1", *[a for a in run_args if a != "--no-resume"], "--stack-folds", "3",
                   "--stack-rounds", "30", "--stack-configs", "clean,clean_d5", "--stack-control", "0", "--stack-chunk", "90"]
    S1, t1 = quiet(suite.main, [*common_args, "--stages", "controls,pooling,stack"])
    assert S1.kv["stage_status"]["stack"] == "done" and "stack stage: settings changed" not in t1
    sig = S1.kv["stack_sig"]
    rows1 = [r["name"] for r in S1.records]
    # same settings: everything is resumed, nothing runs again
    S2, t2 = quiet(suite.main, [*common_args, "--stages", "controls,pooling,stack"])
    assert "stage 'stack' already finished" in t2 and "== T stacker" not in t2 and S2.kv["stack_sig"] == sig and [r["name"] for r in S2.records] == rows1
    # a changed stacker setting re-runs the stack stage alone and does not duplicate its rows
    S3, t3 = quiet(suite.main, [*common_args, "--stack-blend", "1", "--stages", "controls,pooling,stack"])
    assert "stack stage: settings changed" in t3 and "== T stacker" in t3 and "== H8 pooling" not in t3 and "stage 'pooling' already finished" in t3, t3[-600:]
    assert S3.kv["stack_sig"] != sig and [r["name"] for r in S3.records].count("stack_llm") == 1 and len(S3.kv["stack"]["members"]) == 1
    assert sorted(r["name"] for r in S3.records) == sorted(rows1), "the same rows, once each"
    # --restack forces it with the same settings
    S4, t4 = quiet(suite.main, [*common_args, "--stack-blend", "1", "--restack", "1", "--stages", "controls,pooling,stack"])
    assert "stack stage: settings changed (or --restack 1)" in t4 and "== T stacker" in t4 and "== H8 pooling" not in t4
    # a setting that changes the other stages starts fresh
    S5, t5 = quiet(suite.main, [*common_args, "--um-dim", "32", "--stages", "controls,pooling"])
    assert "belongs to a different configuration" in t5
    ok("(45) restart: unchanged settings resume everything; a changed --stack-* setting (or --restack 1) re-runs the stack stage alone without duplicating rows; "
       "a changed non-stack setting starts fresh")
    # ---- release
    S_a, _ = quiet(suite.main, [*common_args, "--work", os.path.join(tmp, "outRelA"), "--stages", "controls,pooling", "--no-resume"])
    S_b, _ = quiet(suite.main, [*common_args, "--work", os.path.join(tmp, "outRelB"), "--stages", "controls,pooling", "--no-resume"])
    before = {r["name"]: r["ndcg@10"] for r in S_a.records}
    n_cache = len(S_a.cache)
    S_a.release()
    assert S_a.data is None and S_a.es is None and S_a.E0 is None and S_a.meta is None and S_a.store == {}
    assert {r["name"]: r["ndcg@10"] for r in S_a.records} == before and len(S_a.cache) == n_cache and S_a.kv["stage_status"]
    rows, text = quiet(suite.leaderboard, [S_a, S_b], ["a", "b"], ("b", "frozen"), None)
    assert rows and "LEADERBOARD" in text
    pairs, _ = quiet(suite.cross_compare, [S_a, S_b], ["a", "b"], [(("a", "pool_lse"), ("b", "pool_lse"))])
    assert pairs and pairs[0]["n"] > 0 and abs(pairs[0]["d_ndcg@10"][0]) < 1e-9, "two identical runs agree after one of them was released"
    S_a.release()                                                                       # idempotent
    ok("(46) Suite.release(): data / evaluation sets / embeddings are freed, the records, per-impression results and facts survive, leaderboard and cross_compare still work, "
       "releasing twice is harmless")


# ------------------------------------------------------------------------------------------------ (47)
def test_concat_encoder(tmp, mind, enc_dir, cache_dir, run_args, S):
    import argparse
    import adapt
    import stages_ext as X
    dec_dir = os.path.join(os.path.dirname(enc_dir), "tiny-qwen3-embedding")
    meta = S.meta
    a = argparse.Namespace(encoder=f"concat:{enc_dir}+{dec_dir}", max_len=32, doc_prefix="", enc_bs=64)
    cdir = os.path.join(tmp, "concat_cache")
    shutil.rmtree(cdir, ignore_errors=True)
    V, secs = suite.embed_catalogue(meta, a, "cpu", cdir)
    A_, _ = X.embed_news(meta, enc_dir, 32, "", "cpu", 64, cdir)
    B_, _ = X.embed_news(meta, dec_dir, 32, "", "cpu", 64, cdir)
    nrm = lambda v: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    assert V.shape == (len(meta["ids"]), A_.shape[1] + B_.shape[1]) and V.dtype == np.float32 and secs > 0
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-5), "a unit vector"
    assert np.allclose(V[:, :A_.shape[1]], nrm(A_) / np.sqrt(2), atol=1e-6) and np.allclose(V[:, A_.shape[1]:], nrm(B_) / np.sqrt(2), atol=1e-6)
    i, j = 3, 11
    assert abs(float(V[i] @ V[j]) - 0.5 * (float(nrm(A_)[i] @ nrm(A_)[j]) + float(nrm(B_)[i] @ nrm(B_)[j]))) < 1e-5, "dot product = mean of the parts' dot products"
    V2, secs2 = suite.embed_catalogue(meta, a, "cpu", cdir)
    assert secs2 == 0 and np.array_equal(V, V2), "served from the cache of the single encoders"
    single = argparse.Namespace(encoder=enc_dir, max_len=32, doc_prefix="", enc_bs=64)
    One, _ = suite.embed_catalogue(meta, single, "cpu", cdir)
    assert np.array_equal(One, A_), "a plain encoder name is unchanged"
    raises(SystemExit, suite.embed_catalogue, meta, argparse.Namespace(encoder=f"concat:{enc_dir}", max_len=32, doc_prefix="", enc_bs=64), "cpu", cdir)
    # a whole run on the composite encoder: zero-shot control + the learned readers over 2 x the width
    ref_keep = suite.EQUIV_REFERENCE
    suite.EQUIV_REFERENCE = lambda t: (_ for _ in ()).throw(AssertionError("the pooling guard compares one encoder with its reference; it must not run for a composite"))
    try:
        Sc, _ = quiet(suite.main, [*mind, "--work", os.path.join(tmp, "outConcat"), "--encoder", a.encoder, "--cache-dir", cdir, "--lean-data", "1",
                                   *[x for x in run_args if x != "--no-resume"], "--no-resume", "--stages", "controls,graph,usermodel", "--um-views", "frozen,knn_rag", "--um-extra", "prior:frozen",
                                   "--um-dim", "32"])
    finally:
        suite.EQUIV_REFERENCE = ref_keep
    assert Sc.E0.shape[1] == V.shape[1] and {"frozen", "um_frozen", "um_knn_rag", "um_prior_frozen"} <= {r["name"] for r in Sc.records}
    ok("(47) concat:A+B -> unit vectors, dot product == mean of the parts' dot products, served from the single encoders' cache, a lone part is an error, "
       "a whole run on it (controls, graph, readers with --um-dim 32) works and skips the single-encoder pooling guard")


# ------------------------------------------------------------------------------------------------ (48)
def test_knn_view_memory():
    """The v16 run on MIND-large (104k articles, k=20) died in ``Memory.knn_view``: the gather ``E[nbr]`` is ``[articles, k, d]`` float32 and its product with the
    weights is a second one -- 2 x 8.5 GB at d=1024, 2 x 17 GB at d=2048 on top of ~12 GB.  Toy sizes never showed it (test 47 runs the whole concat pipeline on
    a few thousand articles), so this test pins the *law*: the view equals the one-shot definition and allocates no more than a block of it."""
    import tracemalloc
    import semgraph as G
    rng = np.random.default_rng(0)
    n, d, k = 2400, 96, 20
    E = rng.standard_normal((n, d)).astype(np.float32)
    E[0] = 0
    E[1:] /= np.linalg.norm(E[1:], axis=1, keepdims=True)
    mem = rng.random(n) < 0.8
    mem[0] = False
    m = G.Memory(E, None, mem, "cpu")
    rows = np.arange(1, n)
    nbr, sim = m.neighbours(rows, k)
    w = np.clip(sim, 0, None)
    w = w / (w.sum(1, keepdims=True) + 1e-8)
    want = E.copy()
    want[rows] = G._norm(E[rows] + 0.5 * G._norm((E[nbr] * w[:, :, None]).sum(1)))                  # the definition, in one shot
    full = n * k * d * 4                                                                              # one [articles, k, d] float32 array
    peaks = {}
    for name, budget in (("one block", 1e12), ("300 articles", 300 * k * d), ("floor of 64 articles", 1)):
        tracemalloc.start()
        got = m.knn_view(0.5, k, budget=budget)
        peaks[name] = tracemalloc.get_traced_memory()[1] / full
        tracemalloc.stop()
        assert np.allclose(got, want, atol=1e-6, rtol=0), (name, float(np.abs(got - want).max()))
        assert np.array_equal(got[0], E[0]), "the padding row stays zero"
    assert peaks["one block"] > 1.8, f"the one-shot path must show the 2 x [n,k,d] it allocates (else this test cannot see the bug): {peaks}"
    assert peaks["300 articles"] < 0.45 and peaks["floor of 64 articles"] < 0.25, peaks
    sub = np.array([5, 17, 800, 2301])                                                                # a subset of rows: the others are untouched
    part = m.knn_view(0.5, k, rows=sub, budget=1)
    keep = np.setdiff1d(np.arange(n), sub)
    assert np.array_equal(part[keep], E[keep]) and np.allclose(part[sub], want[sub], atol=1e-6)
    ok(f"(48) Memory.knn_view == the one-shot definition (max |diff| < 1e-6); peak allocation {peaks['one block']:.1f} x [n,k,d] in one block, "
       f"{peaks['300 articles']:.2f} x with 300-article blocks, {peaks['floor of 64 articles']:.2f} x at the floor; a row subset leaves the rest untouched")


# ------------------------------------------------------------------------------------------------ (49)
def test_peak_ram():
    common.PEAK.reset()
    base = common.PEAK.gb()
    big = np.ones(60_000_000, np.float32)                                                             # 240 MB, touched
    time.sleep(1.0)
    seen = common.PEAK.gb()
    del big
    gc.collect()
    time.sleep(0.6)
    assert seen >= base + 0.2, (base, seen)
    assert common.PEAK.gb() >= seen - 1e-3, "a peak does not fall back when the memory is freed"
    assert common.lifetime_peak_gb() >= seen - 0.05
    common.PEAK.reset()
    assert common.PEAK.gb() < seen - 0.15, "reset() starts a new stage"
    note = common.mem_note()
    assert "stage peak" in note and "run peak" in note, note
    assert common.ram_total_gb() > 0
    # the alarm: silent below 70% of the memory the process may use, then one line per 5% step naming the stage and where the main thread is
    import contextlib
    import io
    import threading
    pk, lim, me = common.PeakRAM(alarm=0.7), common.ram_total_gb() * 1e9, threading.main_thread().ident
    pk.reset("usermodel")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for frac in (0.5, 0.8, 0.82, 0.9):                                                        # silent | warns | < 5% above the last warning: silent | warns
            pk._warn(frac * lim, me)
    out = buf.getvalue()
    assert out.count("!! RAM") == 2 and "stage 'usermodel'" in out and "test_peak_ram" in out and "80%" in out and "90%" in out, out
    ok(f"(49) PeakRAM: a 240 MB transient is seen in the stage peak ({base:.2f} -> {seen:.2f} GB), survives the free, reset() starts a new stage; "
       f"mem_note reports now / stage peak / run peak; the alarm is silent below 70%, then warns once per 5% step with the stage and the main thread's stack; "
       f"container memory {common.ram_total_gb():.0f} GB")


# ------------------------------------------------------------------------------------------------ driver
def run_all(tmp, mind, enc_dir, cache_dir, run_args, S, mind_train, mind_dev, S_stack):
    """Runs (39)-(49).  ``S``: the main selftest suite; ``S_stack``: a finished run of the stack stages on the lean data."""
    test_knn_view_memory()
    test_peak_ram()
    test_cold_weights()
    test_popwin_and_times(S, mind_train, mind_dev)
    test_entity_features()
    test_stack_helpers(S_stack)
    test_planted_memorisation()
    test_weighted_fusion(S)
    test_restack_and_release(tmp, mind, enc_dir, cache_dir, run_args)
    test_concat_encoder(tmp, mind, enc_dir, cache_dir, run_args, S)
