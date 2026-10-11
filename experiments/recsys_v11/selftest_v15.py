"""Tests (32)-(38) of the v15 additions: label-free exposure features, the history / slate features, the stacker's feature table, its leakage
guards, the gradient-boosting backends, the stage end to end and the history-free prior reader.  Called by ``selftest.main()``.

    (32)  data_lean: exposure features (age, shown before, shown in the last 1 / 6 / 24 completed hours) and per-article training counts equal a naive
          re-computation from the raw files; a hand-made case with ties in time
    (33)  featscore.batch_features == the plain-python definition, padding is 0
    (34)  stacker.Builder: every column of the feature table equals an independent recomputation; chunked == whole; variants are column prefixes
    (35)  guards: folds are by user and disjoint; the test table reads no label; a whole second run with the TEST labels shuffled gives identical
          test scores for the offline variants (only the streaming popularity may change); a fold's out-of-fold scores do not depend on its own labels
    (36)  backends (xgboost / lightgbm / scikit-learn) learn a planted ranking signal; GPU request falls back to CPU; import failures fall through
    (37)  stage end to end: rows, families, references, out-of-fold == the validation row, test == mean of the fold models' z-scores, decline paths
    (37b) the final report's stacker section and the published-numbers table
    (38)  the history-free prior reader ignores the history, trains, is exported as stacker-only input and never enters the supermodel
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from bisect import bisect_left
from datetime import datetime

import numpy as np
import torch

import common
import data_lean
import featscore as FS
import metrics
import stacker as ST
import suite
import usermodel as UM
from selftest_v14 import ok, quiet, raises

TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"
STACK_STAGES = "controls,pooling,moredata,graph,usermodel,supermodel,stack"
STACK_ARGS = ["--lean-data", "1", "--um-extra", "prior:frozen,mi4:frozen", "--stack-folds", "4", "--stack-rounds", "40", "--stack-eta", "0.1", "--stack-depth", "4",
              "--stack-objectives", "rank:ndcg,binary:logistic",
              "--stack-chunk", "90", "--stack-save-scores", "2", "--no-resume"]


# ------------------------------------------------------------------------------------------------ (32)
def _read_behaviors(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            t = datetime.strptime(p[2].strip(), TS_FORMAT)
            items = [x.rsplit("-", 1) for x in p[4].split()]
            out.append({"id": p[0], "t": t.toordinal() * 86400 + t.hour * 3600 + t.minute * 60 + t.second, "cands": [a for a, _ in items], "labels": [int(b) for _, b in items]})
    return out


def test_exposure_and_stats(S, mind_train, mind_dev):
    # --- a hand-made case: ties in time, hour windows, an article that never appeared before
    cands = [[1, 2], [1, 3], [1, 2, 4], [2], [1, 3]]
    tkeys = [0, 100, 2 * 3600 + 5, 3 * 3600, 3 * 3600]
    out = data_lean.exposure_features_lean(cands, tkeys, 5, np.ones(5, bool))
    L = np.log1p
    want = [
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[100 / 3600, L(1), 0, 0, 0], [0, 0, 0, 0, 0]],
        [[7205 / 3600, L(2), 0, L(2), L(2)], [7205 / 3600, L(1), 0, L(1), L(1)], [0, 0, 0, 0, 0]],
        [[3, L(2), L(1), L(2), L(2)]],
        [[3, L(3), L(1), L(3), L(3)], [(10800 - 100) / 3600, L(1), 0, L(1), L(1)]],
    ]
    for k in range(5):
        assert out[k].dtype == np.float32 and out[k].shape == (len(cands[k]), 5), (k, out[k].shape)
        assert np.allclose(out[k], np.array(want[k]), atol=1e-5), (k, out[k], want[k])
    sub = data_lean.exposure_features_lean(cands, tkeys, 5, np.array([False, True, False, False, True]))
    assert sub[0] is None and sub[2] is None and sub[3] is None and np.allclose(sub[1], out[1]) and np.allclose(sub[4], out[4]), \
        "only wanted impressions are scored, and the scores do not depend on the others being wanted"
    pos, shown = data_lean.item_stats_lean([[1, 2], [1], [2, 3]], [[1, 0], [0], [1, 1]], [0, 2], 4, chunk=1)
    assert pos.tolist() == [0, 1, 1, 1] and shown.tolist() == [0, 1, 2, 1], (pos, shown)

    # --- the loader, against a plain-python recomputation from the raw files (independent of data_lean's parser, loops and arrays)
    emb = dict(zip(S.meta["ids"], S.E0[1:]))
    d = data_lean.from_mind_lean(mind_train, mind_dev, llm_embeddings=emb, log=lambda *a: None)
    d0 = data_lean.from_mind_lean(mind_train, mind_dev, llm_embeddings=emb, log=lambda *a: None, aux=False)
    assert not hasattr(d0, "aux_validation") and not hasattr(d0, "item_stats") and hasattr(d, "item_stats")
    tr, te = _read_behaviors(os.path.join(mind_train, "behaviors.tsv")), _read_behaviors(os.path.join(mind_dev, "behaviors.tsv"))
    allr = tr + te
    t0 = min(r["t"] for r in allr)
    hour = [(r["t"] - t0) // 3600 for r in allr]
    times, hours = {}, {}
    for r, h in zip(allr, hour):
        for c in r["cands"]:
            times.setdefault(c, []).append(r["t"])
            hours.setdefault(c, []).append(h)
    for c in times:
        times[c].sort()
        hours[c].sort()

    def reference(q):
        now, b = allr[q]["t"], hour[q]
        rows = []
        for c in allr[q]["cands"]:
            ts, hs = times[c], hours[c]
            before = bisect_left(ts, now)                                    # impressions strictly earlier in time
            age = (now - ts[0]) / 3600.0 if before > 0 else 0.0
            win = [bisect_left(hs, b) - bisect_left(hs, max(b - w, 0)) for w in data_lean.EXPO_WINDOWS]
            rows.append([age, np.log1p(before)] + [np.log1p(x) for x in win])
        return np.array(rows, np.float32)

    pos_of = {r["id"]: i for i, r in enumerate(tr)}
    bad, checked, nonzero_win = 0, 0, 0
    for r, a in zip(d.validation, d.aux_validation):
        bad += int(not np.allclose(reference(pos_of[r[0]]), a, atol=1e-4))
        nonzero_win += int((a[:, 2:] > 0).any())
        checked += 1
    for i, (r, a) in enumerate(zip(d.test, d.aux_test)):
        bad += int(not np.allclose(reference(len(tr) + i), a, atol=1e-4))
        nonzero_win += int((a[:, 2:] > 0).any())
        checked += 1
    assert bad == 0 and checked == len(d.validation) + len(d.test) and nonzero_win > 50, (bad, checked, nonzero_win)
    # --- per-article counts of the training rows (not of validation / test)
    val_ids = {r[0] for r in d.validation}
    names = {n: i + 1 for i, n in enumerate(S.meta["ids"])}
    shown, clicked = np.zeros(d.n_news), np.zeros(d.n_news)
    for r in tr:
        if r["id"] in val_ids:
            continue
        for c, y in zip(r["cands"], r["labels"]):
            shown[names[c]] += 1
            clicked[names[c]] += y
    shown_tv, clicked_tv = shown.copy(), clicked.copy()
    for r in tr:                                                                 # + the validation days: what is labelled before the test day
        if r["id"] in val_ids:
            for c, y in zip(r["cands"], r["labels"]):
                shown_tv[names[c]] += 1
                clicked_tv[names[c]] += y
    st = d.item_stats
    assert np.array_equal(st["trainval_shown"], shown_tv.astype(np.float32)) and np.array_equal(st["trainval_pos"], clicked_tv.astype(np.float32)) and shown_tv.sum() > shown.sum()
    assert st["train_shown"].dtype == np.float32 and np.array_equal(st["train_shown"], shown.astype(np.float32)) and np.array_equal(st["train_pos"], clicked.astype(np.float32))
    assert shown.sum() == 10 * d.split_info["train_core_impressions"], "every fixture impression shows 10 articles; only training-core impressions are counted"
    ok(f"(32) exposure features (age, shown before, shown in the last 1 / 6 / 24 completed hours) and per-article training counts == a plain-python recomputation "
       f"({checked} impressions, {nonzero_win} with a non-empty window); hand-made case with ties in time; aux=False builds nothing")


# ------------------------------------------------------------------------------------------------ (33)
def test_featscore(S):
    n_news = S.data.n_news
    cat, sub = FS.code_arrays(S.meta, n_news)
    assert cat[0] == 0 and sub[0] == 0 and cat[1:].min() >= 1 and len(set(cat[1:])) == len(set(S.meta["cats"]))
    E = S.tensor(S.E0)
    cat_t, sub_t = torch.from_numpy(cat), torch.from_numpy(sub)
    vec = S.E0.astype(np.float64)
    worst, n_rows, with_hist = 0.0, 0, 0
    for bi, b in enumerate(S.es["test"]):
        f = FS.batch_features(b, E, cat_t, sub_t)
        assert set(f) == set(FS.NAMES) and all(v.shape == b["cand"].shape and v.dtype == torch.float32 for v in f.values())
        cm = b["cand_mask"].numpy()
        for k, r in enumerate(b["rows"][:80]):
            row = S.data.test[r]
            hist = [int(x) for x in list(row[2])[-b["hist"].shape[1]:]]
            cands = [int(x) for x in row[3]]
            ref = FS.reference_features(hist, cands, vec, cat, sub)
            for name in FS.NAMES:
                got = f[name][k].numpy()
                worst = max(worst, float(np.abs(got[:len(cands)] - ref[name]).max()))
                assert (got[~cm[k]] == 0).all(), f"{name}: padded candidates must be 0"
            n_rows += 1
            with_hist += int(len(hist) > 0)
        if bi >= 3:
            break
    assert worst < 2e-4 and n_rows > 40 and with_hist > 20, (worst, n_rows, with_hist)
    # a hand-made impression: history 1, 3, 2 (the last click is article 2); candidates 1 (clicked before), 4, 5
    Eh = torch.zeros(6, 3)
    Eh[1] = torch.tensor([1.0, 0, 0]); Eh[2] = torch.tensor([0, 1.0, 0]); Eh[3] = torch.tensor([0, 0, 1.0]); Eh[4] = torch.tensor([0.6, 0.8, 0]); Eh[5] = torch.tensor([0, 0.6, 0.8])
    ct = torch.tensor([0, 1, 1, 2, 2, 1])                                       # categories: articles 1, 2, 5 -> 1; 3, 4 -> 2
    st = torch.tensor([0, 1, 2, 3, 4, 5])                                       # every article its own sub-category
    b = {"hist": torch.tensor([[1, 3, 2, 0]]), "hist_mask": torch.tensor([[True, True, True, False]]),
         "cand": torch.tensor([[1, 4, 5, 0]]), "cand_mask": torch.tensor([[True, True, True, False]])}
    h = {k: v[0].numpy() for k, v in FS.batch_features(b, Eh, ct, st).items()}
    assert np.allclose(h["in_hist"], [1, 0, 0, 0]) and np.allclose(h["last1"], [0, 0.8, 0.6, 0], atol=1e-6)
    assert np.allclose(h["last3"], [1 / 3, 1.4 / 3, 1.4 / 3, 0], atol=1e-6)
    assert np.allclose(h["cat_share"], [2 / 3, 1 / 3, 2 / 3, 0]) and np.allclose(h["sub_share"], [1 / 3, 0, 0, 0]) and np.allclose(h["cat_last"], [1, 0, 1, 0])
    assert np.allclose(h["n_hist"], [3, 3, 3, 0]) and np.allclose(h["n_cand"], [3, 3, 3, 0]) and np.allclose(h["slate_cat"], [2 / 3, 1 / 3, 2 / 3, 0])
    assert np.allclose(h["slate_mean"], [0.3, 0.54, 0.24, 0], atol=1e-6) and np.allclose(h["slate_max"], [0.6, 0.6, 0.48, 0], atol=1e-6)
    # an impression without history: everything that is about the history is 0
    f0 = {k: v[0].numpy() for k, v in FS.batch_features(dict(b, hist_mask=torch.zeros(1, 4, dtype=torch.bool)), Eh, ct, st).items()}
    assert all((f0[k] == 0).all() for k in ("last1", "last3", "in_hist", "cat_share", "sub_share", "cat_last", "n_hist")) and np.allclose(f0["slate_cat"], h["slate_cat"])
    ok(f"(33) featscore.batch_features == the plain-python definition ({n_rows} test impressions, max |diff| {worst:.1e}); hand-made case; empty history -> 0; padding -> 0")


# ------------------------------------------------------------------------------------------------ (34)
def test_builder(S):
    """``S``: a finished run of the stage list that exports scores, on the lean data (exposure features present)."""
    val, test = _splits(S.data)
    comps = ST.load_components([S.work], val, test)
    names = {c.name for c in comps}
    assert {"frozen", "pool_lse", "pool_topk", "txt_cta_lse", "um_frozen", "um_prior_frozen", "knn_rag"} <= names and not any(n.startswith("txt_ctr") for n in names), sorted(names)
    assert {c.kind for c in comps} == {"zs", "aux", "reader"} and all(c.kind == "aux" for c in comps if c.name in ("frozen", "um_prior_frozen"))
    bld = ST.Builder(S, comps, ST.GROUPS)
    n = val.n
    X = bld.build("val", val, 0, n)
    assert X.shape == (val.N, bld.n_features) and X.dtype == np.float32 and np.isfinite(X).all()
    parts = [bld.build("val", val, a, min(a + 70, n)) for a in range(0, n, 70)]
    assert np.array_equal(np.concatenate(parts), X), "a chunk must give the rows the whole table has"
    # scores: z-score inside the impression, raw value for a few scorers; the exported file is what the column holds
    j = 0
    for c in comps:
        rag = {"ids": np.arange(n), "ptr": val.ptr, "flat": c.flat["val"]}
        assert np.allclose(X[:, j], common.ragged_zscore(rag)["flat"], atol=1e-5), c.key
        j += 1
        if c.raw:
            assert np.array_equal(X[:, j], c.flat["val"]), c.key
            j += 1
    zc = common.load_scores(os.path.join(S.work, "scores", "pool_lse.npz"))
    assert np.array_equal(next(c for c in comps if c.name == "pool_lse").flat["val"], zc["val"]["flat"])
    # ctx: the plain-python definition on a sample of impressions
    cat, sub = FS.code_arrays(S.meta, S.data.n_news)
    vec = S.E0.astype(np.float64)
    j0 = int(np.flatnonzero(bld.group_of == "ctx")[0])
    for i in range(0, n, 17):
        row = val.rows[i]
        ref = FS.reference_features([int(x) for x in list(row[2])[-50:]], [int(x) for x in row[3]], vec, cat, sub)
        assert np.allclose(X[val.ptr[i]:val.ptr[i + 1], j0:j0 + len(FS.NAMES)], np.stack([ref[k] for k in FS.NAMES], 1), atol=2e-4), i
    # meta: article counts, the cold / unseen flags, category codes, exposure arrays, within-impression z-scores
    jm = int(np.flatnonzero(bld.group_of == "meta")[0])
    st = S.data.item_stats
    cand = val.cand(0, n)
    pos, shown = st["train_pos"][cand].astype(np.float64), st["train_shown"][cand].astype(np.float64)
    meta = X[:, jm:jm + len(ST.META_ITEM)]
    ctr = np.log((pos + 1.0) / (shown + 20.0))
    assert np.allclose(meta[:, 0], np.log1p(shown), atol=1e-5) and np.allclose(meta[:, 1], np.log1p(pos), atol=1e-5) and np.allclose(meta[:, 2], ctr, atol=1e-5)
    assert np.array_equal(meta[:, 3], (shown == 0).astype(np.float32)) and np.array_equal(meta[:, 4], (~S.seen_tr[cand]).astype(np.float32))
    for col, codes in ((5, cat), (6, sub)):                                      # pooled CTR of the category / sub-category over the training impressions
        gp = np.bincount(codes, weights=st["train_pos"].astype(np.float64), minlength=codes.max() + 1)
        gs = np.bincount(codes, weights=st["train_shown"].astype(np.float64), minlength=codes.max() + 1)
        assert np.allclose(meta[:, col], np.log((gp + 1.0) / (gs + 20.0))[codes[cand]], atol=1e-5), col
        assert len(np.unique(np.round(meta[:, col], 5))) > 1
    ex = np.concatenate(S.data.aux_validation)
    je = jm + len(ST.META_ITEM)
    assert np.array_equal(X[:, je:je + len(ST.META_EXPO)], ex)
    jr = je + len(ST.META_EXPO)
    rz = lambda v: common.ragged_zscore({"ids": np.arange(n), "ptr": val.ptr, "flat": np.asarray(v, np.float32)})["flat"]
    assert np.allclose(X[:, jr], rz(ctr), atol=1e-4)
    for q, col in ((0, 1), (1, 2), (3, 3)):
        assert np.allclose(X[:, jr + col], rz(ex[:, q]), atol=1e-4), (q, col)
    jp = int(np.flatnonzero(bld.group_of == "pop")[0])
    assert np.array_equal(X[:, jp], val.pop(0, n)) and np.allclose(X[:, jp + 1], rz(val.pop(0, n)), atol=1e-4)
    for grp, last in ((("scores", "ctx"), "ctx"), (("scores", "ctx", "meta"), "meta"), (("scores", "ctx", "meta", "pop"), "pop")):
        m = bld.prefix(grp)
        assert set(bld.group_of[:m]) == set(grp) and bld.group_of[m - 1] == last and (m == bld.n_features or bld.group_of[m] not in grp)
    # the test table reads no label: shuffled test labels give the identical table (and the labels helper reads the labels)
    rng = np.random.default_rng(3)
    rows2 = [(r[0], r[1], r[2], r[3], list(rng.permutation(np.asarray(r[4]))), r[5]) for r in S.data.test]
    assert any(list(a[4]) != list(b[4]) for a, b in zip(rows2, S.data.test))
    Xt = bld.build("test", test, 0, test.n)
    assert np.array_equal(Xt, bld.build("test", ST.Split(rows2, S.data.aux_test, S.data.aux_pop_test, S.data.time_test), 0, test.n)), "the test feature table depends on test labels"
    cold_t = Xt[:, jm + 3]                                                      # the fixture's test day shows articles no training row showed
    assert cold_t.min() == 0 and cold_t.max() == 1 and np.array_equal(Xt[:, jm + 4] >= cold_t, np.ones(len(cold_t), bool)), "never shown in a labelled impression implies unseen"
    cand_t = test.cand(0, test.n)                                               # test impressions: counts over train_core + the validation days, "unseen" = in neither
    ptv, stv = st["trainval_pos"][cand_t].astype(np.float64), st["trainval_shown"][cand_t].astype(np.float64)
    assert np.allclose(Xt[:, jm], np.log1p(stv), atol=1e-5) and np.allclose(Xt[:, jm + 1], np.log1p(ptv), atol=1e-5) and np.allclose(Xt[:, jm + 2], np.log((ptv + 1.0) / (stv + 20.0)), atol=1e-5)
    assert np.array_equal(Xt[:, jm + 3], (stv == 0).astype(np.float32)) and np.array_equal(Xt[:, jm + 4], (~(S.seen_tr | S.seen_va)[cand_t]).astype(np.float32))
    assert (Xt[:, jm] >= np.log1p(st["train_shown"][cand_t])).all() and (Xt[:, jm] > np.log1p(st["train_shown"][cand_t])).any(), "the test counts include the validation days"
    assert np.array_equal(test.labels(0, test.n), np.concatenate([np.asarray(r[4], np.float32) for r in S.data.test]))
    # a component exported for another candidate layout is an error, as in the supermodel; a subset is skipped
    bad = os.path.join(os.path.dirname(S.work), "badstack")
    shutil.rmtree(bad, ignore_errors=True)
    os.makedirs(os.path.join(bad, "scores"))
    short = {k: (v[:-1] if k == "flat" else v) for k, v in zc["val"].items()}
    short["ptr"] = np.concatenate([short["ptr"][:-1], [short["ptr"][-1] - 1]])
    common.save_scores(os.path.join(bad, "scores"), "pool_lse", short, zc["test"], {})
    raises(ValueError, ST.load_components, [bad], val, test)
    shutil.rmtree(bad, ignore_errors=True)
    os.makedirs(os.path.join(bad, "scores"))
    ids = np.arange(0, val.n, 2)
    part = common.make_ragged(ids, [zc["val"]["flat"][zc["val"]["ptr"][i]:zc["val"]["ptr"][i + 1]] for i in ids])
    common.save_scores(os.path.join(bad, "scores"), "pool_lse", part, zc["test"], {})
    assert ST.load_components([bad], val, test) == []
    shutil.rmtree(bad, ignore_errors=True)
    ok(f"(34) stacker.Builder: {bld.n_features} columns ({', '.join(f'{g}={int((bld.group_of == g).sum())}' for g in ST.GROUPS)}) equal an independent recomputation; "
       "chunked == whole; variants are prefixes; the test table is blind to test labels; other layouts raise, subsets are skipped")


# ------------------------------------------------------------------------------------------------ (35)
def test_folds():
    rng = np.random.default_rng(0)
    users = rng.integers(0, 500, 3000)
    f = ST.user_folds(users, 4, seed=0)
    assert set(f) == {0, 1, 2, 3} and np.array_equal(f, ST.user_folds(users, 4, seed=0)) and not np.array_equal(f, ST.user_folds(users, 4, seed=1))
    assert min(np.bincount(f)) > 0.15 * len(f), np.bincount(f)
    assert all(len(set(f[users == u])) == 1 for u in np.unique(users)), "a user is in exactly one fold"
    esm = ST.es_mask(users, 0)
    assert 0.03 < esm.mean() < 0.2 and all(len(set(esm[users == u])) == 1 for u in np.unique(users))
    usable = rng.random(len(users)) < 0.8
    inside = np.ones(len(users), bool)
    for k in range(4):
        tr, es, held = ST.fold_split(k, f, esm, usable, inside)
        assert not (set(users[tr]) & set(users[held])) and not (set(users[es]) & set(users[held])), "a user of the held-out fold is in the training part"
        assert not (set(tr) & set(es)) and set(tr) | set(es) == set(np.flatnonzero(usable & (f != k))) and np.array_equal(held, np.flatnonzero(f == k))
        assert not (set(users[tr]) & set(users[es])), "early-stopping users are disjoint from the training users"
    p = np.array([0, 3, 7, 7, 12])
    ln = np.diff(p)
    assert ST.expand(p, ln, [0, 2, 3]).tolist() == [0, 1, 2, 7, 8, 9, 10, 11] and ST.expand(p, ln, [1, 3]).tolist() == [3, 4, 5, 6, 7, 8, 9, 10, 11] and ST.expand(p, ln, []).tolist() == []
    v = (np.arange(12) ** 1.5).astype(np.float32)
    assert np.allclose(ST.make_seg_z(p)(v), common.ragged_zscore({"ids": np.arange(4), "ptr": p, "flat": v})["flat"], atol=1e-5)
    ok("(35a) folds: by user, deterministic, balanced, disjoint from the training / early-stopping users; expand / z-score helpers agree with common")


def _splits(data):
    """The validation / test ``Split`` of the stacker, as ``stage_stack`` builds them."""
    return (ST.Split(data.validation, data.aux_validation, data.aux_pop_validation, data.time_validation),
            ST.Split(data.test, data.aux_test, data.aux_pop_test, data.time_test))


def _stack_args(**kw):
    a = suite.build_parser().parse_args(["--mind-train", "x", "--mind-dev", "y", *STACK_ARGS])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_leakage_and_oof(tmp, enc_dir, cache_dir, run_args, S):
    # (b) a second complete run on a copy of the data whose TEST labels are shuffled inside every impression
    dev2 = os.path.join(tmp, "dev_shuffled")
    shutil.rmtree(dev2, ignore_errors=True)
    shutil.copytree(os.path.join(tmp, "dev"), dev2)
    rng = np.random.default_rng(5)
    lines = []
    with open(os.path.join(tmp, "dev", "behaviors.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            items = [x.rsplit("-", 1) for x in p[4].split()]
            labs = list(rng.permutation([b for _, b in items]))
            p[4] = " ".join(f"{a}-{b}" for (a, _), b in zip(items, labs))
            lines.append("\t".join(p))
    with open(os.path.join(dev2, "behaviors.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out2 = os.path.join(tmp, "outStk_shuffled")
    S2, _ = quiet(suite.main, ["--mind-train", f"{tmp}/train", "--mind-dev", dev2, "--work", out2, "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", STACK_STAGES,
                               *run_args, *STACK_ARGS])
    assert S2.kv["stage_status"]["stack"] == "done"
    a, b = np.load(os.path.join(S.work, "stack_scores.npz")), np.load(os.path.join(out2, "stack_scores.npz"))
    assert np.array_equal(a["val_ptr"], b["val_ptr"]) and np.array_equal(a["test_ptr"], b["test_ptr"])
    for v in ("stack_llm", "stack_meta"):
        assert np.array_equal(a[f"{v}__val_oof"], b[f"{v}__val_oof"]), f"{v}: the validation fit changed although only test labels differ"
        assert np.array_equal(a[f"{v}__test"], b[f"{v}__test"]), f"{v}: test scores depend on test labels"
    assert not np.array_equal(a["stack_pop__test"], b["stack_pop__test"]), "the streaming popularity is the one input that legitimately reads (earlier) test clicks"
    r1, r2 = S.row("stack_meta"), S2.row("stack_meta")
    assert abs(r1["ndcg@10"] - r2["ndcg@10"]) > 1e-6 and abs(r1["val_ndcg@10"] - r2["val_ndcg@10"]) < 1e-9, "metrics follow the labels, the validation side does not"
    # (c) out-of-fold scores of a fold do not depend on that fold's own labels (single objective, so that the objective scan cannot differ)
    args = _stack_args(stack_objectives="rank:ndcg", stack_configs="clean", super_dirs="")
    quiet(ST.stage_stack, S, args)
    ref = dict(np.load(os.path.join(S.work, "stack_scores.npz")))
    folds, ptr = ref["folds"], ref["val_ptr"]
    flipped = [(r[0], r[1], r[2], r[3], [1 - y for y in r[4]] if folds[i] == 1 else list(r[4]), r[5]) for i, r in enumerate(S.data.validation)]
    kept, S.data.validation = S.data.validation, flipped
    try:
        quiet(ST.stage_stack, S, args)
    finally:
        S.data.validation = kept
    new = np.load(os.path.join(S.work, "stack_scores.npz"))
    for v in ("stack_llm", "stack_meta"):
        same, diff = 0, 0
        for i in range(len(ptr) - 1):
            eq = np.array_equal(ref[f"{v}__val_oof"][ptr[i]:ptr[i + 1]], new[f"{v}__val_oof"][ptr[i]:ptr[i + 1]])
            if folds[i] == 1:
                assert eq, f"{v}: impression {i} of fold 1 is scored by a model that saw its labels"
                same += 1
            else:
                diff += int(not eq)
        assert same > 20 and diff > 20, (v, same, diff)
    ok("(35b) a whole second run with shuffled TEST labels gives identical validation fit and test scores for the offline variants (only stack_pop moves); "
       "(35c) out-of-fold scores of fold 1 are unchanged when only fold 1's labels are flipped (the other folds move)")


# ------------------------------------------------------------------------------------------------ (36)
def _planted(rng, n_groups=500, size=16):
    X = rng.standard_normal((n_groups * size, 5)).astype(np.float32)
    logit = 1.6 * X[:, 0] - 0.4 + 0.5 * X[:, 1] * (X[:, 2] > 0)
    y = (rng.random(len(X)) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    return X, y, np.full(n_groups, size)


def _ndcg_groups(scores, y, g, k=10):
    out, p = [], 0
    for n in g:
        out.append(metrics.ndcg_score(y[p:p + n], scores[p:p + n].astype(np.float32), k) if y[p:p + n].any() else 0.0)
        p += n
    return float(np.mean(out))


def test_backends():
    rng = np.random.default_rng(0)
    X, y, g = _planted(rng)
    a, b = 300 * 16, 400 * 16
    Xt, yt, gt, Xe, ye, ge, Xh, yh, gh = X[:a], y[:a], g[:300], X[a:b], y[a:b], g[300:400], X[b:], y[b:], g[400:]
    base = _ndcg_groups(rng.standard_normal(len(yh)), yh, gh)
    p = {"eta": 0.1, "depth": 4, "rounds": 80, "patience": 15, "seed": 0}
    results = {}
    for name, objs in (("xgboost", ("rank:ndcg", "rank:pairwise", "binary:logistic")), ("lightgbm", ("rank:ndcg", "rank:pairwise", "binary:logistic")), ("sklearn", ("binary:logistic",))):
        be = ST.make_backend(name, "cpu")
        assert be.name == name
        for o in objs:
            m = be.fit(Xt, yt, gt, Xe, ye, ge, o, p)
            s = be.predict(m, Xh)
            assert s.shape == (len(yh),) and s.dtype == np.float32 and np.isfinite(s).all() and 1 <= m[1] <= p["rounds"]
            results[(name, o)] = _ndcg_groups(s, yh, gh)
            assert results[(name, o)] > base + 0.05, (name, o, results[(name, o)], base)
            gain = be.gain(m, X.shape[1])
            assert gain.shape == (X.shape[1],)
            if name != "sklearn":
                assert gain.argmax() == 0 and gain.sum() > 0, (name, o, gain)
    # a GPU run that fails (driver, memory) falls back to the CPU histogram method and still trains; xgboost itself already switches to the CPU, with a
    # warning, when there is no GPU at all -- so the failure is simulated
    import types
    be = ST.XGBBackend("cuda")
    real = be.xgb
    calls = []

    def train(params, *a, **k):
        calls.append(params.get("device"))
        if params.get("device") == "cuda":
            raise RuntimeError("CUDA error: out of memory (simulated)")
        return real.train(params, *a, **k)
    be.xgb = types.SimpleNamespace(QuantileDMatrix=real.QuantileDMatrix, DMatrix=real.DMatrix, train=train)
    m, text = quiet(be.fit, Xt, yt, gt, Xe, ye, ge, "rank:ndcg", p)
    assert calls == ["cuda", None] and be.device == "cpu" and "failed" in text and 1 <= m[1] <= p["rounds"], (calls, be.device, text)
    be.xgb = types.SimpleNamespace(QuantileDMatrix=real.QuantileDMatrix, DMatrix=real.DMatrix, train=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    raises(RuntimeError, be.fit, Xt, yt, gt, Xe, ye, ge, "rank:ndcg", p)                    # a CPU failure is not swallowed
    # import failures fall through xgboost -> lightgbm -> scikit-learn -> an error that says so
    keys = ("xgboost", "lightgbm", "sklearn.ensemble")
    saved = {k: sys.modules.get(k) for k in keys}
    try:
        sys.modules["xgboost"] = None
        assert quiet(ST.make_backend, "auto", "cpu")[0].name == "lightgbm"
        sys.modules["lightgbm"] = None
        assert quiet(ST.make_backend, "auto", "cpu")[0].name == "sklearn"
        sys.modules["sklearn.ensemble"] = None
        e = raises(RuntimeError, quiet, ST.make_backend, "auto", "cpu")
        assert "no gradient-boosting backend" in str(e)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    ok("(36) backends learn a planted ranking signal (" + ", ".join(f"{k[0]}/{k[1]} {v:.3f}" for k, v in results.items()) + f" vs random {base:.3f}); "
       "GPU request falls back to CPU; import failures fall through xgboost -> lightgbm -> scikit-learn")


# ------------------------------------------------------------------------------------------------ (37)
def test_stack_stage(S, tmp, mind, enc_dir, cache_dir, run_args):
    st = S.kv["stack"]
    assert S.kv["stage_status"]["stack"] == "done" and st["backend"] in ("xgboost", "lightgbm", "sklearn") and st["folds"] == 4
    names = [r["name"] for r in S.records if r["family"].startswith("STACK")]
    assert names == ["stack_meta_v15cfg", "stack_llm", "stack_meta", "stack_pop"], names                 # the v15-recipe control comes first, then the three variants
    assert [S.row(n)["family"] for n in names] == ["STACK-control", "STACK", "STACK", "STACK+pop"] and [S.row(n)["ref"] for n in names] == ["frozen", "frozen", "frozen", "frozen+pop"]
    assert not any(r["family"].startswith("STACK") for r in S.reps + S.reps_pop), "stacker rows are not candidate directions"
    assert st["cv"] == "time" and len(st["members"]) == 2 and st["selection_metric"] and len(st["selection"]) == 10, (st["cv"], st["members"], len(st.get("selection", [])))
    z = np.load(os.path.join(S.work, "stack_scores.npz"))
    vptr, tptr = z["val_ptr"], z["test_ptr"]
    for v in names:
        row = S.row(v)
        for rows, flat, ptr, key, want in ((S.data.validation, z[f"{v}__val_oof"], vptr, "val", row["val_ndcg@10"]), (S.data.test, z[f"{v}__test"], tptr, "test", row["ndcg@10"])):
            got = metrics.aggregate([(np.array(r[4]), flat[ptr[i]:ptr[i + 1]]) for i, r in enumerate(rows)])
            assert abs(got["ndcg@10"] - want) < 2e-5, (v, key, got["ndcg@10"], want)
            if key == "test":
                assert abs(got["auc"] - row["auc"]) < 2e-5 and abs(got["mrr"] - row["mrr"]) < 2e-5
        assert z[f"{v}__val_have"].all()
        n_mem = len(st["variants"][v]["rounds"])                                  # one list of fold rounds per member (configuration) of the blend
        assert n_mem == (1 if v == "stack_meta_v15cfg" else len(st["members"])) and all(len(r) == 4 for r in st["variants"][v]["rounds"])
        zt = [common.ragged_zscore({"ids": np.arange(len(tptr) - 1), "ptr": tptr, "flat": z[f"{v}__test_m{mi}_fold{k}"]})["flat"] for mi in range(n_mem) for k in range(4)]
        assert np.allclose(np.mean(zt, 0), z[f"{v}__test"], atol=1e-5), "test score = mean over the fold models (of every member) of their within-impression z-scores"
        assert abs(st["variants"][v]["test_ndcg10"] - row["ndcg@10"]) < 1e-12
        assert abs(sum(st["variants"][v]["group_gain_share"].values()) - 1.0) < 1e-6 or st["backend"] == "sklearn"
    assert set(st["variants"]["stack_llm"]["group_gain_share"]) == {"scores", "ctx"} and set(st["variants"]["stack_pop"]["group_gain_share"]) == {"scores", "ctx", "meta", "pop"}
    assert {(r["config"], r["objective"]) for r in st["selection"]} == {(c, o) for c in ("v15", "clean", "clean_d5", "clean_d3", "noprior_d5") for o in ("rank:ndcg", "binary:logistic")}
    assert st["objective"] in ("rank:ndcg", "binary:logistic")
    dflt = suite.build_parser().parse_args(["--mind-train", "x", "--mind-dev", "y"])
    assert dflt.stack_objectives == "binary:logistic" and dflt.stack_cv == "time" and dflt.stack_blend == 2 and dflt.stack_control == 1, "v16 defaults: pointwise objective, time folds, blend of two"
    # no exported scores -> the stage declines and says so
    Sx, text = quiet(suite.main, [*mind, "--work", os.path.join(tmp, "outStk_noexp"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "controls,stack", "--export-scores", "0",
                                  "--lean-data", "1", "--no-resume", *run_args])
    assert Sx.kv["stage_status"]["stack"].startswith("declined") and "stacker not run" in text and not any(r["family"].startswith("STACK") for r in Sx.records)
    # the verbatim loader has no exposure features / article counts: the offline "meta" variant is left out, the others run
    Sv, text = quiet(suite.main, [*mind, "--work", os.path.join(tmp, "outStk_verbatim"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "controls,pooling,stack", "--stack-folds", "3",
                                  "--stack-rounds", "30", "--stack-objectives", "rank:ndcg", "--no-resume", *run_args])
    nv = [r["name"] for r in Sv.records if r["family"].startswith("STACK")]
    assert nv == ["stack_llm", "stack_pop"] and "--lean-data 1" in text and Sv.kv["stack"]["groups"] == ["scores", "ctx", "pop"], (nv, Sv.kv.get("stack"))
    assert Sv.kv["stack"]["cv"] == "time" and not Sv.kv["stack"]["windowed_ctr"], "the verbatim loader has neither impression times nor windowed click-through: the order of the rows stands for time"
    # the other backends, and a variant list
    for be in ("lightgbm", "sklearn"):
        free = be == "lightgbm"                                                  # one of the two runs also frees the training samples before the stacker builds its tables
        Sb, text = quiet(suite.main, [*mind, "--work", os.path.join(tmp, f"outStk_{be}"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "controls,pooling,stack", "--stack-folds", "3",
                                      "--stack-rounds", "30", "--stack-backend", be, "--stack-variants", "llm", "--stack-objectives", "rank:ndcg", "--stack-max-rows-cpu", "1200",
                                      "--no-resume", *(["--stack-free-memory", "1", "--train-sub", "300"] if free else []), *run_args])
        assert Sb.kv["stack"]["backend"] == be and [r["name"] for r in Sb.records if r["family"].startswith("STACK")] == ["stack_llm"]
        assert "CPU backend: training on" in text and "1,200 candidate rows at most" in text, "a CPU backend must cap its training rows"
        if free:
            import final_report
            assert Sb.data.train_core == [] and Sb.data.train_core_all is None and "freed the training samples" in text
            assert Sb.kv["train_samples"]["used"] == 300 and Sb.kv["train_samples"]["all"] > 1000, Sb.kv["train_samples"]
            assert f"training samples of the learned stages 300 of {Sb.kv['train_samples']['all']:,}" in final_report.build(Sb, None, None), "the report keeps the sizes after they were freed"
    ok("(37) stack stage: rows / families / references, validation row == out-of-fold recomputation (V10 metrics.py), test == mean of the fold models' z-scores, "
       "declines without exports, drops 'meta' on the verbatim loader, lightgbm and scikit-learn backends run")


# ------------------------------------------------------------------------------------------------ (37b)
def test_report(S, tmp):
    import final_report
    path, jpath = os.path.join(tmp, "FINAL_REPORT_stack.md"), os.path.join(tmp, "final_tables_stack.json")
    md = final_report.build(S, path, jpath)
    assert "failed:" not in md, md[md.index("failed:") - 300:md.index("failed:") + 300]
    assert "## 1b. The stacker" in md and "OUT-OF-FOLD" in md and "## 4. Published MIND-large numbers" not in md
    for n in ("stack_llm", "stack_meta", "stack_pop"):
        line = next(l for l in md.splitlines() if l.startswith(f"| `{n}` | {len(S.data.test):,} |"))
        cells = [c.strip() for c in line.strip("|").split("|")]
        r = S.row(n)
        assert abs(float(cells[2]) - r["val_ndcg@10"]) < 6e-5 and abs(float(cells[3]) - r["auc"]) < 6e-5 and abs(float(cells[6]) - r["ndcg@10"]) < 6e-5, (n, cells)
    sm = S.kv["supermodel"]
    plain = max((r for r in sm["rungs"] if not r["pop"] and "apex" not in r["name"]), key=lambda r: r["val_ndcg10_w"])["name"]
    popr = max((r for r in sm["rungs"] if r["pop"] and "apex" not in r["name"]), key=lambda r: r["val_ndcg10_w"])["name"]
    assert f"| `stack_meta` | `{plain}` | best supermodel rung on validation (without popularity) |" in md
    assert f"| `stack_pop` | `{popr}` | best supermodel rung on validation (with popularity) |" in md
    assert "| `stack_meta` | `stack_llm` | what the article counts and exposure add |" in md and "Share of the trees' total split gain" in md
    line = next(l for l in md.splitlines() if l.startswith("| `stack_meta` | `stack_llm` |"))
    d = suite.R.compare(S.cache["stack_meta"]["test"], S.cache["stack_llm"]["test"], "ndcg@10")
    assert f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" in line, (line, d)
    assert json.load(open(jpath))["stack"]["backend"] == S.kv["stack"]["backend"]
    keep = final_report.MIND_LARGE_DEV
    final_report.MIND_LARGE_DEV = len(S.data.test)
    try:
        md2 = final_report.build(S, None, None)
    finally:
        final_report.MIND_LARGE_DEV = keep
    assert "## 4. Published MIND-large numbers (secondary: from search results, the papers were not opened)" in md2 and "PLM-NR | 0.7064 | 0.4438" in md2 and "| `stack_meta` |" in md2.split("## 4.")[1]
    # a run without the stack stage prints no section for it
    assert final_report.stack_section(S) != [] and final_report.stack_section(_NoStack(S)) == []
    ok("(37b) final report: stacker section (rows == the run's rows, paired pairs against the validation-chosen supermodel rungs, gain shares), published numbers only for MINDlarge_dev and flagged secondary")


class _NoStack:
    """A suite whose stage list did not contain the stacker."""

    def __init__(self, S):
        self.kv = {k: v for k, v in S.kv.items() if k != "stack"}
        self.kv["stage_status"] = {k: v for k, v in S.kv["stage_status"].items() if k != "stack"}


# ------------------------------------------------------------------------------------------------ (38)
def test_prior_reader(S):
    E = S.E0
    m = UM.make_model("prior", E, dim=16, dropout=0.0)
    assert UM.is_reader("prior") and isinstance(m, UM.LLMEncPrior)
    H1, H2 = torch.randint(1, 50, (5, 7)), torch.randint(1, 50, (5, 7))
    cand = torch.randint(1, 50, (5, 4))
    mask = torch.ones(5, 7, dtype=torch.bool)
    m.eval()
    assert torch.equal(m.score(H1, mask, cand), m.score(H2, ~mask, cand)), "the prior ignores the history"
    # it learns what kind of article is clicked: topic 0 (every sixth article) against competitors of the other topics
    rng = np.random.default_rng(0)
    topic0 = np.array([i for i in range(1, 361) if i % 6 == 0])
    others = np.array([i for i in range(1, 361) if i % 6 != 0])

    def sample(N):
        return np.concatenate([rng.choice(topic0, N)[:, None], rng.choice(others, (N, 4))], 1)
    C = sample(1500)
    calls = []
    model, info = UM.train_user_model(E, (np.zeros((len(C), 5), np.int64), np.zeros((len(C), 5), bool), C), lambda mod: calls.append(1) or float(len(calls)), seed=0, dim=16,
                                      max_epochs=4, min_epochs=4, device="cpu", kind="prior", log=lambda *a: None)
    Ct = torch.from_numpy(sample(400))
    with torch.no_grad():
        s = model.score(torch.zeros(400, 1, dtype=torch.long), torch.zeros(400, 1, dtype=torch.bool), Ct)
    acc = float((s[:, :1] > s[:, 1:].max(1, keepdim=True).values).float().mean())
    assert info["epochs"] == 4 and acc > 0.6, f"the prior should rank the clicked topic first more often than chance (0.2): {acc}"
    ok(f"(38) prior reader: ignores the history, learns a click prior (clicked article first in {acc:.0%} of held-out slates of 5, chance 20%)")


def test_prior_export(S):
    assert common.load_scores(os.path.join(S.work, "scores", "um_prior_frozen.npz"))["meta"]["kind"] == "aux"
    assert common.load_scores(os.path.join(S.work, "scores", "um_frozen.npz"))["meta"]["kind"] == "reader"
    aux_names = {z for z in ("frozen", "um_prior_frozen", "pool_topk", "pool_maxsim", "knn_rag", "entity_rag", "community_graphrag", "txt_cta")}
    single = S.kv["supermodel"]["single_val"]
    assert single and not any(k.split(":", 1)[1] in aux_names for k in single), sorted(single)
    assert any(k.endswith(":um_frozen") for k in single) and any(k.endswith(":pool_lse") for k in single)
    ok("(38b) um_prior_* and the other auxiliary scorers are exported as kind 'aux' (stacker input) and never enter the supermodel's candidates")


# ------------------------------------------------------------------------------------------------ driver
def run_all(tmp, mind, enc_dir, cache_dir, run_args, S, mind_train, mind_dev):
    """Runs (32)-(38).  ``S``: the main selftest suite (its fixture is used for the loaders, featscore and the prior reader)."""
    test_exposure_and_stats(S, mind_train, mind_dev)
    test_featscore(S)
    test_prior_reader(S)
    test_folds()
    S_stack, _ = quiet(suite.main, [*mind, "--work", os.path.join(tmp, "outStk"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", STACK_STAGES, *run_args, *STACK_ARGS])
    assert S_stack.kv["stage_status"]["stack"] == "done" and S_stack.kv["stage_status"]["supermodel"] == "done", S_stack.kv["stage_status"]
    test_prior_export(S_stack)
    test_builder(S_stack)
    test_stack_stage(S_stack, tmp, mind, enc_dir, cache_dir, run_args)
    test_report(S_stack, tmp)
    test_leakage_and_oof(tmp, enc_dir, cache_dir, run_args, S_stack)
    test_backends()
    return S_stack
