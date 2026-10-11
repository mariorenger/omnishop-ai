"""Tests (23)-(31) of the v14 additions: exported component scores, the category-swap control, the supermodel, the final report,
the Hugging Face loader, the memory-lean MIND loader and the scale caps.  Called by ``selftest.main()`` after the runs it re-uses.

    (23)  ragged helpers + score export: files are the scores the rows were evaluated with
    (24)  the wrong-category control (``ctr``): swapped phrases, own families, never a candidate direction / reader view / export
    (25)  Fusion / fit_weights / ndcg_at_k unit tests
    (26)  supermodel stage: weights, base choice, independent recomputation from the exported files, tower subset, leave-one-group-out
    (27)  trained baselines in the hybrid rung, restricted by --super-baselines; the fit never reads the test split
    (28)  final report: sections, V10 checks, the static baseline table arithmetic
    (29)  hf_mind: fake Hub (zip, nested zip, extracted folders, no listing, unlabelled dev, traversal), reuse, HTTP resume
    (30)  data_lean == MindData.from_mind
    (31)  scale caps, train_core_all alignment, flags (stage status, --export-scores 0, --enc-variants, sample std)
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import http.server
import io
import json
import os
import re
import shutil
import socketserver
import threading
import zipfile

import numpy as np
import torch

import common
import data as D
import data_lean
import fields as FD
import final_report
import hf_mind
import metrics
import refdata as RD
import suite
import supermodel as SM


def ok(msg):
    print("OK", msg, flush=True)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


def zrow(v):
    v = np.asarray(v, np.float64)
    return (v - v.mean()) / (v.std() + 1e-6)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


# ------------------------------------------------------------------------------------------------ (23)
def test_ragged_and_export(tmp):
    rng = np.random.default_rng(0)
    lens = [1, 4, 7, 2]
    rows = [rng.standard_normal(n).astype(np.float32) for n in lens]
    rag = common.make_ragged([3, 5, 8, 9], rows)
    assert rag["ptr"].tolist() == [0, 1, 5, 12, 14] and rag["flat"].dtype == np.float32 and rag["ids"].dtype == np.int64
    assert all(np.array_equal(a, b) for a, b in zip(common.ragged_rows(rag), rows))
    z = common.ragged_zscore(rag)
    X, M = torch.zeros(4, 7), torch.zeros(4, 7, dtype=torch.bool)
    for i, r in enumerate(rows):
        X[i, :len(r)], M[i, :len(r)] = torch.from_numpy(r), True
    zt = common.zscore(X, M)
    for i, r in enumerate(common.ragged_rows(z)):
        assert np.allclose(r, zt[i, :lens[i]].numpy(), atol=1e-5), "ragged z-score differs from the torch recipe of common.zscore"
    assert common.ragged_mean_z([rag]) is rag
    rag_affine = common.make_ragged([3, 5, 8, 9], [r * 3 + 1 for r in rows])
    rag_neg = common.make_ragged([3, 5, 8, 9], [-r for r in rows])
    assert np.allclose(common.ragged_mean_z([rag, rag_affine])["flat"], z["flat"], atol=1e-4), "z-scores are invariant to a positive affine map"
    assert np.allclose(common.ragged_mean_z([rag, rag_neg])["flat"], 0, atol=1e-4)
    raises(ValueError, common.ragged_mean_z, [rag, common.make_ragged([3, 5, 8, 10], rows)])
    raises(ValueError, common.ragged_mean_z, [rag, common.make_ragged([3, 5, 8, 9], [r[:max(len(r) - 1, 1)] for r in rows])])
    for nm in ("lora+hn", "ut_causal@1", "txt_cta_lse", "odd name/with:chars"):
        p = common.save_scores(os.path.join(tmp, "sc"), nm, rag, rag_affine, {"kind": "zs"})
        got = common.load_scores(p)
        assert got["name"] == nm and got["meta"] == {"kind": "zs"} and os.path.dirname(p) == os.path.join(tmp, "sc")
        assert np.array_equal(got["val"]["flat"], rag["flat"]) and np.array_equal(got["test"]["ptr"], rag_affine["ptr"]) and np.array_equal(got["val"]["ids"], rag["ids"])
    assert not [f for f in os.listdir(os.path.join(tmp, "sc")) if ".tmp" in f]
    yes = ["pool_lse", "pool_km5", "lora+hn", "txt_t_lse", "txt_cta_lse", "txt_ctae_lse", "txt_views_lse", "hist100_lse", "hist200_lse", "ut_causal", "ut_causal@1"]
    no = ["frozen", "pool_topk", "txt_ctr_lse", "txt_ctr", "txt_cta", "ut_causal+meanpool", "um_frozen", "pool_km2", "lora+mrl+hn", "ut_bidir", "pool_lse+pop"]
    assert all(common.export_wanted(n) for n in yes) and not any(common.export_wanted(n) for n in no)
    ok("(23) ragged scores: z-score == common.zscore, seed ensemble = mean of z, id / candidate mismatches raise, npz round trip, export whitelist (controls excluded)")


def tower_members(S):
    return sorted(f[:-4] for f in os.listdir(os.path.join(S.work, "scores")) if f.startswith("ut_causal") and f.endswith(".npz"))


def test_exports(S):
    sd = os.path.join(S.work, "scores")
    comps = {}
    for f in sorted(os.listdir(sd)):
        z = common.load_scores(os.path.join(sd, f))
        comps[z["name"]] = z
    need = {"pool_lse", "txt_t_lse", "txt_cta_lse", "txt_ctae_lse", "txt_views_lse", "hist100_lse", "lora+hn", "um_frozen", "um_mi4_frozen"}
    assert need <= set(comps), sorted(need - set(comps))
    assert not any(n.startswith("txt_ctr") for n in comps) and "frozen" not in comps and not any(n.endswith("+pop") for n in comps)
    assert comps["um_frozen"]["meta"]["kind"] == "reader" and comps["um_mi4_frozen"]["meta"]["kind"] == "reader" and comps["pool_lse"]["meta"].get("kind", "zs") == "zs"
    checked = 0
    for n, z in comps.items():
        if z["meta"].get("kind") in ("reader", "baseline") or n not in S.cache:
            continue                                             # seed-ensembled z-scores are not the single row's scores
        for sp, rows in (("val", S.data.validation), ("test", S.data.test)):
            rag = z[sp]
            assert np.all(np.diff(rag["ids"]) > 0) and rag["ids"].max() < len(rows)
            sub = [rows[i] for i in rag["ids"]]
            assert np.array_equal(np.diff(rag["ptr"]), [len(r[3]) for r in sub]), (n, sp)
            got = metrics.aggregate([(np.array(r[4]), s) for r, s in zip(sub, common.ragged_rows(rag))])
            want = S.cache[n][sp]["mean"]
            assert abs(got["ndcg@10"] - want["ndcg@10"]) < 1e-5 and abs(got["auc"] - want["auc"]) < 1e-5 and abs(got["mrr"] - want["mrr"]) < 1e-5, (n, sp, got, want)
        checked += 1
    assert checked >= 6, checked
    # recording must not change what is evaluated
    sc = lambda: suite.R.mean_pool_scorer(S.tensor(S.E0))
    plain = S.run(sc(), "test")
    rec, rag = S.run_rec(sc(), "test")
    assert plain["mean"] == rec["mean"] and np.array_equal(plain["ids"], rec["ids"]) and np.array_equal(rag["ids"], rec["ids"])
    ok(f"(23b) exported raw scores reproduce the AUC / MRR / nDCG of the rows they belong to ({checked} components, validation and test, subsets included); recording changes nothing")


# ------------------------------------------------------------------------------------------------ (24)
def test_ctr_control(S):
    meta = S.meta
    own = [FD.category_phrase(c, s) for c, s in zip(meta["cats"], meta["subcats"])]
    wp = FD.wrong_category_phrases(meta)
    assert wp == FD.wrong_category_phrases(meta) and wp != FD.wrong_category_phrases(meta, seed=1) and len(wp) == len(own)
    owner = {}
    for c, ph in zip(meta["cats"], own):
        owner.setdefault(ph, set()).add(c)
    assert all(c not in owner[w] for c, w in zip(meta["cats"], wp)), "an article kept a phrase of its own category"
    cta, ctr = FD.build_texts(meta, "cta"), FD.build_texts(meta, "ctr")
    assert len(ctr) == len(cta) and all(t.endswith(x) for t, x in zip(ctr, meta["text"])) and all(a != b for a, b in zip(ctr, cta)) and "ctr" in FD.MODES and FD.CONTROL_MODES == ("ctr",)
    one = {"cats": ["a", "a", "a", "a"], "subcats": ["w", "x", "y", "z"]}
    assert sorted(FD.wrong_category_phrases(one)) == sorted(FD.category_phrase("a", s) for s in "wxyz"), "one category: a plain permutation of the phrases"
    names = {r["name"]: r for r in S.records}
    assert {"txt_ctr", "txt_ctr_lse"} <= set(names), "the control rows are missing (run A uses --text-modes ...,ctr)"
    assert names["txt_ctr"]["family"] == "F-control" and names["txt_ctr_lse"]["family"] == "F-control-lse"
    assert names["txt_ctr"]["ref"] == "frozen" and names["txt_ctr_lse"]["ref"] == "pool_lse"
    assert all(r["family"].replace("+pop", "") not in ("F-control", "F-control-lse") for r in S.reps + S.reps_pop), "a control entered the direction summary"
    assert S.kv["best_text"]["name"] not in ("txt_ctr_lse", "txt_ctr") and not os.path.exists(os.path.join(S.work, "scores", "txt_ctr_lse.npz"))
    ok("(24) wrong-category control: donors always come from another category, deterministic, own families, never the readers' text view, never exported, never a direction")


# ------------------------------------------------------------------------------------------------ (25)
def test_fusion_units(S):
    rows = S.data.validation
    n = len(rows)
    rng = np.random.default_rng(1)

    def comp(name, signal, kind="zs"):
        rr = [(signal * np.asarray(r[4], np.float64) + rng.standard_normal(len(r[3]))).astype(np.float32) for r in rows]
        rag = common.make_ragged(np.arange(n), rr)
        return SM.Comp("t", name, kind, rag, rag, {})
    A, B, Cn = comp("a", 1.2), comp("b", 1.0), comp("noise", 0.0)
    fu = SM.Fusion(rows, [A, B, Cn], "val", np.arange(n), "cpu", es=S.es["val"])
    assert fu.es is S.es["val"] and fu.K == 3 and fu.names == ["t:a", "t:b", "t:noise"]
    rr = {c.name: common.ragged_rows(c.val) for c in (A, B, Cn)}
    for w in ([1, 0, 0], [1, 0.5, 0.25], [0.3, 2, 1]):
        imps = []
        for i, r in enumerate(rows):
            s = sum(wi * zrow(rr[c.name][i]) for wi, c in zip(w, (A, B, Cn)))
            imps.append((np.array(r[4]), s))
        want = metrics.aggregate(imps)["ndcg@10"]
        assert abs(fu.ndcg10(w) - want) < 1e-5, (w, fu.ndcg10(w), want)
    # nDCG@10 helper == metrics.ndcg_score for short and long candidate lists
    labs = [np.array(r[4]) for r in rows[:40]]
    Cm = max(len(l) for l in labs)
    Y, Mk, Sc = torch.zeros(40, Cm), torch.zeros(40, Cm, dtype=torch.bool), torch.zeros(40, Cm)
    sc = [rng.standard_normal(len(l)) for l in labs]
    for i, (l, s) in enumerate(zip(labs, sc)):
        Y[i, :len(l)], Mk[i, :len(l)], Sc[i, :len(l)] = torch.from_numpy(l).float(), True, torch.from_numpy(s).float()
    got = SM.ndcg_at_k(Sc, Y, Mk).numpy()
    assert np.allclose(got, [metrics.ndcg_score(l, s, 10) for l, s in zip(labs, sc)], atol=1e-5)
    # exact ties (users with an empty history score 0 everywhere): the fit objective must equal what the evaluator reports (float32 rounding aside)
    tied = [np.zeros(len(r[3]), np.float32) if i % 4 == 0 else (1.0 * np.asarray(r[4], np.float64) + rng.standard_normal(len(r[3]))).astype(np.float32) for i, r in enumerate(rows)]
    rt = common.make_ragged(np.arange(n), tied)
    Ct = SM.Comp("t", "tied", "zs", rt, rt, {})
    ft_ = SM.Fusion(rows, [Ct], "val", np.arange(n), "cpu", es=S.es["val"])
    reported = S.run(ft_.scorer(np.array([1.0])), "val")["mean"]["ndcg@10"]
    assert abs(ft_.ndcg10([1.0]) - reported) < 1e-6, (ft_.ndcg10([1.0]), reported)       # float32 sums vs the evaluator's float64 mean
    imps = [(np.array(r[4]), zrow(common.ragged_rows(rt)[i]) if i % 4 else np.zeros(len(r[3]))) for i, r in enumerate(rows)]
    assert abs(metrics.aggregate(imps)["ndcg@10"] - reported) < 1e-6, "ties follow V10's metrics.py convention"
    # popularity as the last column
    fp = SM.Fusion(rows, [A], "val", np.arange(n), "cpu", with_pop=True, es=S.es["val"])
    imps = [(np.array(r[4]), zrow(rr["a"][i]) + 2.0 * zrow(np.asarray(r[5], np.float64))) for i, r in enumerate(rows)]
    assert fp.names == ["t:a", "pop"] and abs(fp.ndcg10([1.0, 2.0]) - metrics.aggregate(imps)["ndcg@10"]) < 1e-4
    # a subset of the impressions
    ids = np.arange(0, n, 3)
    fs = SM.Fusion(rows, [A, B], "val", ids, "cpu")
    imps = [(np.array(rows[i][4]), zrow(rr["a"][i]) + 0.5 * zrow(rr["b"][i])) for i in ids]
    assert abs(fs.ndcg10([1, 0.5]) - metrics.aggregate(imps)["ndcg@10"]) < 1e-5 and len(fs.es.rows) == len(ids)
    # coordinate ascent: never below the start, stays on the grid, only the free coordinates move, the informative second scorer helps, pure noise does not
    w, best = SM.fit_weights(fu, [1.0, 0.0, 0.0], free=[1, 2])
    assert w[0] == 1.0 and best >= fu.ndcg10([1, 0, 0]) - 1e-12 and abs(best - fu.ndcg10(w)) < 1e-12 and all(x in SM.GRID for x in w[1:])
    assert w[1] > 0 and best > fu.ndcg10([1, 0, 0]) + 1e-4, (w, best)
    w2, _ = SM.fit_weights(fu, [1.0, 0.7, 0.0], free=[2])
    assert w2[1] == 0.7 and w2[0] == 1.0, "coordinates outside `free` must not move"
    # broken inputs
    short = common.make_ragged(np.arange(n), [np.zeros(max(len(r[3]) - 1, 1), np.float32) for r in rows])
    raises(ValueError, SM.Fusion, rows, [SM.Comp("t", "short", "zs", short, short, {})], "val", np.arange(n), "cpu")
    part = common.make_ragged(np.arange(n - 5), [np.zeros(len(r[3]), np.float32) for r in rows[:n - 5]])
    raises(ValueError, SM.Fusion, rows, [SM.Comp("t", "part", "zs", part, part, {})], "val", np.arange(n), "cpu")
    unsorted = common.make_ragged(np.arange(n)[::-1].copy(), [np.zeros(len(r[3]), np.float32) for r in rows])
    raises(ValueError, SM.Comp, "t", "unsorted", "zs", unsorted, unsorted, {})
    # towers of one run merge into one component (mean of z); another run's tower stays separate
    t1, t2 = comp("ut_causal", 1.0, "tower"), comp("ut_causal@1", 1.0, "tower")
    merged = SM.merge_towers([A, t1, t2])
    mt = [c for c in merged if c.kind == "tower"]
    assert len(mt) == 1 and mt[0].name == "ut_causal[ensemble]" and mt[0].meta["members"] == ["ut_causal", "ut_causal@1"] and A in merged
    ok("(25) Fusion == independent metrics.py on the z-scored weighted sum (also +pop and subsets); ndcg_at_k == metrics.ndcg_score; fit_weights keeps the base, "
       "the grid and the free set; mismatching candidates / ids raise; tower seeds merge")


# ------------------------------------------------------------------------------------------------ (26)
def _load_component(S, name):
    return common.load_scores(os.path.join(S.work, "scores", common._safe(name) + ".npz"))


def recompute_rung(S, rung, split="test"):
    """The rung's nDCG@10 from the exported files alone: no Fusion, no supermodel code."""
    rows = S.data.test if split == "test" else S.data.validation
    parts = []
    for key, w in rung["weights"].items():
        if key == "pop":
            continue
        nm = key.split(":", 1)[1]
        if nm.endswith("[ensemble]"):
            members = [_load_component(S, m)[split] for m in tower_members(S)]
            ids = members[0]["ids"]
            zs = [[zrow(x) for x in common.ragged_rows(m)] for m in members]
            per = [zrow(np.mean([z[i] for z in zs], axis=0)) for i in range(len(ids))]
        else:
            rag = _load_component(S, nm)[split]
            ids, per = rag["ids"], [zrow(x) for x in common.ragged_rows(rag)]
        parts.append((w, dict(zip(ids.tolist(), per))))              # zero-weight components still define the impressions the rung is evaluated on
    keep = sorted(set.intersection(*[set(d) for _w, d in parts]))
    imps = []
    for i in keep:
        s = sum(w * d[i] for w, d in parts)
        if rung["weights"].get("pop"):
            s = s + rung["weights"]["pop"] * zrow(np.asarray(rows[i][5], np.float64))
        imps.append((np.array(rows[i][4]), s))
    return metrics.aggregate(imps)["ndcg@10"], len(keep)


def test_supermodel(S, expect_tower):
    sm = S.kv["supermodel"]
    assert S.kv["stage_status"]["supermodel"] == "done"
    names = [r["name"] for r in sm["rungs"]]
    g = sm["groups"]
    exp = (["super_base"] + (["super_zs"] if g["zs"] else []) + (["super_llm", "super_llm+pop"] if g["reader"] else [])
           + (["super_hybrid", "super_hybrid+pop"] if g["baseline"] else []) + (["super_apex", "super_apex+pop"] if g["tower"] else []))
    assert names == exp, (names, exp)
    assert g["reader"] and bool(g["tower"]) == expect_tower and not g["baseline"], g
    kinds = {}
    for f in os.listdir(os.path.join(S.work, "scores")):
        z = common.load_scores(os.path.join(S.work, "scores", f))
        kinds[f"{os.path.basename(S.work.rstrip('/'))}:{z['name']}"] = z["meta"].get("kind") or ("tower" if z["name"].startswith("ut_") else "zs")
    llm = {k: v for k, v in sm["single_val"].items() if kinds.get(k) in ("zs", "reader")}
    assert sm["base"] == max(llm, key=llm.get), "the base is the best LLM-side single model on VALIDATION"
    base = sm["base"]
    for r in sm["rungs"]:
        assert all(w >= 0 for w in r["weights"].values()) and r["weights"][base] == 1.0 and (r["weights"].get("pop", 1.0) >= 0)
        assert S.row(r["name"])["family"] == ("SUPER+pop" if r["pop"] else "SUPER") and abs(S.row(r["name"])["ndcg@10"] - r["test_ndcg10"]) < 1e-12
    plain = [r for r in sm["rungs"] if not r["pop"] and "apex" not in r["name"]]
    assert abs(plain[0]["val_ndcg10"] - sm["single_val"][base]) < 1e-5 and plain[0]["weights"] == {base: 1.0}
    assert all(b["val_ndcg10"] >= a["val_ndcg10"] - 1e-6 for a, b in zip(plain, plain[1:])), "validation must not fall along the ladder (warm start + coordinate ascent)"
    assert all(plain[0]["val_ndcg10"] >= v - 1e-6 for k, v in llm.items()), ("no LLM-side component beats the base on validation", plain[0], llm)
    for r in sm["rungs"]:
        want, n = recompute_rung(S, r)
        assert n == r["n_test"] and abs(want - r["test_ndcg10"]) < (1e-4 if r["pop"] else 1e-5), (r["name"], want, r["test_ndcg10"])
    if expect_tower:
        apex = next(r for r in sm["rungs"] if r["name"] == "super_apex")
        tw = _load_component(S, tower_members(S)[0])
        assert apex["n_test"] == len(tw["test"]["ids"]) < len(S.data.test) and apex["n_val"] == len(tw["val"]["ids"]) < len(S.data.validation)
        assert any(k.endswith("ut_causal[ensemble]") for k in apex["weights"]) and all(k in apex["weights"] for k in sm["groups"]["tower"])
        assert apex["weights"] and set(apex["weights"]) >= set(next(r for r in sm["rungs"] if r["name"] == "super_llm")["weights"]), "apex keeps the earlier components"
    assert sm["loo"] and all(np.isfinite(l["d_vs_full"]).all() for l in sm["loo"]) and {l["of"] for l in sm["loo"]} <= set(names)
    assert not any(r["family"] in ("SUPER", "SUPER+pop") for r in S.reps + S.reps_pop), "SUPER rows are not candidate directions"
    lb, _ = quiet(suite.leaderboard, [S], ["x"], ("x", "frozen"), None)
    assert any(r["name"] == "super_llm" for r in lb)
    return sm


def test_supermodel_paths(S, tmp):
    # no exported scores -> the stage declines (and says so), the run goes on
    keep = S.work
    S.work = os.path.join(tmp, "empty_run")
    os.makedirs(S.work, exist_ok=True)
    try:
        res, text = quiet(SM.stage_supermodel, S, argparse.Namespace(super_dirs="", super_baselines=""))
    finally:
        S.work = keep
    assert res is False and "supermodel not run" in text
    # a corrupted export (candidate counts of another split / cap) is refused loudly, not fused silently
    bad = os.path.join(tmp, "bad_run")
    z = common.load_scores(os.path.join(S.work, "scores", "pool_lse.npz"))
    z["val"]["ptr"] = np.concatenate([[0], np.cumsum(np.diff(z["val"]["ptr"]) + 1)])
    z["val"]["flat"] = np.zeros(int(z["val"]["ptr"][-1]), np.float32)
    common.save_scores(os.path.join(bad, "scores"), "pool_lse", z["val"], z["test"], {})
    comps = SM.load_components([bad])
    raises(ValueError, SM.Fusion, S.data.validation, comps, "val", np.arange(len(S.data.validation)), "cpu")
    ok("(26b) supermodel: no exports -> declined with a message; exports for other candidate counts -> ValueError")


# ------------------------------------------------------------------------------------------------ (27)
def test_hybrid(tmp, mind, enc_dir, cache_dir, run_args, ref_dir):
    """Trained baselines enter only the hybrid rung and only if --super-baselines allows them; the fit never evaluates the test split."""
    seen = []
    orig = SM.Fusion.ndcg10

    def guard(self, w):
        seen.append(self.split)
        assert self.split == "val", "the weight fit evaluated the test split"
        return orig(self, w)
    SM.Fusion.ndcg10 = guard
    try:
        out = {}
        for tag, allow in (("ff", "fastformer_ref"), ("two", "nrms_ref,naml_ref")):
            Sx, text = quiet(suite.main, [*mind, "--work", os.path.join(tmp, f"outH_{tag}"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "controls,pooling,supermodel",
                                         "--super-dirs", ref_dir, "--super-baselines", allow, *run_args])
            out[tag] = (Sx, text)
    finally:
        SM.Fusion.ndcg10 = orig
    tag_ref = os.path.basename(ref_dir.rstrip("/"))
    Sx = out["ff"][0]
    sm = Sx.kv["supermodel"]
    assert sm["groups"]["baseline"] == [f"{tag_ref}:fastformer_ref"], sm["groups"]
    names = [r["name"] for r in sm["rungs"]]
    assert "super_hybrid" in names and "super_hybrid+pop" in names, names
    hy = next(r for r in sm["rungs"] if r["name"] == "super_hybrid")
    assert f"{tag_ref}:fastformer_ref" in hy["weights"] and not any("nrms_ref" in k or "naml_ref" in k for r in sm["rungs"] for k in r["weights"])
    assert Sx.row("super_hybrid")["family"] == "SUPER" and "val" in seen and "test" not in seen
    sm2 = out["two"][0].kv["supermodel"]
    assert sorted(sm2["groups"]["baseline"]) == sorted([f"{tag_ref}:nrms_ref", f"{tag_ref}:naml_ref"]) and "fastformer" not in json.dumps(sm2["rungs"])
    # the "LLM" rungs never contain a baseline
    assert not any("_ref" in k for r in sm["rungs"] if r["name"] in ("super_base", "super_zs", "super_llm", "super_llm+pop") for k in r["weights"])
    ok("(27) hybrid rung: trained baselines only there, restricted by --super-baselines, never in the LLM-only rungs; the weight fit evaluates the validation split only")
    return Sx


def test_tower_math(tmp, mind, enc_dir, cache_dir, run_args):
    """(27b) a tower with real signal on a subset gets a positive weight, and the apex rung -- evaluated on the tower's subset only -- equals an
    independent recomputation from the exported files (the tower seeds are merged as mean of z, then z-scored again)."""
    work = os.path.join(tmp, "outTW")
    args = [*mind, "--work", work, "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "controls,pooling,usermodel,supermodel", "--no-resume", *run_args]
    S0, _ = quiet(suite.main, args)
    rng = np.random.default_rng(5)
    ids_v, ids_t = common.random_subset(len(S0.data.validation), 120, seed=3), common.random_subset(len(S0.data.test), 150, seed=4)

    def rag(rows, ids):
        return common.make_ragged(ids, [(2.0 * np.asarray(rows[i][4], np.float64) + rng.standard_normal(len(rows[i][3]))).astype(np.float32) for i in ids])
    for nm in ("ut_causal", "ut_causal@1"):
        common.save_scores(os.path.join(work, "scores"), nm, rag(S0.data.validation, ids_v), rag(S0.data.test, ids_t), {"kind": "tower"})
    S1, _ = quiet(suite.main, args)
    sm = S1.kv["supermodel"]
    assert sm["groups"]["tower"] == [f"{os.path.basename(work)}:ut_causal[ensemble]"], sm["groups"]
    for name in ("super_apex", "super_apex+pop"):
        r = next(x for x in sm["rungs"] if x["name"] == name)
        assert r["n_val"] == 120 and r["n_test"] == 150 and r["weights"][sm["groups"]["tower"][0]] > 0, r
        want, n = recompute_rung(S1, r)
        assert n == 150 and abs(want - r["test_ndcg10"]) < (1e-4 if r["pop"] else 1e-5), (name, want, r["test_ndcg10"])
    d = suite.R.compare(S1.cache["super_apex"]["test"], S1.cache["super_llm"]["test"], "ndcg@10")
    assert d[1] > 0 and len(S1.cache["super_apex"]["test"]["ids"]) == 150, d
    ok("(27b) tower with signal: positive weight, apex evaluated on the tower's 150 test impressions, equals the independent recomputation (also with +pop), paired gain over super_llm")


# ------------------------------------------------------------------------------------------------ (28)
def test_final_report(tmp, S, S_noswp, S_check):
    path, jpath = os.path.join(tmp, "FINAL_REPORT.md"), os.path.join(tmp, "final_tables.json")
    md = final_report.build(S, path, jpath, extra_suites=[S_check, None])
    assert open(path, encoding="utf-8").read() == md and json.load(open(jpath))["supermodel"]["base"] == S.kv["supermodel"]["base"]
    for head in ("# Final report", "## 0. Run health", "## 1. The supermodel and its ladder", "## 2. What each technique does", "## 3. Reference baselines"):
        assert head in md, head
    assert "failed:" not in md, md[md.index("failed:") - 200:md.index("failed:") + 200]
    assert "Not printed: the earlier NRMS / NAML / Fastformer numbers exist only for the MIND-small V10 split" in md
    assert "MISMATCH" in md and "do not trust any number below" in md and "protocol check with" in md
    assert "`super_base`" in md and "Leave-one-group-out" in md and "`txt_ctr_lse` is the control" in md
    assert all(f"`{r['name']}`" in md for r in S.kv["supermodel"]["rungs"])
    assert isinstance(common.mem_note(), str)
    if S.kv.get("memory"):
        assert "Memory after each stage" in md and all(f"`{k}`" in md for k in S.kv["memory"]) and set(S.kv["memory"]) <= set(S.kv["stage_status"])
    chk = json.load(open(jpath))["checks"]
    assert any(k.endswith("frozen == V10 bge_zeroshot") for k in chk), chk
    # the static baseline table: patch the split size so the section prints, then check the arithmetic against the rows
    keep = final_report.MIND_SMALL_TEST
    final_report.MIND_SMALL_TEST = len(S.data.test)
    try:
        md2 = final_report.build(S, None, None)
    finally:
        final_report.MIND_SMALL_TEST = keep
    assert "## 3. NRMS / NAML / Fastformer (measured earlier on this split, not trained here)" in md2 and "Not printed" not in md2
    r = S.row("super_llm")
    a = S.cache["super_llm"]["test"]["arr"]["ndcg@10"]
    half = 1.96 * np.sqrt(2) * np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))
    line = next(l for l in md2.splitlines() if l.startswith("| `super_llm` | NRMS, seed 0 (v7)"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    m = re.fullmatch(r"([+-]\d\.\d{4}) \[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]", cells[5])
    assert m, cells
    assert abs(float(m.group(1)) - (r["ndcg@10"] - 0.3937)) < 6e-5 and abs((float(m.group(3)) - float(m.group(2))) / 2 - half) < 1.5e-4
    assert abs(float(cells[2]) - (r["auc"] - 0.6392)) < 6e-5 and abs(float(cells[3]) - (r["mrr"] - 0.3037)) < 6e-5
    pop_line = next(l for l in md2.splitlines() if l.startswith("| `super_llm+pop` | Fastformer + pop (v13)"))
    mp = re.search(r"\| ([+-]\d\.\d{4}) \[", pop_line)
    assert mp and abs(float(mp.group(1)) - (S.row("super_llm+pop")["ndcg@10"] - 0.4381)) < 6e-5
    assert not any(l.startswith("| `super_llm+pop` | NRMS, seed") for l in md2.splitlines()), "content-only baselines are not compared with +pop rows"
    # a run without the supermodel still reports
    md3 = final_report.build(S_noswp, None, None)
    assert "(the supermodel stage did not run: see section 0)" in md3 and "## 2. What each technique does" in md3 and "failed:" not in md3
    ok("(28) final report: sections, V10 check rows (with the MISMATCH warning), static baseline arithmetic (difference of means, unpaired CI), +pop rows only vs +pop baselines, "
       "no-supermodel run still reports")


# ------------------------------------------------------------------------------------------------ (29)
def _write_fixture_zip(path, src_dir, prefix="", extra=None, drop=()):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in ("news.tsv", "behaviors.tsv"):
            if fn not in drop:
                z.write(os.path.join(src_dir, fn), prefix + fn)
        for name, data in (extra or {}).items():
            z.writestr(name, data)


class FakeHub:
    """Replaces the two network functions of hf_mind with a local folder of files."""

    def __init__(self, root, listing=True):
        self.root, self.listing, self.calls = root, listing, []

    def __enter__(self):
        self.saved = hf_mind.list_repo, hf_mind._hf_download, hf_mind._stream

        def lst(repo=hf_mind.REPO, token=None, log=print):
            if not self.listing:
                log("  cannot list (fake)")
                return []
            return sorted(os.path.relpath(os.path.join(dp, f), self.root) for dp, _d, fs in os.walk(self.root) for f in fs)

        def dl(repo, filename, local_dir, token=None):
            src = os.path.join(self.root, filename)
            if not os.path.isfile(src):
                raise FileNotFoundError(filename)
            self.calls.append(filename)
            dst = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            return dst

        def no_http(*a, **k):
            raise RuntimeError("no network in the self-test")
        hf_mind.list_repo, hf_mind._hf_download, hf_mind._stream = lst, dl, no_http
        return self

    def __exit__(self, *a):
        hf_mind.list_repo, hf_mind._hf_download, hf_mind._stream = self.saved


def _same_bytes(a, b):
    return open(a, "rb").read() == open(b, "rb").read()


def test_hf_mind(tmp, mind_train, mind_dev):
    log = lambda *a: None
    # --- local layout helpers
    root = os.path.join(tmp, "hf_local")
    for sub in ("dev_stuff/MINDlarge_train", "dev_stuff/MINDlarge_dev", "MINDsmall_train", "MINDsmall_dev", "partial/MINDlarge_train"):
        os.makedirs(os.path.join(root, sub))
        shutil.copyfile(os.path.join(mind_train, "news.tsv"), os.path.join(root, sub, "news.tsv"))
        shutil.copyfile(os.path.join(mind_train, "behaviors.tsv"), os.path.join(root, sub, "behaviors.tsv"))
    os.remove(os.path.join(root, "partial/MINDlarge_train/behaviors.tsv"))
    tr, dv = hf_mind.find_extracted(root, "large")
    assert tr == os.path.join(root, "dev_stuff/MINDlarge_train") and dv == os.path.join(root, "dev_stuff/MINDlarge_dev"), (tr, dv)
    assert hf_mind.find_extracted(root, "small") == (os.path.join(root, "MINDsmall_train"), os.path.join(root, "MINDsmall_dev"))
    assert hf_mind.find_extracted(os.path.join(root, "partial"), "large") == (None, None)
    assert not hf_mind.has_mind(os.path.join(root, "partial/MINDlarge_train")) and not hf_mind.has_mind(None)
    # --- validate
    raises(ValueError, hf_mind.validate, mind_train, mind_train, log)
    hf_mind.validate(mind_train, mind_dev, log)
    unl = os.path.join(tmp, "unlabelled")
    os.makedirs(unl)
    shutil.copyfile(os.path.join(mind_dev, "news.tsv"), os.path.join(unl, "news.tsv"))
    with open(os.path.join(mind_dev, "behaviors.tsv"), encoding="utf-8") as f, open(os.path.join(unl, "behaviors.tsv"), "w", encoding="utf-8") as g:
        for line in f:
            p = line.rstrip("\n").split("\t")
            g.write("\t".join(p[:4] + [" ".join(x.rsplit("-", 1)[0] for x in p[4].split())]) + "\n")
    e = raises(ValueError, hf_mind.validate, mind_train, unl, log)
    assert "unlabelled" in str(e)
    short = os.path.join(tmp, "short_cols")
    os.makedirs(short)
    shutil.copyfile(os.path.join(mind_dev, "news.tsv"), os.path.join(short, "news.tsv"))
    open(os.path.join(short, "behaviors.tsv"), "w").write("1\tU1\t11/15/2019 1:00:00 AM\n")
    raises(ValueError, hf_mind.validate, mind_train, short, log)

    # --- the Hub: standard layout (dev archive keeps its files in a folder), reuse, case-insensitive names
    hub = os.path.join(tmp, "hub_std")
    os.makedirs(hub)
    _write_fixture_zip(os.path.join(hub, "MINDlarge_train.zip"), mind_train, extra={"entity_embedding.vec": "Q1\t0.1\n"})
    _write_fixture_zip(os.path.join(hub, "MINDlarge_dev.zip"), mind_dev, prefix="MINDlarge_dev/", extra={"MINDlarge_dev/relation_embedding.vec": "P1\t0.1\n"})
    open(os.path.join(hub, "MINDlarge_test.zip"), "wb").write(b"garbage: never downloaded")
    _write_fixture_zip(os.path.join(hub, "MINDsmall_train.zip"), mind_train)
    dest = os.path.join(tmp, "dest_std")
    with FakeHub(hub) as fh:
        t, d = hf_mind.ensure_mind("large", dest, log=log)
        assert sorted(fh.calls) == ["MINDlarge_dev.zip", "MINDlarge_train.zip"], fh.calls
        assert _same_bytes(os.path.join(t, "news.tsv"), os.path.join(mind_train, "news.tsv")) and _same_bytes(os.path.join(d, "behaviors.tsv"), os.path.join(mind_dev, "behaviors.tsv"))
        assert sorted(os.listdir(t)) == ["behaviors.tsv", "news.tsv"] and not [f for f in os.listdir(dest) if f.endswith(".zip")], "only the two tsv files are kept, the archives are removed"
        fh.calls.clear()
        assert hf_mind.ensure_mind("large", dest, log=log) == (t, d) and fh.calls == [], "a second call must reuse the extracted files"
        # a Kaggle input that already holds extracted folders wins over a download
        fh.calls.clear()
        t2, d2 = hf_mind.ensure_mind("large", os.path.join(tmp, "dest_other"), extra_roots=[os.path.join(root, "dev_stuff")], log=log)
        assert fh.calls == [] and t2.endswith("MINDlarge_train") and d2.endswith("MINDlarge_dev")
    # --- archives that already sit in a (read-only) Kaggle-input-like folder: extracted without touching the Hub; a mix of local and Hub works too
    kin = os.path.join(tmp, "kaggle_input", "mind-large-zips")
    os.makedirs(kin)
    shutil.copyfile(os.path.join(hub, "MINDlarge_train.zip"), os.path.join(kin, "MINDlarge_train.zip"))
    shutil.copyfile(os.path.join(hub, "MINDlarge_dev.zip"), os.path.join(kin, "mindlarge_dev.zip"))
    with FakeHub(hub) as fh:
        listed = []
        orig_list = hf_mind.list_repo
        hf_mind.list_repo = lambda *a, **k: (listed.append(1), orig_list(*a, **k))[1]
        t3, d3 = hf_mind.ensure_mind("large", os.path.join(tmp, "dest_zipin"), extra_roots=[os.path.join(tmp, "kaggle_input")], log=log)
        assert fh.calls == [] and listed == [], "archives in the input folder must not trigger any Hub access"
        assert _same_bytes(os.path.join(t3, "news.tsv"), os.path.join(mind_train, "news.tsv")) and _same_bytes(os.path.join(d3, "behaviors.tsv"), os.path.join(mind_dev, "behaviors.tsv"))
        assert os.path.exists(os.path.join(kin, "MINDlarge_train.zip")) and os.path.exists(os.path.join(kin, "mindlarge_dev.zip")), "the input archives are never deleted"
    kin2 = os.path.join(tmp, "kaggle_input2")
    os.makedirs(kin2)
    shutil.copyfile(os.path.join(hub, "MINDlarge_train.zip"), os.path.join(kin2, "MINDlarge_train.zip"))
    with FakeHub(hub) as fh:
        hf_mind.ensure_mind("large", os.path.join(tmp, "dest_mixed"), extra_roots=[kin2], log=log)
        assert fh.calls == ["MINDlarge_dev.zip"], fh.calls
    hub_ci = os.path.join(tmp, "hub_ci")
    os.makedirs(os.path.join(hub_ci, "data"))
    shutil.copyfile(os.path.join(hub, "MINDlarge_train.zip"), os.path.join(hub_ci, "data", "mindlarge_train.zip"))
    shutil.copyfile(os.path.join(hub, "MINDlarge_dev.zip"), os.path.join(hub_ci, "data", "MINDLARGE_DEV.ZIP"))
    with FakeHub(hub_ci) as fh:
        hf_mind.ensure_mind("large", os.path.join(tmp, "dest_ci"), log=log)
        assert sorted(fh.calls) == ["data/MINDLARGE_DEV.ZIP", "data/mindlarge_train.zip"], fh.calls
    # --- the listing fails: the standard names are tried directly
    with FakeHub(hub, listing=False) as fh:
        hf_mind.ensure_mind("large", os.path.join(tmp, "dest_nolist"), log=log)
        assert sorted(fh.calls) == ["MINDlarge_dev.zip", "MINDlarge_train.zip"]
    # --- extracted folders in the repository instead of archives
    hub_x = os.path.join(tmp, "hub_x")
    for split, src in (("train", mind_train), ("dev", mind_dev)):
        os.makedirs(os.path.join(hub_x, f"MINDlarge_{split}"))
        for fn in ("news.tsv", "behaviors.tsv"):
            shutil.copyfile(os.path.join(src, fn), os.path.join(hub_x, f"MINDlarge_{split}", fn))
    with FakeHub(hub_x):
        t, d = hf_mind.ensure_mind("large", os.path.join(tmp, "dest_x"), log=log)
        assert _same_bytes(os.path.join(d, "news.tsv"), os.path.join(mind_dev, "news.tsv")) and _same_bytes(os.path.join(t, "behaviors.tsv"), os.path.join(mind_train, "behaviors.tsv"))
    # --- failures say what the repository holds
    hub_half = os.path.join(tmp, "hub_half")
    os.makedirs(hub_half)
    shutil.copyfile(os.path.join(hub, "MINDlarge_train.zip"), os.path.join(hub_half, "MINDlarge_train.zip"))
    with FakeHub(hub_half):
        e = raises(RuntimeError, hf_mind.ensure_mind, "large", os.path.join(tmp, "dest_half"), log=log)
        assert "['dev']" in str(e) and "MINDlarge_train.zip" in str(e) and "Add Input" in str(e)
    hub_unl = os.path.join(tmp, "hub_unl")
    os.makedirs(hub_unl)
    shutil.copyfile(os.path.join(hub, "MINDlarge_train.zip"), os.path.join(hub_unl, "MINDlarge_train.zip"))
    _write_fixture_zip(os.path.join(hub_unl, "MINDlarge_dev.zip"), unl)
    with FakeHub(hub_unl):
        e = raises(ValueError, hf_mind.ensure_mind, "large", os.path.join(tmp, "dest_unl"), log=log)
        assert "unlabelled" in str(e)
    raises(ValueError, hf_mind.ensure_mind, "huge", os.path.join(tmp, "dest_huge"), log=log)
    # --- an archive with the wrong content, and one with path-traversal member names
    hub_bad = os.path.join(tmp, "hub_bad")
    os.makedirs(hub_bad)
    _write_fixture_zip(os.path.join(hub_bad, "MINDlarge_train.zip"), mind_train, drop=("behaviors.tsv",))
    _write_fixture_zip(os.path.join(hub_bad, "MINDlarge_dev.zip"), mind_dev, prefix="../evil/")
    with FakeHub(hub_bad):
        e = raises(RuntimeError, hf_mind.ensure_mind, "large", os.path.join(tmp, "dest_bad", "d"), log=log)
        assert "['train']" in str(e)
        assert not any(os.path.exists(os.path.join(tmp, *p)) for p in (("evil",), ("dest_bad", "evil"), ("dest_bad", "d", "evil"))), "a ../ member name escaped the target folder"
        assert hf_mind.has_mind(os.path.join(tmp, "dest_bad", "d", "MINDlarge_dev")), "the traversal archive is read by file name only"
    # --- plain HTTP with resume (also against a server that ignores Range)
    payload = os.urandom(300_000)

    class H(http.server.BaseHTTPRequestHandler):
        honour, cut, cut_n, n = True, None, 1, 0              # `cut`: bytes sent before the connection is dropped, for the first `cut_n` requests

        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            H.n += 1
            start = int(rng.split("=")[1].split("-")[0]) if rng and H.honour else 0
            if start >= len(payload):
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = payload[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if H.cut and H.n <= H.cut_n:
                self.wfile.write(body[:H.cut])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(body)
    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    keep_env = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/f.zip"
        for honour in (True, False):
            H.honour, H.cut, H.cut_n, H.n = honour, 120_000, 1, 0                  # the first attempt is cut after 120 kB
            out = os.path.join(tmp, f"dl_{honour}.zip")
            lines = []
            hf_mind._stream(url, out, log=lines.append, retries=3, chunk=16_384)
            assert open(out, "rb").read() == payload and not os.path.exists(out + ".part"), honour
            assert any("attempt 1/3 failed" in l for l in lines), "the cut connection should have been retried"
        H.cut, H.honour = None, True
        out = os.path.join(tmp, "dl_clean.zip")
        hf_mind._stream(url, out, log=log)
        assert open(out, "rb").read() == payload
        # a connection that keeps dropping must end in an error, never in a silently truncated file
        H.cut, H.cut_n, H.n, H.honour = 50_000, 99, 0, True    # every attempt is cut, whatever it asks for
        out = os.path.join(tmp, "dl_never.zip")
        e = raises(RuntimeError, hf_mind._stream, url, out, log=lambda *a: None, retries=2, chunk=16_384)
        assert "could not download" in str(e) and not os.path.exists(out), "a truncated download must not be accepted"
        # a complete .part left by a crashed run (the server answers 416 to the resume request) is accepted
        H.cut, H.honour = None, True
        out = os.path.join(tmp, "dl_done.zip")
        open(out + ".part", "wb").write(payload)
        hf_mind._stream(url, out, log=log)
        assert open(out, "rb").read() == payload and not os.path.exists(out + ".part")
    finally:
        srv.shutdown()
        srv.server_close()
        for k, v in keep_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    ok("(29) hf_mind: zip / nested zip / extracted folders / case-insensitive / no listing, reuse and Kaggle-input precedence, only train+dev downloaded, unlabelled dev and "
       "missing parts fail with the repository listing, traversal names ignored, HTTP resume (Range honoured or not)")


# ------------------------------------------------------------------------------------------------ (30)
def test_lean_loader(S, mind_train, mind_dev):
    emb = dict(zip(S.meta["ids"], S.E0[1:]))
    a = D.MindData.from_mind(mind_train, mind_dev, llm_embeddings=emb)
    b = data_lean.from_mind_lean(mind_train, mind_dev, llm_embeddings=emb, log=lambda *x: None)
    for f in ("news_title", "news_cat", "llm_emb"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    for f in ("n_news", "n_users", "vocab_size", "n_cat", "title_len", "max_hist", "train_core_impressions", "validation_impressions", "test_impressions",
              "train_days", "validation_days", "test_days"):
        assert getattr(a, f) == getattr(b, f), f
    assert a.split_info == b.split_info and len(a._edges) > 0 and b._edges == [], "the lean loader leaves the (user, article) edge list empty"
    assert len(a.train_core) == len(b.train_core) > 1000
    for (u1, h1, p1, n1), (u2, h2, p2, n2) in zip(a.train_core, b.train_core):
        assert u1 == u2 and list(h1) == list(h2) and int(p1) == int(p2) and [int(x) for x in n1] == [int(x) for x in n2]
        assert all(type(x) is int for x in n2), "negatives are plain ints in the lean loader"
    for sa, sb in ((a.validation, b.validation), (a.test, b.test)):
        assert len(sa) == len(sb)
        for r1, r2 in zip(sa, sb):
            assert r1[0] == r2[0] and r1[1] == r2[1] and list(r1[2]) == list(r2[2]) and list(r1[3]) == list(r2[3]) and list(r1[4]) == list(r2[4])
            assert np.allclose(np.asarray(r1[5], np.float64), np.asarray(r2[5], np.float64), atol=1e-12, rtol=0)
    # the popularity feature itself, on a hand-made case: ties are scored before any of them is counted
    pop = data_lean.online_log_ctr_lean([[1, 2], [1, 2], [1, 3]], [[1, 0], [0, 1], [1, 0]], [5, 5, 9], 4)
    assert np.allclose(pop[0], np.log(np.array([1.0, 1.0]) / 20.0)) and np.allclose(pop[1], pop[0]) and np.allclose(pop[2][0], np.log(2.0 / 22.0)) and np.allclose(pop[2][1], np.log(1.0 / 20.0))
    ok("(30) data_lean.from_mind_lean == MindData.from_mind (articles, vocab, split, popularity, negatives, rows) and online popularity handles ties in time")


# ------------------------------------------------------------------------------------------------ (31)
def test_caps_and_flags(S, mind_train, tmp, mind, enc_dir, cache_dir, run_args):
    # --- scale caps
    args = argparse.Namespace(val_max=100, test_max=150, train_sub=500)
    d1, d2 = copy.copy(S.data), copy.copy(S.data)
    _, text = quiet(suite.cap_data, d1, args)
    suite.cap_data(d2, args)
    assert (len(d1.validation), len(d1.test), len(d1.train_core)) == (100, 150, 500) and "scale caps" in text
    for cap, full in ((d1.validation, S.data.validation), (d1.test, S.data.test), (d1.train_core, S.data.train_core)):
        pos = {id(r): i for i, r in enumerate(full)}
        idx = [pos[id(r)] for r in cap]
        assert idx == sorted(idx) and len(set(idx)) == len(idx), "a cap keeps original rows in their original order"
    assert [id(r) for r in d1.test] == [id(r) for r in d2.test] and [id(r) for r in d1.train_core] == [id(r) for r in d2.train_core], "caps are deterministic"
    assert d1.train_core_all is S.data.train_core and d1.validation_impressions == S.data.validation_impressions
    d0 = copy.copy(S.data)
    suite.cap_data(d0, argparse.Namespace(val_max=0, test_max=0, train_sub=0))
    assert d0.validation is S.data.validation and d0.test is S.data.test and getattr(d0, "train_core_all", None) is None
    core = RD.read_train_core(mind_train, S.meta["ids"], d1)
    ts = RD.TrainSet(core)
    RD.check_train_alignment(ts, d1)
    d3 = copy.copy(d1)
    d3.train_core_all = None
    raises(ValueError, RD.check_train_alignment, ts, d3)
    # --- the whole MIND-large path through suite.main: lean loader + caps + skipped user_rag, and the report says so
    Sl, text = quiet(suite.main, [*mind, "--work", os.path.join(tmp, "outL"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "controls,graph", "--lean-data", "1", "--val-max", "100",
                                  "--test-max", "150", "--train-sub", "500", "--skip-user-rag", "1", "--save-arrays", "0", *run_args])
    assert (len(Sl.data.validation), len(Sl.data.test), len(Sl.data.train_core), len(Sl.data.train_core_all)) == (100, 150, 500, 2511), "caps through suite.main"
    assert {"entity_rag", "knn_rag", "community_graphrag"} <= {r["name"] for r in Sl.records} and "user_rag" not in {r["name"] for r in Sl.records}
    assert not os.path.exists(os.path.join(tmp, "outL", "test_arrays.npz")) and "skipped (--save-arrays 0)" in text and Sl.kv["stage_status"]["graph"] == "done"
    md = final_report.build(Sl, None, None)
    assert "Scale settings: memory-lean loader" in md and "validation 100 of 260" in md and "test 150 of 330" in md and "training samples of the learned stages 500 of 2,511" in md and "`user_rag` skipped" in md
    # --- stage status, --export-scores 0, a run that cannot supermodel
    cdir = os.path.join(tmp, "outX")
    Sx, text = quiet(suite.main, [*mind, "--work", cdir, "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "controls,pooling,supermodel", "--export-scores", "0", *run_args])
    st = Sx.kv["stage_status"]
    assert st["controls"] == "done" and st["pooling"] == "done" and st["supermodel"].startswith("declined") and not os.path.exists(os.path.join(cdir, "scores"))
    assert "supermodel" not in Sx.kv.get("timings", {}) and "supermodel not run" in text
    md = final_report.build(Sx, None, None)
    assert "not done" in md and "declined" in md and "(the supermodel stage did not run: see section 0)" in md
    # --- --enc-variants: unknown value fails the stage with a clear message, a known one runs only that variant and exports it
    Sy, text = quiet(suite.main, [*mind, "--work", os.path.join(tmp, "outY"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "controls,encoder", "--enc-variants", "bogus", *run_args])
    assert Sy.kv["stage_status"]["encoder"].startswith("FAILED") and "--enc-variants: choose from" in text
    Sz, text = quiet(suite.main, [*mind, "--work", os.path.join(tmp, "outZ"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "controls,encoder", "--enc-variants", "lora+hn", *run_args])
    zn = {r["name"] for r in Sz.records}
    assert "lora+hn" in zn and "lora+mrl+hn" not in zn and "dora+mrl+hn" not in zn and os.path.exists(os.path.join(tmp, "outZ", "scores", "lora+hn.npz"))
    # --- the sample (n-1) standard deviation of seed rows
    n_checked = 0
    for r in S.records:
        v = r.get("ndcg10_per_seed")
        if v and len(v) > 1:
            assert abs(r["ndcg10_std"] - float(np.std(v, ddof=1))) < 1e-12, r["name"]
            n_checked += 1
    assert n_checked >= 3, n_checked
    ok(f"(31) scale caps (deterministic, ordered, train_core_all keeps V10 alignment), stage status (declined / FAILED / done), --export-scores 0, --enc-variants, sample std on {n_checked} seed rows")
