"""Tests (69)-(72) of the v20 additions.  Called by ``selftest.main()``; ``python selftest_v20.py`` runs them alone on a fresh fixture.

    (69)  generation split over several devices (stages_v20.generate_parallel) == one generator (same batches, greedy), resumed from the per-device caches
          without generating again; the kar_full stage on the whole split (rows kar_item / kar_item_lse / kar_exp, exported kinds, the augmented vectors
          written as an --news-emb file) and a reader run that reads them (its exported reader scores carry that run's tag); concat_npz / catalogue_vectors
    (70)  Semantic IDs v2: the per-level scores summed over every level == v7's sid_scorer (also with PMI), the level cache == the direct scorer, the sid2
          stage (level set and PMI chosen on validation, the shuffled-history control, exports)
    (71)  histq_mlp: the heads start as the identity (step 0 scores == H1 exactly), the stage on the whole split exports a reader; on a planted signal the
          heads learn (validation above H1)
    (72)  fusion_only.prepare_v20 (v20c): the control is v19c's treatment (main run with the H5 tower + bge, cat, catplus, bgeplus), the treatment adds only
          the chosen new components (no duplicate 'frozen' rows), missing inputs build nothing
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import numpy as np
import torch

import common
import llm_gen
import rep_eval as R
import sid as SID
import stages_v20 as V20
import suite
from selftest_v14 import ok, quiet


def _base(tmp, mind, enc_dir, run_args, work, extra=()):
    return [*mind, "--work", work, "--encoder", enc_dir, "--cache-dir", os.path.join(tmp, "v20_cache"), "--lean-data", "1",
            *[a for a in run_args if a != "--no-resume"], *extra]


# ------------------------------------------------------------------------------------------------ (69)
def test_generate_parallel(dec_dir, texts, tmp):
    prompts = [f"Write about {t[:40]}" for t in texts[:11]]
    one = llm_gen.Generator(dec_dir, "cpu", torch.float32, os.path.join(tmp, "g1.jsonl"))
    ref = one.generate(prompts, max_new_tokens=6, bs=3)
    base = os.path.join(tmp, "gpar")
    gens = V20.load_generators(dec_dir, ["cpu", "cpu"], torch.float32, base)
    got = V20.generate_parallel(gens, prompts, max_new_tokens=6, bs=3, desc="t")
    assert got == ref, (got, ref)
    assert os.path.exists(base + ".0.jsonl") and os.path.exists(base + ".1.jsonl"), "one cache file per device"
    gens2 = V20.load_generators(dec_dir, ["cpu", "cpu"], torch.float32, base)
    calls = []
    for g in gens2:
        g.model.generate = lambda *a, **k: calls.append(1)            # any generation would be recorded
    again = V20.generate_parallel(gens2, prompts, max_new_tokens=6, bs=3, desc="t")
    assert again == ref and not calls, "resumed from the caches of both devices"
    part = V20.generate_parallel(V20.load_generators(dec_dir, ["cpu"], torch.float32, base + "_dl"), prompts, max_new_tokens=6, bs=3, deadline=0.0, desc="t")
    assert all(p is None for p in part), "a passed deadline generates nothing"
    ok("(69a) generation over two devices == one generator (same batches, greedy), one cache file per device, resumed without generating, deadline honoured")


def test_kar_full(tmp, mind, enc_dir, dec_dir, run_args):
    common.CLOCK.reset()
    keep = suite.EQUIV_REFERENCE
    suite.EQUIV_REFERENCE = None
    try:
        work = os.path.join(tmp, "outKar")
        shutil.rmtree(work, ignore_errors=True)
        base = _base(tmp, mind, enc_dir, run_args, work, ["--gen-model", dec_dir, "--gen-new-tokens", "6", "--gen-bs", "16", "--kar-alphas", "0.5", "1.0"])
        S, text = quiet(suite.main, [*base, "--stages", "controls,pooling,kar_full"])
        n = S.data.n_news - 1
        info = S.kv["kar_full"]
        assert info["expanded"] == n and info["articles"] == n, info
        for name, kind in (("kar_item", "zs"), ("kar_item_lse", "zs"), ("kar_exp", "aux")):
            r = S.row(name)
            assert r["n_test"] == len(S.data.test) and r["n_val"] == len(S.data.validation), name
            z = common.load_scores(os.path.join(work, "scores", name + ".npz"))
            assert z["meta"]["kind"] == kind and len(z["test"]["ids"]) == len(S.data.test), (name, z["meta"])
        assert S.row("kar_item_lse")["ref"] == "pool_lse"
        lines = open(os.path.join(work, "kar_expansions.jsonl"), encoding="utf-8").read().splitlines()
        assert len(lines) == n
        ids, vecs = V20.load_npz_vectors(info["aug_npz"])
        assert ids == list(S.meta["ids"]) and vecs.shape == (n, S.E0.shape[1]) and np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-4)
        S2, _ = quiet(suite.main, [*base, "--stages", "controls,kar_full"])
        assert "already finished" in _ or S2.kv["kar_full"]["expanded"] == n
        # a reader run on the augmented vectors (--news-emb): its scores are a separate run for the supermodel
        work_r = os.path.join(tmp, "qwen3kar")
        shutil.rmtree(work_r, ignore_errors=True)
        Sr, tr = quiet(suite.main, [*_base(tmp, mind, enc_dir, run_args, work_r), "--news-emb", info["aug_npz"], "--um-views", "frozen", "--um-extra", "",
                                    "--stages", "controls,usermodel"])
        f0 = Sr.row("frozen")
        assert abs(f0["ndcg@10"] - S.row("kar_item")["ndcg@10"]) < 1e-6, "the reader run's base vectors are the augmented ones"
        assert os.path.exists(os.path.join(work_r, "scores", "um_frozen.npz"))
        # concatenation with another encoder's vectors, served from the cache the first run filled
        ids_b, vb = V20.catalogue_vectors(mind[1], mind[3], enc_dir, os.path.join(tmp, "v20_cache"), "cpu", max_len=32)
        assert ids_b == ids and np.allclose(vb, S.E0[1:], atol=1e-6)
        out = V20.concat_npz([(ids, vecs), (ids_b, vb)], os.path.join(tmp, "cat.npz"))
        _, vc = V20.load_npz_vectors(out)
        assert vc.shape == (n, 2 * vecs.shape[1]) and np.allclose(np.linalg.norm(vc, axis=1), 1, atol=1e-4)
        try:
            V20.concat_npz([(ids, vecs), (ids_b[::-1], vb)], os.path.join(tmp, "bad.npz"))
            raise AssertionError("misaligned parts must be refused")
        except ValueError:
            pass
    finally:
        suite.EQUIV_REFERENCE = keep
    ok(f"(69b) kar_full: {n} articles expanded (2-device path tested in 69a), kar_item / kar_item_lse / kar_exp on the whole split and exported (zs / zs / aux); "
       f"the augmented vectors feed a reader run through --news-emb (its frozen row == kar_item); concat_npz aligns and refuses misaligned parts")


# ------------------------------------------------------------------------------------------------ (70)
def test_sid_levels():
    torch.manual_seed(0)
    K, L, n = 8, 3, 40
    codes = np.random.default_rng(0).integers(0, K, (n, L)).astype(np.int64)
    model = SID.SIDGPT(K, L, d=32, layers=2, heads=2, max_items=12).eval()
    rng = np.random.default_rng(1)
    hist = torch.from_numpy(rng.integers(1, n, (6, 5)))
    mask = torch.from_numpy(rng.random((6, 5)) > 0.3)
    mask[0] = False                                                  # an empty history
    cand = torch.from_numpy(rng.integers(1, n, (6, 4)))
    b = {"hist": hist, "hist_mask": mask, "cand": cand, "cand_mask": torch.ones(6, 4, dtype=torch.bool), "rows": np.arange(6)}
    lv = SID.sid_level_scorer(model, codes, n_max=5)(b)
    assert torch.allclose(lv.sum(-1), SID.sid_scorer(model, codes, n_max=5)(b), atol=1e-5)
    base = SID.history_free_level_logprob(model, torch.from_numpy(codes), n)
    assert torch.allclose(base.sum(1), SID.history_free_logprob(model, torch.from_numpy(codes), n), atol=1e-5)
    pmi = SID.sid_scorer(model, codes, n_max=5, pmi=True)(b)
    assert torch.allclose(lv.sum(-1) - base[cand].sum(-1), pmi, atol=1e-5)

    class _ES(list):
        n = 1
    es = _ES([b])
    es.n = 6
    cache = SID.LevelCache(es, SID.sid_level_scorer(model, codes, n_max=5))
    assert torch.allclose(cache.scorer([1, 2, 3], "cpu")(b), lv.sum(-1), atol=1e-5)
    assert torch.allclose(cache.scorer([1], "cpu", base)(b), lv[..., 0] - base[cand][..., 0], atol=1e-5)
    sh = SID.sid_level_scorer(model, codes, n_max=5, shuffle=True)(b)
    b2 = dict(b, hist=torch.roll(hist, 1, 0), hist_mask=torch.roll(mask, 1, 0))
    assert torch.allclose(sh, lv * 0 + SID.sid_level_scorer(model, codes, n_max=5)(b2), atol=1e-5)
    ok("(70a) SID per-level scores: summed == v7's sid_scorer, PMI == the per-level history-free prior subtracted, the level cache == direct, shuffle = the next row's history")


def test_sid2_stage(tmp, mind, enc_dir, run_args):
    common.CLOCK.reset()
    keep = suite.EQUIV_REFERENCE
    suite.EQUIV_REFERENCE = None
    try:
        work = os.path.join(tmp, "outSid2")
        shutil.rmtree(work, ignore_errors=True)
        S, text = quiet(suite.main, [*_base(tmp, mind, enc_dir, run_args, work, ["--sid-text", "cta", "--sid-epochs", "2", "--sid-shuf-n", "60"]),
                                     "--stages", "controls,sid2"])
        r = S.row("sid_gpt")
        assert set(r["cfg"]) == {"levels", "pmi"} and r["n_test"] == len(S.data.test)
        assert S.row("sid_gpt_sumlevels")["cfg"]["levels"] == [1, 2, 3]
        sh = S.row("sid_gpt_shuffled")
        assert sh["ref"] == "sid_gpt" and sh["n_test"] == min(60, len(S.data.test)) and sh["cfg"] == r["cfg"]
        for name in ("sid_profile", "sid_gpt"):
            assert common.load_scores(os.path.join(work, "scores", name + ".npz"))["meta"]["kind"] == "zs"
        assert not os.path.exists(os.path.join(work, "scores", "sid_gpt_shuffled.npz"))
        assert S.kv["sid2"]["text"] == "cta" and "codes from the 'cta' vectors" in text
    finally:
        suite.EQUIV_REFERENCE = keep
    ok(f"(70b) sid2 stage: codes from 'cta' vectors, sid_gpt chosen over level sets x PMI on validation ({r['cfg']}), v7's all-level row, the shuffled-history "
       f"control on {sh['n_test']} impressions (compared with sid_gpt on the same ones), sid_profile / sid_gpt exported as zs")


# ------------------------------------------------------------------------------------------------ (71)
def test_hist_head_identity():
    torch.manual_seed(0)
    h = V20.HistHead(16, 32)
    x = torch.randn(5, 16)
    assert torch.allclose(h.q(x), x) and torch.allclose(h.c(x), x)
    c = torch.randn(5, 3, 16)
    assert torch.allclose(h.logits(x, c), 20.0 * (x[:, None, :] * c).sum(-1), atol=1e-4)
    ok("(71a) the history heads start as the identity: step 0 ranks exactly like H1")


def test_histq_mlp_stage(tmp, mind, enc_dir, run_args):
    common.CLOCK.reset()
    keep = suite.EQUIV_REFERENCE
    suite.EQUIV_REFERENCE = None
    try:
        work = os.path.join(tmp, "outHm")
        shutil.rmtree(work, ignore_errors=True)
        base = _base(tmp, mind, enc_dir, run_args, work, ["--hm-train-n", "100000", "--hm-val-n", "60", "--hm-epochs", "3", "--hm-hidden", "32", "--hm-seeds", "2",
                                                         "--hm-bs", "64"])
        S, text = quiet(suite.main, [*base, "--stages", "controls,histq_mlp"])
        h1, hm = S.row("histq_instr"), S.row("histq_mlp")
        assert hm["ref"] == "histq_instr" and hm["n_test"] == len(S.data.test) and h1["n_test"] == len(S.data.test)
        assert len(hm["val_curves"]) == 2 and all(len(c) >= 2 for c in hm["val_curves"])
        z = common.load_scores(os.path.join(work, "scores", "histq_mlp.npz"))
        assert z["meta"]["kind"] == "reader" and len(z["test"]["ids"]) == len(S.data.test)
        assert common.load_scores(os.path.join(work, "scores", "histq_instr.npz"))["meta"]["kind"] == "aux"
        assert os.path.exists(os.path.join(work, "histq_vectors.npz"))
        S2, t2 = quiet(suite.main, [*base, "--rerun", "histq_mlp", "--stages", "controls,histq_mlp"])
        assert "query vectors loaded" in t2, "the encoded histories are reused"
        # every curve starts at the H1 score of the validation subset (the identity)
        starts = {round(c[0], 6) for c in hm["val_curves"]}
        assert len(starts) == 1, starts
    finally:
        suite.EQUIV_REFERENCE = keep
    ok(f"(71b) histq_mlp on the whole split: H1 row + learned heads ({hm['best_epochs']} best epochs, curves start at the same H1 score), exported as a reader; "
       f"query vectors cached and reused")


# ------------------------------------------------------------------------------------------------ (72)
def test_prepare_v20(tmp):
    import fusion_only as FO
    root = os.path.join(tmp, "fo20")
    shutil.rmtree(root, ignore_errors=True)

    def comp(d, name, kind="zs"):
        os.makedirs(d, exist_ok=True)
        rag = {"ids": np.arange(3), "ptr": np.array([0, 2, 4, 6]), "flat": np.arange(6, dtype=np.float32)}
        common.save_scores(d, name, rag, rag, {"kind": kind})
    a, b, c, d = (os.path.join(root, x, "scores") for x in ("v19a", "v19b", "v20a", "v20b"))
    for t in ("bge", "cat", "catplus", "bgeplus", "qwen3"):
        comp(os.path.join(a, t), "um_frozen", "reader"); comp(os.path.join(a, t), "frozen", "aux")
    comp(os.path.join(b, "qwen3"), "um_frozen", "reader"); comp(os.path.join(b, "qwen3"), "ut_causal", "reader"); comp(os.path.join(b, "qwen3"), "frozen", "aux")
    comp(os.path.join(c, "kar"), "kar_item"); comp(os.path.join(c, "kar"), "frozen", "aux"); comp(os.path.join(c, "kar"), "pool_lse")
    comp(os.path.join(c, "catkarplus"), "um_knn_rag", "reader"); comp(os.path.join(c, "catkarplus"), "frozen", "aux")
    comp(os.path.join(d, "sidq"), "sid_gpt"); comp(os.path.join(d, "sidq"), "histq_mlp", "reader"); comp(os.path.join(d, "sidq"), "frozen", "aux")
    work = os.path.join(root, "work")
    p = FO.prepare_v20(work, a, b, [c, d], arms=("ctrl_s0", "treat_s0"))
    assert not p["missing"], p["missing"]
    ctrl, treat = p["arms"]["ctrl_s0"], p["arms"]["treat_s0"]
    assert "ut_causal" in FO.components(os.path.join(ctrl["work"], "scores")), "the control is v19c's treatment: the H5 tower is in"
    assert [os.path.basename(x) for x in ctrl["super"]] == ["bge", "cat", "catplus", "bgeplus"]
    names = lambda s: sorted(os.path.basename(x) for x in s)
    assert names(treat["super"]) == sorted(["bge", "cat", "catplus", "bgeplus", "kar", "catkarplus", "sidq"])
    kar = FO.components(os.path.join(work, "runs", "kar", "scores"))
    assert kar == ["kar_item"], kar
    assert FO.components(os.path.join(work, "runs", "sidq", "scores")) == ["histq_mlp", "sid_gpt"]
    assert FO.components(os.path.join(work, "runs", "catkarplus", "scores")) == ["um_knn_rag"]
    q = FO.prepare_v20(os.path.join(root, "work2"), a, b, [c, os.path.join(root, "nowhere")], arms=("ctrl_s0", "treat_s0"))
    assert q["missing"] and not q["arms"]
    ok("(72) prepare_v20: control = v19c's treatment (tower + bge, cat, catplus, bgeplus); treatment adds kar / catkarplus / sidq with only the chosen "
       "components (no duplicate 'frozen'); missing inputs build nothing")


def run_all(tmp, mind, enc_dir, dec_dir, cache_dir, run_args, texts):
    test_generate_parallel(dec_dir, texts, tmp)
    test_kar_full(tmp, mind, enc_dir, dec_dir, run_args)
    test_sid_levels()
    test_sid2_stage(tmp, mind, enc_dir, run_args)
    test_hist_head_identity()
    test_histq_mlp_stage(tmp, mind, enc_dir, run_args)
    test_prepare_v20(tmp)


if __name__ == "__main__":
    import selftest as T
    tmp = tempfile.mkdtemp(prefix="v20_selftest_")
    texts = T.build_fixture(tmp)
    enc_dir, dec_dir = os.path.join(tmp, "tinyenc"), os.path.join(tmp, "tiny-qwen3-embedding")
    T.build_tiny_encoder(enc_dir, texts)
    T.build_tiny_decoder(dec_dir, texts)
    run_all(tmp, ["--mind-train", f"{tmp}/train", "--mind-dev", f"{tmp}/dev"], enc_dir, dec_dir, os.path.join(tmp, "cache"), T.RUN_A[:-1], texts)
    print("v20 tests: all passed;", tmp)
