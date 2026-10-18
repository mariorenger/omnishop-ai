"""Tests (55)-(62) of the v18 additions.  Called by ``selftest.main()``.

    (55)  adapt.params_b / resolve_dtype: the size read from a model name, fp16 weights only for >= 3B parameters on a GPU, fp32 on a CPU; keep_fp32_residual casts
          the embeddings and norms of a half-precision model back to fp32 and leaves the linear layers in fp16
    (56)  adapt.encode_multi (two encoders in two threads, batch k on device k % 2) == HFEncoder.encode, bit for bit
    (57)  fields.embed_texts: a few non-finite rows are encoded again on the CPU in fp32 (the rest untouched, the cache holds the repaired table); many of them stop the run
    (58)  --base-text cta: the run's own vectors read fields.build_texts(meta, 'cta'), the pooling guard sees the same texts, a 'concat:' run on cta is served from the
          caches the single runs' moredata stages filled (nothing encoded), and the moredata stage declines in a run whose own text is not 'ta'
    (59)  --rerun supermodel,stack: after another run added components (--super-dirs), only those two stages run again, without duplicated rows, and the supermodel
          sees the new components; the old suite object keeps its own rows for a paired comparison
    (60)  reader 'sca': at initialisation it scores exactly like an LLMEncCA with the same head; the self-attention block learns (its output projection moves); an empty
          history gives finite scores; it trains inside the usermodel stage as an --um-extra reader
    (61)  learned readers: --um-lr-decay multiplies the learning rate after every epoch from --um-decay-after on, --um-min-delta sets the improvement threshold, the best
          epoch is reported; the defaults reproduce the v16b recipe exactly
    (62)  report_v18: the control / treatment table and the gates (PASS / NO GAIN / WORSE / NOT RUN) on planted per-impression results; the v16b table; epochs table
"""
from __future__ import annotations

import os
import shutil
import types

import numpy as np
import torch
import torch.nn as nn

import adapt
import common
import fields as FD
import report_v18 as RV
import suite
import usermodel as UM
from selftest_v14 import ok, quiet, raises


# ------------------------------------------------------------------------------------------------ (55)
def test_dtype_rules(enc_dir):
    assert adapt.params_b("Qwen/Qwen3-Embedding-4B") == 4.0 and adapt.params_b("Qwen/Qwen3-Embedding-0.6B") == 0.6
    assert adapt.params_b("BAAI/bge-large-en-v1.5") == 0.0 and adapt.params_b("intfloat/e5-mistral-7b-instruct") == 7.0 and adapt.params_b("Qwen/Qwen3-1.7B") == 1.7
    assert adapt.params_b("/tmp/x/tinyenc") == 0.0
    f16, f32 = torch.float16, torch.float32
    assert adapt.resolve_dtype("Qwen/Qwen3-Embedding-4B", "auto", "cuda") == f16 and adapt.resolve_dtype("Qwen/Qwen3-Embedding-4B", "auto", "cuda:1") == f16
    assert adapt.resolve_dtype("Qwen/Qwen3-Embedding-0.6B", "auto", "cuda") == f32 and adapt.resolve_dtype("BAAI/bge-large-en-v1.5", "auto", "cuda") == f32
    assert adapt.resolve_dtype("Qwen/Qwen3-Embedding-4B", "auto", "cpu") == f32 and adapt.resolve_dtype("x", "fp16", "cpu") == f32, "no fp16 matmuls on a CPU"
    assert adapt.resolve_dtype("x", "fp16", "cuda") == f16 and adapt.resolve_dtype("Qwen/Qwen3-Embedding-4B", "fp32", "cuda") == f32
    raises(ValueError, adapt.resolve_dtype, "x", "bf16", "cuda")
    # keep_fp32_residual on a half-precision copy of the tiny BERT
    from transformers import AutoModel
    m = AutoModel.from_pretrained(enc_dir).half()
    n = adapt.keep_fp32_residual(m)
    emb = [mod for mod in m.modules() if isinstance(mod, nn.Embedding)]
    norms = [mod for mod in m.modules() if "norm" in type(mod).__name__.lower()]
    lin = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert n == len(emb) + len(norms) and emb and norms and lin
    assert all(p.dtype == torch.float32 for mod in emb + norms for p in mod.parameters())
    assert all(mod.weight.dtype == torch.float16 for mod in lin), "the linear layers stay fp16"
    ok(f"(55) params_b / resolve_dtype: fp16 weights only for >= 3B on a GPU (fp32 on a CPU, an unknown dtype is an error); keep_fp32_residual cast {len(emb)} embedding "
       f"and {len(norms)} norm modules back to fp32, {len(lin)} linear layers stay fp16")


# ------------------------------------------------------------------------------------------------ (56)
def test_encode_multi(enc_dir, texts):
    texts = list(texts[:150])
    one = adapt.HFEncoder(enc_dir, max_len=32, device="cpu").encode(texts, bs=16)
    two = adapt.encode_multi(enc_dir, texts, 32, "", torch.float32, 16, ["cpu", "cpu"])
    assert two.shape == one.shape and np.array_equal(one, two), float(np.abs(one - two).max())
    three = adapt.encode_multi(enc_dir, texts, 32, "", torch.float32, 16, ["cpu", "cpu", "cpu"])
    assert np.array_equal(one, three)
    # an error in a worker thread reaches the caller
    bad = list(texts)
    bad[40] = None                                                     # the tokenizer fails on it
    raises(Exception, adapt.encode_multi, enc_dir, bad, 32, "", torch.float32, 16, ["cpu", "cpu"])
    # embed_texts takes this path when several GPUs are visible (simulated: the device list is patched)
    keep = FD.gpu_devices
    FD.gpu_devices = lambda device: ["cpu", "cpu"]
    try:
        cdir = os.path.join(os.path.dirname(enc_dir), "multi_cache")
        shutil.rmtree(cdir, ignore_errors=True)
        (v, s), text = quiet(FD.embed_texts, texts, enc_dir, 32, "", "cpu", 16, cdir)
    finally:
        FD.gpu_devices = keep
    assert np.array_equal(v, one) and "on 2 GPUs" in text and s > 0
    assert FD.gpu_devices("cpu") == ["cpu"]
    # out of GPU memory on a batch: it is halved until it fits, and the vectors are those of the smaller batches
    enc = adapt.HFEncoder(enc_dir, max_len=32, device="cpu")
    orig, calls = enc.embed, []

    def tight(tx):
        calls.append(len(tx))
        if len(tx) > 5:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return orig(tx)
    enc.embed = tight
    got = adapt.embed_batch(enc, texts[:16])
    want = torch.cat([orig(texts[i:i + 4]) for i in range(0, 16, 4)])
    assert torch.allclose(got, want, atol=1e-6) and calls[0] == 16 and max(c for c in calls if c <= 5) == 4
    ok("(56) encode_multi over 2 and 3 devices == HFEncoder.encode bit for bit (same batches); a worker's error reaches the caller; embed_texts uses it when several GPUs are "
       "visible; a batch that runs out of GPU memory is halved until it fits (16 -> 8 -> 4)")


# ------------------------------------------------------------------------------------------------ (57)
def test_nonfinite_repair(enc_dir, texts):
    texts = list(texts[:120])
    cdir = os.path.join(os.path.dirname(enc_dir), "nan_cache")
    shutil.rmtree(cdir, ignore_errors=True)
    clean = adapt.HFEncoder(enc_dir, max_len=32, device="cpu").encode(texts, bs=32)
    orig = adapt.HFEncoder.encode
    state = {"calls": 0}

    def broken(self, tx, bs=128, rows=(3, 77)):
        state["calls"] += 1
        v = orig(self, tx, bs=bs)
        if state["calls"] == 1:                                        # the first (GPU-like) pass overflows on two rows; the CPU fp32 pass is clean
            v[list(rows)] = np.nan
        return v
    adapt.HFEncoder.encode = broken
    try:
        (v, _), text = quiet(FD.embed_texts, texts, enc_dir, 32, "", "cpu", 32, cdir)
    finally:
        adapt.HFEncoder.encode = orig
    assert np.isfinite(v).all() and np.allclose(v, clean, atol=1e-6) and "2 non-finite embeddings" in text and state["calls"] == 2
    z = np.load([os.path.join(cdir, f) for f in os.listdir(cdir)][0])
    assert np.isfinite(z["vecs"]).all(), "the cache holds the repaired table"
    shutil.rmtree(cdir, ignore_errors=True)

    def many(self, tx, bs=128):
        v = orig(self, tx, bs=bs)
        v[:70] = np.inf
        return v
    adapt.HFEncoder.encode = many
    try:
        e = raises(RuntimeError, FD.embed_texts, texts, enc_dir, 32, "", "cpu", 32, cdir)
    finally:
        adapt.HFEncoder.encode = orig
    assert "non-finite" in str(e) and not os.path.exists(cdir) or not os.listdir(cdir), "nothing cached"
    ok("(57) embed_texts: 2 non-finite rows re-encoded on the CPU in fp32 (== the clean vectors, the cache holds the repaired table); 70 of 120 stop the run, nothing cached")


# ------------------------------------------------------------------------------------------------ (58)
def test_base_text(tmp, mind, enc_dir, dec_dir, cache_dir, run_args):
    import argparse
    import stages_ext as X
    cdir = os.path.join(tmp, "bt_cache")
    shutil.rmtree(cdir, ignore_errors=True)
    base = [*mind, "--cache-dir", cdir, "--lean-data", "1", *[a for a in run_args if a != "--no-resume"], "--no-resume", "--text-modes", "cta"]
    seen = []
    keep_ref = suite.EQUIV_REFERENCE
    ref_enc = adapt.HFEncoder(enc_dir, max_len=32, device="cpu")
    suite.EQUIV_REFERENCE = lambda t: (seen.append(list(t)), ref_enc.encode(t))[1]
    try:
        # single runs on 'ta' whose moredata stage embeds 'cta' (fills the cache the composite run reads)
        Sa, _ = quiet(suite.main, [*base, "--work", os.path.join(tmp, "outBTa"), "--encoder", enc_dir, "--stages", "controls,pooling,moredata"])
        meta = Sa.meta
        cta = FD.build_texts(meta, "cta")
        assert seen[-1] == list(meta["text"][:64]), "a 'ta' run guards on 'title. abstract'"
        # a run whose own text is 'cta'
        Sb, tb = quiet(suite.main, [*base, "--work", os.path.join(tmp, "outBTb"), "--encoder", enc_dir, "--base-text", "cta", "--stages", "controls,pooling,moredata"])
        assert seen[-1] == cta[:64], "the guard compares on the texts the vectors read"
    finally:
        suite.EQUIV_REFERENCE = keep_ref
    want, secs = X.embed_news(meta, enc_dir, 32, "", "cpu", 64, cdir, texts=cta)
    assert secs == 0, "cta of this encoder was cached by run a's moredata stage"
    assert np.allclose(Sb.E0[1:], want, atol=1e-6) and not np.allclose(Sa.E0[1:], want, atol=1e-3), "the run's own vectors are the cta vectors (the loader re-normalises: float noise only)"
    assert "E0 loaded from cache" in tb
    assert "not run: the run's own vectors already read the text variant 'cta'" in tb and Sb.kv["stage_status"]["moredata"].startswith("declined")
    assert abs(Sb.row("frozen")["ndcg@10"] - Sa.row("txt_cta")["ndcg@10"]) < 1e-9, "frozen of the cta run == txt_cta of the ta run"
    # the second encoder's 'cta' through its own moredata stage, then the composite on 'cta' encodes nothing
    suite.EQUIV_REFERENCE = None
    try:
        quiet(suite.main, [*base, "--work", os.path.join(tmp, "outBTc"), "--encoder", dec_dir, "--stages", "controls,pooling,moredata"])
    finally:
        suite.EQUIV_REFERENCE = keep_ref
    n_files = len(os.listdir(cdir))
    a = argparse.Namespace(encoder=f"concat:{enc_dir}+{dec_dir}", max_len=32, doc_prefix="", enc_bs=64, base_text="cta", enc_dtype="auto")
    V, secs = suite.embed_catalogue(meta, a, "cpu", cdir)
    assert secs == 0 and len(os.listdir(cdir)) == n_files, "served from the single runs' caches"
    B_, _ = X.embed_news(meta, dec_dir, 32, "", "cpu", 64, cdir, texts=cta)
    nrm = lambda v: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    assert np.allclose(V[:, :want.shape[1]], nrm(want) / np.sqrt(2), atol=1e-6) and np.allclose(V[:, want.shape[1]:], nrm(B_) / np.sqrt(2), atol=1e-6)
    a_ta = argparse.Namespace(encoder=f"concat:{enc_dir}+{dec_dir}", max_len=32, doc_prefix="", enc_bs=64)
    V_ta, _ = suite.embed_catalogue(meta, a_ta, "cpu", cdir)
    assert not np.allclose(V_ta, V), "'ta' and 'cta' composites differ"
    raises(SystemExit, suite.main, [*base, "--work", os.path.join(tmp, "outBTx"), "--encoder", enc_dir, "--base-text", "views", "--stages", "controls"])
    ok("(58) --base-text cta: E0 == the cta vectors (from the moredata cache of a 'ta' run), frozen == txt_cta of that run, the guard reads the same texts, moredata declines; "
       "a cta composite encodes nothing and differs from the ta one; an unknown text is rejected")
    return cdir


# ------------------------------------------------------------------------------------------------ (59)
def test_rerun(tmp, mind, enc_dir, dec_dir, cache_dir, run_args):
    keep_ref = suite.EQUIV_REFERENCE
    suite.EQUIV_REFERENCE = None
    try:
        common_ = [*mind, "--cache-dir", cache_dir, "--lean-data", "1", *[a for a in run_args if a != "--no-resume"], "--stack-folds", "3", "--stack-rounds", "30",
                   "--stack-configs", "clean,clean_d5", "--stack-control", "0", "--stack-chunk", "90"]
        dA, dB = os.path.join(tmp, "outRrA"), os.path.join(tmp, "outRrB")
        for d, enc in ((dA, dec_dir), (dB, f"concat:{enc_dir}+{dec_dir}")):
            shutil.rmtree(d, ignore_errors=True)
            quiet(suite.main, [*common_, "--work", d, "--encoder", enc, "--stages", "controls,pooling", "--no-resume"])
        work = os.path.join(tmp, "outRrMain")
        shutil.rmtree(work, ignore_errors=True)
        main_args = [*common_, "--work", work, "--encoder", enc_dir]
        S1, t1 = quiet(suite.main, [*main_args, "--super-dirs", dA, "--stages", "controls,pooling,supermodel,stack"])
        assert S1.kv["stage_status"]["supermodel"] == "done" and S1.kv["stage_status"]["stack"] == "done", t1[-800:]
        rows1 = sorted(r["name"] for r in S1.records)
        comps1 = set(S1.kv["supermodel"]["single_val"])
        base_val = {n: S1.row(n)["ndcg@10"] for n in ("frozen", "pool_lse")}
        stack1 = S1.row("stack_llm")["ndcg@10"]
        S1.release()
        # unchanged: nothing runs again
        S2, t2 = quiet(suite.main, [*main_args, "--super-dirs", dA, "--stages", "controls,pooling,supermodel,stack"])
        assert "== S supermodel" not in t2 and "== T stacker" not in t2
        # another run added components: --rerun supermodel,stack redoes only those
        S3, t3 = quiet(suite.main, [*main_args, "--super-dirs", f"{dA},{dB}", "--rerun", "supermodel,stack", "--stages", "controls,pooling,supermodel,stack"])
        assert "--rerun: running" in t3 and "== S supermodel" in t3 and "== T stacker" in t3 and "== H8 pooling" not in t3 and "stage 'pooling' already finished" in t3, t3[-800:]
        assert sorted(r["name"] for r in S3.records) == rows1, "the same rows, once each"
        comps3 = set(S3.kv["supermodel"]["single_val"])
        assert comps1 < comps3 and any(k.startswith("outRrB:") for k in comps3), (comps1, comps3)
        assert {n: S3.row(n)["ndcg@10"] for n in base_val} == base_val, "the other stages' rows are the saved ones"
        assert S1.row("stack_llm")["ndcg@10"] == stack1 and "stack_llm" in S1.cache, "the released control keeps its own rows"
        d = RV.paired(S3, "stack_llm", S1, "stack_llm")
        assert d is not None and np.isfinite(d[0])
        # --rerun is not part of the saved state's signature
        assert "rerun" in suite.Suite.SIG_IGNORED
    finally:
        suite.EQUIV_REFERENCE = keep_ref
    ok(f"(59) --rerun supermodel,stack after a run added components: only those two stages run again (same rows once each, other rows restored), the supermodel sees "
       f"{len(comps3) - len(comps1)} new component(s); the released control keeps its rows (paired Δ stack_llm {d[0]:+.4f})")


# ------------------------------------------------------------------------------------------------ (60)
def test_sca_reader(tmp, mind, enc_dir, cache_dir, run_args):
    torch.manual_seed(0)
    n, d = 60, 24
    E = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
    E[0] = 0
    sca = UM.make_model("sca", E, dim=16, dropout=0.0)
    ca = UM.LLMEncCA(E, 16, 0.0)
    ca.head.load_state_dict(sca.head.state_dict())
    ca.cand_proj.load_state_dict(sca.cand_proj.state_dict())
    sca.eval(), ca.eval()
    rng = np.random.default_rng(1)
    hist = torch.from_numpy(rng.integers(1, n, (8, 10)))
    mask = torch.from_numpy(rng.random((8, 10)) < 0.7)
    mask[2] = False                                                    # a reader without history
    hist[~mask] = 0
    cand = torch.from_numpy(rng.integers(1, n, (8, 5)))
    s1, s2 = sca.score(hist, mask, cand), ca.score(hist, mask, cand)
    assert torch.isfinite(s1).all() and torch.allclose(s1, s2, atol=1e-6), float((s1 - s2).abs().max())
    assert UM.is_reader("sca") and isinstance(sca, UM.LLMEncSCA)
    # one optimisation step moves the zero-initialised output projection, and then the scores differ from the plain reader
    opt = torch.optim.Adam(sca.parameters(), lr=1e-2)
    sca.train()
    loss = torch.nn.functional.cross_entropy(sca.score(hist, mask, cand), torch.zeros(8, dtype=torch.long))
    loss.backward()
    assert sca.sa.o.weight.grad is not None and float(sca.sa.o.weight.grad.abs().sum()) > 0
    opt.step()
    sca.eval()
    with torch.no_grad():
        assert float(sca.sa.o.weight.abs().sum()) > 0 and not torch.allclose(sca.score(hist, mask, cand), s2, atol=1e-6)
    # inside the usermodel stage as an --um-extra reader
    keep_ref = suite.EQUIV_REFERENCE
    suite.EQUIV_REFERENCE = None
    try:
        S, text = quiet(suite.main, [*mind, "--work", os.path.join(tmp, "outSCA"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--lean-data", "1",
                                     *[a for a in run_args if a != "--no-resume"], "--no-resume", "--stages", "controls,graph,usermodel", "--um-views", "frozen,knn_rag",
                                     "--um-extra", "sca:knn_rag,sca:frozen", "--um-seeds", "1", "--um-seeds-focus", "1", "--um-epochs", "3", "--um-dim", "16"])
    finally:
        suite.EQUIV_REFERENCE = keep_ref
    names = {r["name"]: r for r in S.records}
    assert {"um_sca_knn_rag", "um_sca_frozen", "um_knn_rag", "um_frozen"} <= set(names), sorted(names)
    assert names["um_sca_knn_rag"]["ref"] == "um_frozen" and names["um_sca_knn_rag"]["cfg"]["reader"] == "sca"
    z = common.load_scores(os.path.join(S.work, "scores", "um_sca_knn_rag.npz"))
    assert z["meta"]["kind"] == "reader", "a personal reader: an input of the supermodel and the stacker"
    ok("(60) reader 'sca': == LLMEncCA with the same head at initialisation (also for an empty history), its self-attention learns (zero-initialised output projection moves); "
       f"rows um_sca_knn_rag / um_sca_frozen in the usermodel stage, exported as readers (test nDCG@10 {names['um_sca_knn_rag']['ndcg@10']:.4f} vs ca {names['um_knn_rag']['ndcg@10']:.4f})")


# ------------------------------------------------------------------------------------------------ (61)
def test_lr_decay():
    rng = np.random.default_rng(0)
    n, d = 80, 16
    E = rng.standard_normal((n, d)).astype(np.float32)
    E[0] = 0
    N = 400
    H = rng.integers(1, n, (N, 8))
    M = np.ones((N, 8), bool)
    C = rng.integers(1, n, (N, 5))
    pack = (H, M, C)
    vals = iter([0.30, 0.31, 0.3105, 0.3106, 0.31065, 0.3200])

    def val_fn(model):
        return next(vals)
    logs = []
    model, info = UM.train_user_model(E, pack, val_fn, max_epochs=6, bs=64, lr_decay=0.5, decay_after=2, min_delta=2e-4, log=logs.append)
    lrs = [float(l.split("lr=")[1]) for l in logs if "lr=" in l]
    assert np.allclose(lrs, [1e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5][:len(lrs)]), lrs
    # min_delta 2e-4: epochs 1, 2, 3 (+0.0005) improve, 4 (+0.0001) and 5 (+0.00015) are stale -> stop after epoch 5 (patience 2)
    assert info["epochs"] == 5 and info["best_epoch"] == 3 and abs(info["best_val"] - 0.3105) < 1e-12, info
    vals = iter([0.30, 0.31, 0.3105, 0.3106, 0.31065, 0.3200])
    logs = []
    model, info0 = UM.train_user_model(E, pack, val_fn, max_epochs=6, bs=64, log=logs.append)
    lrs0 = [float(l.split("lr=")[1]) for l in logs if "lr=" in l]
    assert all(abs(x - 1e-3) < 1e-12 for x in lrs0), "the default keeps the constant rate of V10"
    assert info0["best_epoch"] == 2 and info0["epochs"] == 4, info0          # min_delta 1e-3 (V10): epoch 3 (+0.0005) is not an improvement
    ok(f"(61) readers: --um-lr-decay 0.5 after epoch 2 gives lr {', '.join(f'{x:.2e}' for x in lrs)}; --um-min-delta 2e-4 keeps epoch 3 (+0.0005) as the best and stops after "
       f"two stale epochs; the defaults keep V10's constant rate and 1e-3 threshold (best epoch 2, 4 epochs)")


# ------------------------------------------------------------------------------------------------ (62)
def _fake_suite(rows):
    """A stand-in with ``records`` / ``row`` / ``cache`` built from {name: per-impression nDCG@10 array}."""
    s = types.SimpleNamespace()
    s.records, s.cache = [], {}
    for name, arr in rows.items():
        arr = np.asarray(arr, np.float64)
        res = {"ids": np.arange(len(arr)), "arr": {"ndcg@10": arr, "auc": 0.5 + arr / 2}, "mean": {"ndcg@10": float(arr.mean()), "auc": float((0.5 + arr / 2).mean())}}
        s.records.append({"name": name, "ndcg@10": float(arr.mean()), "auc": float((0.5 + arr / 2).mean()), "stage": "stack", "epochs": [3, 4], "best_epochs": [3, 2],
                          "val_ndcg@10": 0.4, "ndcg10_per_seed": [float(arr.mean()) - 0.001, float(arr.mean()) + 0.001]})
        s.cache[name] = {"test": res}
    s.row = lambda name: next(r for r in reversed(s.records) if r["name"] == name)
    return s


def test_reports(tmp):
    rng = np.random.default_rng(0)
    base = rng.random(4000) * 0.8
    ctrl = _fake_suite({"super_llm": base, "stack_meta": base + 0.02, "stack_meta_v15cfg": base + 0.021, "stack_pop": base + 0.04})
    treat = _fake_suite({"super_llm": base + 0.004 + rng.normal(0, 0.01, 4000), "stack_meta": base + 0.02 + rng.normal(0, 0.05, 4000),
                         "stack_meta_v15cfg": base + 0.019, "stack_pop": base + 0.03 + rng.normal(0, 0.01, 4000)})
    md_, out = RV.fusion_table(ctrl, treat, os.path.join(tmp, "fusion.md"))
    assert set(out) == {"super_llm", "stack_meta", "stack_meta_v15cfg", "stack_pop"} and abs(out["super_llm"]["d_ndcg@10"][0] - 0.004) < 0.002
    assert "| `super_llm` |" in md_ and "0.4373" in md_ and os.path.exists(os.path.join(tmp, "fusion.md"))
    gmd, gates = RV.gate_v18a(ctrl, treat, os.path.join(tmp, "gate.md"))
    v = {g["row"]: g["verdict"] for g in gates}
    assert v == {"super_llm": "PASS", "stack_meta": "NO GAIN", "stack_pop": "WORSE"}, v
    g2 = next(g for g in gates if g["row"] == "stack_meta")
    assert g2["treatment_best"] == max(treat.row("stack_meta")["ndcg@10"], treat.row("stack_meta_v15cfg")["ndcg@10"]) and g2["best_so_far"] == RV.BEST_OFFLINE == 0.4543
    missing = _fake_suite({"super_llm": base})
    _, gates_m = RV.gate_v18a(ctrl, missing)
    assert {g["row"]: g["verdict"] for g in gates_m} == {"super_llm": "NO GAIN", "stack_meta": "NOT RUN", "stack_pop": "NOT RUN"}
    vmd, vout = RV.vs_v16b_table([("stack_meta", treat, "stack_meta", "stack_meta"), ("absent", treat, "nope", "stack_meta"), ("cat:x", treat, "super_llm", "cat:um_frozen")])
    assert [r["row"] for r in vout] == ["stack_meta", "cat:x"] and abs(vout[0]["diff"] - (treat.row("stack_meta")["ndcg@10"] - 0.4516)) < 1e-12
    emd = RV.epochs_table([(ctrl, "ctrl"), (None, "x")])
    assert "| ctrl |" not in emd                                         # no um_ rows in the fake suite
    ctrl.records.append({"name": "um_frozen", "epochs": [5, 5], "best_epochs": [5, 4], "val_ndcg@10": 0.41, "ndcg@10": 0.42, "stage": "usermodel"})
    assert "| ctrl | um_frozen | [5, 5] | [5, 4] |" in RV.epochs_table([(ctrl, "ctrl")])
    ok("(62) report_v18: control / treatment table with paired Δ and the v16b column; gates PASS / NO GAIN / WORSE / NOT RUN on planted data, G2 counts stack_meta_v15cfg "
       "against 0.4543; the v16b table skips missing rows; the epochs table lists the readers")


def run_all(tmp, mind, enc_dir, dec_dir, cache_dir, run_args, texts):
    test_dtype_rules(enc_dir)
    test_encode_multi(enc_dir, texts)
    test_nonfinite_repair(enc_dir, texts)
    test_base_text(tmp, mind, enc_dir, dec_dir, cache_dir, run_args)
    test_rerun(tmp, mind, enc_dir, dec_dir, cache_dir, run_args)
    test_sca_reader(tmp, mind, enc_dir, cache_dir, run_args)
    test_lr_decay()
    test_reports(tmp)
