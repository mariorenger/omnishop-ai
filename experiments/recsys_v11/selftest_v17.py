"""Tests (50)-(54) of the v17 addition -- the ``finetune`` stage (``ftune.py``): the top layers of the encoder unfrozen under the learned reader.  Called by ``selftest.main()``.

    (50)  the cache of the lower layers + TopEncoder reproduce the full encoder (CLS and mean pooling, padded batches, truncation), rows == chunk, resume / invalidation of
          the cache, a short read is an error, refresh_table == the per-row top outputs
    (51)  the training step: gradient reaches the top layers and the reader, not the history table; score_raw == score; the cached file is never written by training
    (52)  gate_verdict thresholds, unsupported() reasons, ft_signature ignores the cache directory and tracks the settings
    (53)  the stage inside the suite: rows ft_base / ft_reader with the right families and references, scores exported like any other reader, ft_gate.md / kv / vectors, the two
          arms ran the same number of steps, the phase-0 reader == the ordinary frozen reader (determinism), resume, a changed --ft-* setting re-runs this stage alone
    (54)  the stage declines (no rows, not marked finished) for a decoder-only encoder and for the composite encoder
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common
import ftune as FT
import suite
import usermodel as UM
from selftest_v14 import ok, quiet, raises


def _tiny_bert(tmp, n_layers=4, d=64):
    """A random BERT with ``n_layers`` layers and the word-piece tokenizer of the selftest's tiny encoder."""
    import selftest
    from transformers import AutoTokenizer, BertConfig, BertModel
    texts = [f"word{i % 37} topic{i % 11} news about thing{i % 53} " + " ".join(f"w{(i * j) % 41}" for j in range(1 + i % 9)) for i in range(300)]
    path = os.path.join(tmp, "tiny_bert4")
    selftest.build_tiny_encoder(path, texts)
    tok = AutoTokenizer.from_pretrained(path)
    torch.manual_seed(0)
    model = BertModel(BertConfig(vocab_size=len(tok), hidden_size=d, num_hidden_layers=n_layers, num_attention_heads=2, intermediate_size=2 * d,
                                 max_position_embeddings=128)).eval()
    return model, tok, texts


def _full_embeddings(model, tok, texts, L, pooling):
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=L, return_tensors="pt")
        h = model(**{k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).last_hidden_state
        if pooling == "cls":
            v = h[:, 0]
        else:
            m = batch["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
    return F.normalize(v, dim=-1)


# ------------------------------------------------------------------------------------------------ (50)
def test_cache_and_top(tmp):
    model, tok, texts = _tiny_bert(tmp)
    texts = texts[:120]
    L, cut, d = 12, 2, 64                                                                  # L small enough that some texts are truncated
    n = len(texts)
    for pooling in ("cls", "mean"):
        cdir = os.path.join(tmp, f"ftcache_{pooling}")
        shutil.rmtree(cdir, ignore_errors=True)
        cache = FT.HiddenCache(cdir, n, L, d, np.float32, {"model": "tiny", "cut": cut, "texts": "t"})
        assert not cache.is_complete()
        cache.build(model, tok, texts, "", cut, "cpu", amp=False, bs=32, logf=lambda *a: None)
        assert cache.is_complete() and os.path.getsize(cache.path) == n * L * d * 4
        assert len(FT.encoder_layers(model)) == 4, "build() puts the full stack back"
        cache.open()
        layers = FT.encoder_layers(model)
        top = FT.TopEncoder([copy.deepcopy(x) for x in list(layers)[cut:]], model.config, pooling).eval()
        ids = np.arange(n)
        with torch.no_grad():
            got = top(torch.from_numpy(cache.rows(ids)), torch.from_numpy(cache.lens))
        want = _full_embeddings(model, tok, texts, L, pooling)
        assert float((got - want).abs().max()) < 1e-5, (pooling, float((got - want).abs().max()))
        n_cut = int((cache.lens >= L).sum())
        assert 0 < n_cut < n, "the test needs both truncated and untruncated texts"
        # padding invariance: a text's vector does not depend on what it was batched with
        solo = _full_embeddings(model, tok, [texts[7]], L, pooling)
        assert float((solo[0] - got[7]).abs().max()) < 1e-5
        # rows vs chunk vs shuffled ids
        a = cache.rows(np.array([5, 40, 3]))
        b = cache.chunk(3, 6)
        assert np.array_equal(a[2], b[0]) and np.array_equal(a[0], b[2]) and cache.rows(np.array([])).shape == (0, L, d)
        # refresh_table == the per-row outputs, row 0 is padding
        T = FT.refresh_table(top, cache, "cpu", n + 1, bs=50, amp=False)
        assert T.shape == (n + 1, d) and float(T[0].abs().max()) == 0.0 and float((T[1:] - got).abs().max()) < 1e-6
        # the equivalence check used by the stage
        cmin, cmean = FT.check_equivalence(model, tok, top, cache, texts, "", pooling, "cpu", amp=False)
        assert cmin > 0.99999 and cmean > 0.99999, (cmin, cmean)
        cache.close()
    # the additive-mask fallback (what 4.x layers take) and a float16 cache (what the GPU run stores) reproduce the encoder too
    c16 = FT.HiddenCache(os.path.join(tmp, "ftcache_f16"), n, L, d, np.float16, {"model": "tiny", "cut": cut, "texts": "t"})
    shutil.rmtree(c16.dir, ignore_errors=True)
    c16.build(model, tok, texts, "", cut, "cpu", amp=False, bs=32, logf=lambda *a: None)
    c16.open()
    layers = FT.encoder_layers(model)
    top16 = FT.TopEncoder([copy.deepcopy(x) for x in list(layers)[cut:]], model.config, "cls").eval()
    for mode in ("additive", "auto"):
        FT.MASK["mode"] = mode
        cmin16, _ = FT.check_equivalence(model, tok, top16, c16, texts, "", "cls", "cpu", amp=False)
        assert cmin16 > 0.999, (mode, cmin16)
    FT.MASK["mode"] = "auto"
    c16.close()
    # invalidation: other cut / model tag, a truncated file, a missing file
    same = FT.HiddenCache(cdir, n, L, d, np.float32, {"model": "tiny", "cut": cut, "texts": "t"})
    assert same.is_complete()
    assert not FT.HiddenCache(cdir, n, L, d, np.float32, {"model": "tiny", "cut": cut + 1, "texts": "t"}).is_complete()
    assert not FT.HiddenCache(cdir, n, L, d, np.float32, {"model": "tiny", "cut": cut, "texts": "other"}).is_complete()
    assert not FT.HiddenCache(cdir, n + 1, L, d, np.float32, {"model": "tiny", "cut": cut, "texts": "t"}).is_complete()
    assert not FT.HiddenCache(cdir, n, L + 4, d, np.float32, {"model": "tiny", "cut": cut, "texts": "t"}).is_complete()
    with open(same.path, "r+b") as f:
        f.truncate(same.nbytes - 10)
    assert not same.is_complete(), "a cache cut short is rebuilt, not trusted"
    big = FT.HiddenCache(cdir, n + 5, L, d, np.float32, {"model": "tiny", "cut": cut, "texts": "t"})
    big.lens = np.zeros(n + 5, np.int64)
    big._fd = os.open(same.path, os.O_RDONLY)
    raises(OSError, big.chunk, n, n + 5)                                                    # reads past the end of the file: short read -> error, not garbage
    os.close(big._fd)
    ok("(50) cached lower layers + TopEncoder == the full encoder (CLS and mean pooling, padded batches, truncated texts: max |diff| < 1e-5), a text's vector does not depend on its "
       "batch, rows == chunk, refresh_table == the per-row outputs, the cache is invalidated by another cut / texts / size / a cut-short file, a short read raises")


# ------------------------------------------------------------------------------------------------ (51)
def test_training_step(tmp):
    model, tok, texts = _tiny_bert(tmp)
    texts = texts[:80]
    L, cut, d, n = 12, 2, 64, len(texts)
    cdir = os.path.join(tmp, "ftcache_train")
    shutil.rmtree(cdir, ignore_errors=True)
    cache = FT.HiddenCache(cdir, n, L, d, np.float32, {"model": "tiny", "cut": cut, "texts": "t"})
    cache.build(model, tok, texts, "", cut, "cpu", amp=False, bs=32, logf=lambda *a: None)
    cache.open()
    with open(cache.path, "rb") as f:
        before = f.read()
    layers = FT.encoder_layers(model)
    top = FT.TopEncoder([copy.deepcopy(x) for x in list(layers)[cut:]], model.config, "cls")
    top0 = copy.deepcopy(top.state_dict())
    T = FT.refresh_table(top, cache, "cpu", n + 1, amp=False)
    assert not T.requires_grad
    rng = np.random.default_rng(0)
    B, Hn = 6, 5
    hist = torch.from_numpy(rng.integers(0, n + 1, (B, Hn)))
    hmask = hist > 0
    cand = rng.integers(1, n + 1, (B, 5))
    uniq, inv = np.unique(cand, return_inverse=True)
    reader = FT.FTReader(T.numpy(), dim=16, device="cpu")
    # score_raw == score on the same inputs
    cand_t = torch.from_numpy(cand)
    reader.eval()                                                                          # dropout off: the two paths are the same function
    with torch.no_grad():
        assert float((reader.score_raw(T[hist], hmask, T[cand_t]) - reader.score(hist, hmask, cand_t)).abs().max()) < 1e-6
    reader.train()
    top.train()
    ev = top(torch.from_numpy(cache.rows(uniq - 1)), torch.from_numpy(cache.lens[uniq - 1]))
    logits = reader.score_raw(T[hist], hmask, ev[torch.from_numpy(inv.reshape(cand.shape))])
    F.cross_entropy(logits, torch.zeros(B, dtype=torch.long)).backward()
    g_top = [p.grad for p in top.parameters()]
    assert all(g is not None for g in g_top) and sum(float(g.abs().sum()) for g in g_top) > 0, "the candidate vectors send gradient to the top layers"
    assert all(p.grad is not None for p in reader.parameters()) and T.grad is None, "the history table is a constant"
    opt = torch.optim.AdamW(top.parameters(), lr=1e-3)
    opt.step()
    moved = max(float((top.state_dict()[k] - top0[k]).abs().max()) for k in top0)
    assert moved > 1e-6, "an optimiser step moves the unfrozen layers"
    with open(cache.path, "rb") as f:
        assert f.read() == before, "training never writes the cache"
    cache.close()
    ok("(51) training step: candidate vectors from the top layers send gradient to them and to the reader, the history table is a constant, score_raw == score, an optimiser step moves "
       "the unfrozen layers, the cached file is untouched")


# ------------------------------------------------------------------------------------------------ (52)
def test_gate_and_signature():
    gv = FT.gate_verdict
    assert gv(0.0061, 0.004, 0.006) == "PASS" and gv(0.006, 0.0, 0.006) == "PASS"
    assert gv(0.0059, 0.004, 0.006) == "INCONCLUSIVE" and gv(0.002, 0.002, 0.006) == "INCONCLUSIVE"
    assert gv(0.0019, 0.002, 0.006) == "FAIL" and gv(-0.01, -0.01, 0.006) == "FAIL"
    assert gv(0.02, -0.002, 0.006) == "INCONCLUSIVE", "a big cold-weighted gain with a plain loss is not a PASS"
    import argparse
    a = argparse.Namespace(encoder="concat:A+B", news_emb=None)
    assert "composite" in FT.unsupported(a)
    assert "no encoder" in FT.unsupported(argparse.Namespace(encoder="BAAI/bge-large-en-v1.5", news_emb="vecs.npz"))
    assert "decoder-only" in FT.unsupported(argparse.Namespace(encoder="Qwen/Qwen3-Embedding-0.6B", news_emb=None))
    assert FT.unsupported(argparse.Namespace(encoder="BAAI/bge-large-en-v1.5", news_emb=None)) is None
    assert FT.unsupported(argparse.Namespace(encoder="intfloat/e5-large-v2", news_emb=None)) is None
    p = suite.build_parser()
    a1 = p.parse_args(["--mind-train", "x", "--mind-dev", "y"])
    a2 = p.parse_args(["--mind-train", "x", "--mind-dev", "y", "--ft-cache-dir", "/elsewhere"])
    a3 = p.parse_args(["--mind-train", "x", "--mind-dev", "y", "--ft-steps", "5"])
    assert FT.ft_signature(a1) == FT.ft_signature(a2) != FT.ft_signature(a3)
    ok("(52) gate: PASS needs the cold-weighted gain (and no plain loss), a third of it is INCONCLUSIVE, less is FAIL; unsupported() names the reason (composite, vectors from a file, "
       "decoder-only); ft_signature ignores the cache directory and tracks every other --ft-* setting")


# ------------------------------------------------------------------------------------------------ (53) (54)
def test_stage(tmp, mind, enc_dir, dec_dir, cache_dir, run_args):
    work = os.path.join(tmp, "outFT")
    shutil.rmtree(work, ignore_errors=True)
    ftdir = os.path.join(tmp, "ft_hidden_stage")
    shutil.rmtree(ftdir, ignore_errors=True)
    base = [*mind, "--work", work, "--encoder", enc_dir, "--cache-dir", cache_dir, "--lean-data", "1", *[a for a in run_args if a not in ("--no-resume",)],
            "--um-views", "frozen", "--um-extra", "", "--um-seeds", "1", "--um-seeds-focus", "1", "--um-epochs", "3",
            "--ft-layers", "1", "--ft-maxlen", "32", "--ft-steps", "60", "--ft-batch", "8", "--ft-refresh", "10", "--ft-eval-every", "20", "--ft-minutes", "5",
            "--ft-min-gain", "0.0", "--ft-cache-dir", ftdir, "--export-scores", "1"]
    S, text = quiet(suite.main, [*base, "--stages", "controls,usermodel,finetune"])
    assert S.kv["stage_status"]["finetune"] == "done", text[-1500:]
    names = {r["name"]: r for r in S.records}
    assert {"um_frozen", "ft_base", "ft_reader"} <= set(names)
    assert names["ft_base"]["family"] == "FT-control" and names["ft_reader"]["family"] == "H12-finetune"
    assert names["ft_base"]["ref"] == "um_frozen" and names["ft_reader"]["ref"] == "ft_base"
    assert "FT-control" in suite.REFERENCE_FAMILIES and "H12-finetune" in suite.TECHNIQUE_FAMILIES_EXCLUDED
    # the exported scores are those of the row, shaped like every other reader's
    z = common.load_scores(os.path.join(work, "scores", "ft_reader.npz"))
    zr = common.load_scores(os.path.join(work, "scores", "um_frozen.npz"))
    assert z["meta"]["kind"] == "reader" and len(z["val"]["ids"]) == len(zr["val"]["ids"]) and len(z["test"]["flat"]) == len(zr["test"]["flat"])
    assert not os.path.exists(os.path.join(work, "scores", "ft_base.npz")), "the control is not an input of the supermodel / stacker"
    # the gate, its report and the vectors
    g = S.kv["ft_gate"]
    assert g["verdict"] in ("PASS", "INCONCLUSIVE", "FAIL") and g["curve"][0]["step"] == 0 and g["curve_frozen"][0]["step"] == 0
    assert [c["step"] for c in g["curve"]][-1] == [c["step"] for c in g["curve_frozen"]][-1] == g["steps_run"], "both arms ran the same number of steps"
    assert abs(g["gain_w"] - (g["val_w_best"] - g["val_w_frozen"])) < 1e-12 and g["cos_min"] > 0.99
    md = open(os.path.join(work, "ft_gate.md"), encoding="utf-8").read()
    assert f"gate {g['verdict']}" in md and "| step |" in md
    vec = np.load(os.path.join(work, "ft_vectors.npy"))
    assert vec.shape == (S.data.n_news - 1, S.E0.shape[1]) and vec.dtype == np.float16
    # determinism: the phase-0 reader is the ordinary frozen reader (same seed, same table, same epochs)
    assert abs(g["val_start"] - names["um_frozen"]["val_ndcg@10"]) < 1e-9, (g["val_start"], names["um_frozen"]["val_ndcg@10"])
    assert any(c["name"] == "finetune: cached top layers == full encoder" and c["ok"] for c in [dict(name=k, **v) for k, v in S.kv["checks"].items()])
    ok("(53) the stage: rows ft_base (FT-control, vs um_frozen) and ft_reader (H12-finetune, vs ft_base), scores exported like a reader (not for the control), gate / report / vectors "
       f"written, both arms ran {g['steps_run']} steps, the phase-0 reader == the ordinary frozen reader (val {g['val_start']:.4f}), verdict {g['verdict']} on a toy")
    # resume: nothing runs again; a changed --ft-* setting re-runs this stage alone and does not duplicate rows
    rows = sorted(r["name"] for r in S.records)
    S2, t2 = quiet(suite.main, [*base, "--stages", "controls,usermodel,finetune"])
    assert "stage 'finetune' already finished" in t2 and "== FT" not in t2 and sorted(r["name"] for r in S2.records) == rows
    S3, t3 = quiet(suite.main, [*base, "--ft-steps", "40", "--stages", "controls,usermodel,finetune"])
    assert "finetune stage: settings changed" in t3 and "== FT" in t3 and "== H10" not in t3 and "hidden states of the lower 1 layers: reused" in t3, t3[-800:]
    assert sorted(r["name"] for r in S3.records) == rows and S3.kv["ft_gate"]["steps_run"] <= 40
    ok("(53b) resume: unchanged settings run nothing again; a changed --ft-* setting re-runs the finetune stage alone (the cache of the lower layers is reused, no duplicated rows)")
    # (54) declines
    for enc, why, kind in ((dec_dir, "decoder-only", "decoder"), ("concat:" + enc_dir + "+" + dec_dir, "composite", "concat")):
        w2 = os.path.join(tmp, f"outFTdecl_{kind}")
        shutil.rmtree(w2, ignore_errors=True)
        args = [a for a in base]
        args[args.index("--work") + 1] = w2
        args[args.index("--encoder") + 1] = enc
        ref_keep = suite.EQUIV_REFERENCE
        suite.EQUIV_REFERENCE = None
        try:
            Sd, td = quiet(suite.main, [*args, "--stages", "controls,finetune", "--um-views", "frozen"])
        finally:
            suite.EQUIV_REFERENCE = ref_keep
        assert "not run:" in td and why in td and Sd.kv["stage_status"]["finetune"].startswith("declined"), td[-600:]
        assert not any(r["name"] in ("ft_base", "ft_reader") for r in Sd.records)
    ok("(54) the stage declines (rows absent, status 'declined', not marked finished) for a decoder-only encoder and for the composite encoder")
    return S


def run_all(tmp, mind, enc_dir, dec_dir, cache_dir, run_args):
    test_cache_and_top(tmp)
    test_training_step(tmp)
    test_gate_and_signature()
    return test_stage(tmp, mind, enc_dir, dec_dir, cache_dir, run_args)
