"""Tests (63)-(68) of the v19 additions.  Called by ``selftest.main()``.

    (63)  --min-stack-s gates only the stack stage: with little budget left the supermodel still runs and the stacker is skipped (v18b lost both to
          --min-stage-s); the flag is not part of the saved state's signature
    (64)  train_user_tower(minutes=...): after the probe steps the step count shrinks to what fits, the cosine schedule and the evaluation points are laid
          over that count (the learning rate reaches 0 at the last step), the info reports planned / done steps and no deadline hit
    (65)  llm_user.encode_parallel (a copy of the tower per extra device, batch k on device k % n) == UserTower.encode, also with a deadline that stops early
    (66)  the H5 stage on the whole split: training-time evaluations on --llm-eval-n impressions, final scoring on every impression, the scores exported as a
          'reader' that the supermodel and the stacker read; --llm-user-minutes is not part of the saved state's signature; H1 subset sizes (--hq-val-n /
          --hq-test-n)
    (67)  report_v18.gate_v18a(bests=...) compares with the given best-so-far numbers
    (68)  fusion_only (v19c): the output of another notebook is found (a full run before a QUICK one), the arms are laid out with links (control without the
          H5 tower, treatment with it and with the readers trained harder; the supermodel's loader sees the right run tags), a stage exporting into an arm
          replaces the link and never touches the source, missing inputs build nothing, two notebooks' copies are compared file by file, and the seed table
          (seed noise, per-seed effects, 2-seed means of the stackers' saved test scores)
"""
from __future__ import annotations

import os
import shutil
import types

import numpy as np
import torch

import common
import llm_user as LU
import report_v18 as RV
import suite
from selftest_v14 import ok, quiet


# ------------------------------------------------------------------------------------------------ (63)
def test_stack_gate(tmp, mind, enc_dir, cache_dir, run_args):
    keep_ref = suite.EQUIV_REFERENCE
    suite.EQUIV_REFERENCE = None
    try:
        work = os.path.join(tmp, "outGate")
        shutil.rmtree(work, ignore_errors=True)
        base = [*mind, "--work", work, "--encoder", enc_dir, "--cache-dir", cache_dir, "--lean-data", "1", *[a for a in run_args if a != "--no-resume"],
                "--stack-folds", "3", "--stack-rounds", "30", "--stack-configs", "clean,clean_d5", "--stack-control", "0", "--stack-chunk", "90"]
        S1, t1 = quiet(suite.main, [*base, "--stages", "controls,pooling"])
        # 0.5 h of budget: every stage needs 300 s by default, but the stack is asked to leave 1 h -> supermodel runs, stack is skipped
        S2, t2 = quiet(suite.main, [*base, "--budget-hours", "0.5", "--min-stack-s", "3600", "--stages", "controls,pooling,supermodel,stack"])
        st = S2.kv["stage_status"]
        assert st["supermodel"] == "done" and st["stack"].startswith("skipped"), (st, t2[-600:])
        assert "== S supermodel" in t2 and "== T stacker" not in t2
        # without the flag the stacker runs (the same budget is plenty for the toy)
        S3, t3 = quiet(suite.main, [*base, "--budget-hours", "0.5", "--stages", "controls,pooling,supermodel,stack"])
        assert S3.kv["stage_status"]["stack"] == "done" and "== S supermodel" not in t3, "resumed: the supermodel is not re-run, the flag is not in the signature"
        assert "min_stack_s" in suite.Suite.SIG_IGNORED
    finally:
        suite.EQUIV_REFERENCE = keep_ref
        common.CLOCK.reset()                                           # the first --budget-hours of a process is shared by every later suite.main call
    ok("(63) --min-stack-s 3600 with 0.5 h left: the supermodel runs, only the stacker is skipped; without it the stacker runs on resume (the flag is not in the signature)")


# ------------------------------------------------------------------------------------------------ (64)
class _ToyTower:
    """The parts of UserTower that train_user_tower touches: a linear map over bag-of-ids features, with a controllable step time."""

    def __init__(self, d=16, n=60, delay=0.0):
        torch.manual_seed(0)
        self.model = torch.nn.Sequential(torch.nn.EmbeddingBag(n, d, mode="mean"))
        self.mode, self.lam, self.amp, self.device, self.delay = "causal", 1.0, False, "cpu", delay
        self.d, self.n = d, n

    def attach_peft(self, kind, r):
        return sum(p.numel() for p in self.model.parameters())

    def trainable(self):
        return list(self.model.named_parameters())

    def embed(self, texts, mode=None, lam=None):
        import time as _t
        if self.delay:
            _t.sleep(self.delay)
        ids = [torch.tensor([int(w) % self.n for w in t.split()] or [0]) for t in texts]
        offs = torch.tensor([0] + [len(x) for x in ids[:-1]]).cumsum(0)
        return self.model[0](torch.cat(ids), offs)

    def encode(self, texts, bs=64, mode=None, lam=1.0, deadline=None):
        with torch.no_grad():
            return torch.nn.functional.normalize(self.embed(texts), dim=-1).numpy()


def test_time_boxed_training():
    rng = np.random.default_rng(0)
    n, d = 60, 16
    E = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    rows = [(0, list(rng.integers(1, n, 5)), int(rng.integers(1, n)), list(rng.integers(1, n, 4))) for _ in range(4000)]
    qfn = lambda h: " ".join(str(x) for x in h)
    evals = []

    def eval_fn(tw):
        evals.append(1)
        return float(len(evals))                                       # later evaluations are "better": the last one is kept

    tw = _ToyTower(d, n, delay=0.02)
    lrs = []
    orig_step = torch.optim.lr_scheduler.LambdaLR.step

    def spy(self, *a, **k):
        orig_step(self, *a, **k)
        lrs.append(self.get_last_lr()[0])
    torch.optim.lr_scheduler.LambdaLR.step = spy
    try:
        info = LU.train_user_tower(tw, qfn, rows, E, steps=100000, bs=8, warmup=2, eval_fn=eval_fn, minutes=0.05, probe_steps=10, log=lambda *a: None)
    finally:
        torch.optim.lr_scheduler.LambdaLR.step = orig_step
    planned = info["planned_steps"]
    assert 10 < planned < 1000 and info["steps"] == planned and not info["deadline_hit"], info
    assert len(evals) == 3 and info["best_step"] == planned, (len(evals), info["best_step"])
    assert lrs[-1] < 1e-6 and max(lrs) > 5e-5, "the cosine schedule ends at 0 over the planned steps"
    tw2 = _ToyTower(d, n)
    info2 = LU.train_user_tower(tw2, qfn, rows, E, steps=30, bs=8, warmup=2, eval_fn=None, log=lambda *a: None)
    assert info2["planned_steps"] == 30 and info2["steps"] == 30, "without minutes: exactly --llm-user-steps"
    ok(f"(64) time-boxed tower training: {planned} steps fit in 3 s at ~0.02 s per step (probe after 10), 3 evaluations over them, best at the last, learning rate "
       f"decays to {lrs[-1]:.1e}; without minutes exactly 30 steps")


# ------------------------------------------------------------------------------------------------ (65)
def test_encode_parallel(dec_dir, texts):
    tw = LU.UserTower(dec_dir, "causal", max_len=48, device="cpu")
    texts = list(texts[:90])
    one = tw.encode(texts, bs=8)
    two = LU.encode_parallel(tw, texts, bs=8, devices=["cpu", "cpu"])
    three = LU.encode_parallel(tw, texts, bs=8, devices=["cpu", "cpu", "cpu"])
    assert np.allclose(one, two, atol=1e-6) and np.allclose(one, three, atol=1e-6), float(np.abs(one - two).max())
    assert np.array_equal(LU.encode_parallel(tw, texts, bs=8, devices=["cpu"]), one), "one device: plain encode"
    import time as _t
    early = LU.encode_parallel(tw, texts, bs=8, devices=["cpu", "cpu"], deadline=_t.time() - 1)
    assert np.abs(early).sum() == 0, "a passed deadline leaves every row at zero"
    ok("(65) encode_parallel over 2 and 3 devices == UserTower.encode (same batches, a copy of the tower per extra device); one device = encode; a passed deadline encodes nothing")


# ------------------------------------------------------------------------------------------------ (66)
def test_h5_full_split(tmp, mind, dec_dir, cache_dir, run_args):
    common.CLOCK.reset()                                               # no budget left over from an earlier test (the stage skips with < 30 min left)
    keep_ref = suite.EQUIV_REFERENCE
    suite.EQUIV_REFERENCE = None
    try:
        work = os.path.join(tmp, "outH5full")
        shutil.rmtree(work, ignore_errors=True)
        base = [*mind, "--work", work, "--encoder", dec_dir, "--cache-dir", os.path.join(tmp, "h5_cache"), "--lean-data", "1",
                *[a for a in run_args if a != "--no-resume"], "--max-len", "48", "--q-max-len", "96",
                "--llm-user-modes", "causal", "--llm-val-n", "0", "--llm-test-n", "0", "--llm-eval-n", "60", "--llm-user-steps", "40", "--llm-user-bs", "8",
                "--llm-user-r", "4", "--llm-user-lr", "3e-3", "--hq-subset", "1", "--hq-plain", "0", "--hq-val-n", "50", "--hq-test-n", "80",
                "--stack-folds", "3", "--stack-rounds", "30", "--stack-configs", "clean,clean_d5", "--stack-control", "0", "--stack-chunk", "90"]
        S0, _ = quiet(suite.main, [*base, "--stages", "controls,pooling"])
        S, text = quiet(suite.main, [*base, "--llm-user-minutes", "0.2", "--stages", "controls,pooling,histquery,llm_user,supermodel,stack"])
        assert "stage 'pooling' already finished" in text, "resumed although --llm-user-minutes was added: it is not part of the signature"
        assert "llm_user_minutes" in suite.Suite.SIG_IGNORED
        assert "during training on 60 validation impressions" in text and "exported as a reader" in text, text[-1500:]
        r = S.row("ut_causal")
        assert r["n_test"] == len(S.data.test) and r["n_val"] == len(S.data.validation), "the tower is scored on the whole split"
        assert r.get("planned_steps") is not None and r["steps"] <= 40
        z = common.load_scores(os.path.join(work, "scores", "ut_causal.npz"))
        assert z["meta"]["kind"] == "reader" and len(z["test"]["ids"]) == len(S.data.test)
        sv = S.kv["supermodel"]["single_val"]
        assert any(k.endswith(":ut_causal") for k in sv), sorted(sv)
        assert any(k.endswith(":ut_causal") for k in S.kv["stack"]["components"]), "the stacker reads it"
        h1 = S.row("histq_instr")
        assert h1["n_test"] == min(80, len(S.data.test)) and h1["n_val"] == min(50, len(S.data.validation)), (h1["n_test"], h1["n_val"])
    finally:
        suite.EQUIV_REFERENCE = keep_ref
    ok(f"(66) H5 on the whole split: evaluations during training on 60 validation impressions, ut_causal scored on all {r['n_test']} test / {r['n_val']} validation "
       f"impressions and exported as a reader (in the supermodel's single components and the stacker); --llm-user-minutes resumes the saved state; H1 on "
       f"{h1['n_val']} / {h1['n_test']} impressions (--hq-val-n / --hq-test-n)")


# ------------------------------------------------------------------------------------------------ (67)
def test_gate_bests():
    import selftest_v18 as T18
    rng = np.random.default_rng(1)
    base = 0.5 + rng.random(3000) * 0.4                                # every toy row is far above the real numbers
    ctrl = T18._fake_suite({"super_llm": base, "stack_meta": base + 0.02, "stack_pop": base + 0.04})
    treat = T18._fake_suite({"super_llm": base + 0.01, "stack_meta": base + 0.03, "stack_pop": base + 0.05})
    _, g_default = RV.gate_v18a(ctrl, treat)
    _, g_v18 = RV.gate_v18a(ctrl, treat, bests={"super_llm": 0.9, "stack_meta": 0.9, "stack_pop": 0.9})
    assert all(g["verdict"] == "PASS" for g in g_v18) and all(g["beats_best_so_far"] is False for g in g_v18)
    assert all(g["beats_best_so_far"] is True for g in g_default), "toy rows are above the v16b numbers"
    assert RV.BEST_V18 == {"super_llm": 0.4373, "stack_meta": 0.4570, "stack_pop": 0.4731}
    ok("(67) gate_v18a(bests=...): the verdicts come from the paired CI, 'above the best so far' from the given numbers (v19 uses 0.4373 / 0.4570 / 0.4731)")


# ------------------------------------------------------------------------------------------------ (68)
def test_fusion_inputs(tmp):
    import fusion_only as FO
    import selftest_v18 as T18
    import supermodel as SM
    root = os.path.join(tmp, "fusion_in")
    shutil.rmtree(root, ignore_errors=True)
    ptr = np.array([0, 3, 5, 9, 12, 14], np.int64)

    def comp(d, name, seed):
        rag = {"ids": np.arange(5), "ptr": ptr, "flat": np.random.default_rng(seed).random(14).astype(np.float32)}
        common.save_scores(d, name, rag, rag, {"kind": "reader"})

    kin = os.path.join(root, "kaggle_input")
    A, Aq, B = (os.path.join(kin, s, r, "scores") for s, r in (("v19a-x", "results_v19a"), ("v19a-q", "results_v19a_quick"), ("v19b-x", "results_v19b")))
    seeds = {"bge": 10, "cat": 20, "qwen3": 30}
    for run, sd in seeds.items():
        for i, nm in enumerate(("frozen", "um_frozen")):
            comp(os.path.join(A, run), nm, sd + i)
            comp(os.path.join(B, run), nm, sd + i)
    comp(os.path.join(A, "catplus"), "um_knn_rag", 1)
    comp(os.path.join(A, "bgeplus"), "um_knn_rag", 2)
    comp(os.path.join(B, "qwen3"), "ut_causal", 5)
    comp(os.path.join(Aq, "bge"), "frozen", 99)
    comp(os.path.join(B, "cat"), "frozen", 77)                                       # one file of B's copy differs
    assert FO.find_scores("a", [kin]) == os.path.normpath(A), "a full run's output comes before a QUICK one"
    assert FO.find_scores("b", [kin]) == os.path.normpath(B) and FO.find_scores("c", [kin]) is None and FO.find_scores("a", [], override=Aq) == Aq
    work = os.path.join(root, "work")
    P = FO.prepare(work, A, B, arms=("ctrl_s0", "treat_s0", "ctrl_s1"))
    assert not P["missing"] and P["tower"] == ["ut_causal"]
    assert P["checks"]["bge"]["identical"] == 2 and P["checks"]["cat"]["differ"] == ["frozen"] and P["checks"]["qwen3"]["identical"] == 2
    ctrl, treat = P["arms"]["ctrl_s0"], P["arms"]["treat_s0"]
    assert FO.components(os.path.join(ctrl["work"], "scores")) == ["frozen", "um_frozen"], "the control has no H5 tower"
    assert FO.components(os.path.join(treat["work"], "scores")) == ["frozen", "um_frozen", "ut_causal"]
    assert [os.path.basename(d) for d in ctrl["super"]] == ["bge", "cat"] and [os.path.basename(d) for d in treat["super"]] == ["bge", "cat", "catplus", "bgeplus"]
    sd_t = os.path.join(treat["work"], "scores")
    assert os.path.basename(treat["work"]) == "qwen3" and all(os.path.islink(os.path.join(sd_t, f)) for f in os.listdir(sd_t))
    keys = sorted(c.key for c in SM.load_components([treat["work"]] + treat["super"]))
    assert keys == ["bge:frozen", "bge:um_frozen", "bgeplus:um_knn_rag", "cat:frozen", "cat:um_frozen", "catplus:um_knn_rag", "qwen3:frozen", "qwen3:um_frozen",
                    "qwen3:ut_causal"], keys
    src = os.path.join(B, "qwen3", "frozen.npz")
    before = open(src, "rb").read()
    comp(os.path.join(ctrl["work"], "scores"), "frozen", 1234)                       # the controls stage exports 'frozen' into the arm
    assert not os.path.islink(os.path.join(ctrl["work"], "scores", "frozen.npz")) and open(src, "rb").read() == before
    assert os.path.islink(os.path.join(sd_t, "frozen.npz")), "another arm keeps its link"
    shutil.rmtree(os.path.join(A, "bgeplus"))
    P2 = FO.prepare(os.path.join(root, "work2"), A, B)
    assert P2["missing"] == ["v19a:bgeplus"] and not P2["arms"] and not os.path.exists(os.path.join(root, "work2")), "missing inputs build nothing"
    comp(os.path.join(A, "bgeplus"), "um_knn_rag", 2)
    os.remove(os.path.join(B, "qwen3", "ut_causal.npz"))
    assert FO.prepare(os.path.join(root, "work3"), A, B)["missing"] == ["v19b:qwen3:ut_* (the H5 tower)"]
    # the seed table over stand-ins: the seed noise, the per-seed effects and the 'above' flags
    rng = np.random.default_rng(4)
    base = rng.random(3000) * 0.8
    c0, c1 = T18._fake_suite({"stack_meta": base, "stack_pop": base + 0.02}), T18._fake_suite({"stack_meta": base + 0.001, "stack_pop": base + 0.019})
    t0, t1 = T18._fake_suite({"stack_meta": base + 0.01, "stack_pop": base + 0.03}), T18._fake_suite({"stack_meta": base + 0.011, "stack_pop": base + 0.031})
    md_, out = FO.seed_table({"ctrl_s0": c0, "ctrl_s1": c1, "treat_s0": t0, "treat_s1": t1}, {}, bests={"stack_meta": 0.4570, "stack_pop": 0.4731})
    assert abs(out["stack_meta"]["d_seed_ctrl"][0] - 0.001) < 1e-9 and abs(out["stack_pop"]["d_s1"][0] - 0.012) < 1e-9 and abs(out["stack_meta"]["d_s0"][0] - 0.01) < 1e-9
    assert out["stack_meta"]["d_mean"] is None and "stack_llm" not in out and "at seed 0: yes, at seed 1: yes" in md_ and "best so far 0.4731" in md_
    # 2-seed means of the stackers' saved test scores: the mean of the two files' scores, candidate by candidate
    f0, f1 = os.path.join(root, "s0.npz"), os.path.join(root, "s1.npz")
    a0, a1 = rng.random(14).astype(np.float32), rng.random(14).astype(np.float32)
    np.savez_compressed(f0, test_ptr=ptr, stack_pop__test=a0)
    np.savez_compressed(f1, test_ptr=ptr, stack_pop__test=a1)
    seen = {}

    class Eval:
        device = "cpu"

        def run(self, scorer, split):
            b = {"rows": np.arange(5), "cand": np.zeros((5, 4))}
            seen["scores"] = scorer(b).numpy()
            return {"mean": {"ndcg@10": 0.5}}

    assert FO.seed_mean(Eval(), [f0, f1], "stack_pop") == {"mean": {"ndcg@10": 0.5}}
    want = (a0.astype(np.float64) + a1) / 2
    assert np.allclose(seen["scores"][1, :2], want[3:5]) and seen["scores"][1, 2] == 0.0 and np.allclose(seen["scores"][3, :3], want[9:12])
    assert FO.seed_mean(Eval(), [f0, os.path.join(root, "missing.npz")], "stack_pop") is None and FO.seed_mean(Eval(), [f0, f1], "stack_llm") is None
    np.savez_compressed(f1, test_ptr=ptr + 1, stack_pop__test=a1)
    try:
        FO.seed_mean(Eval(), [f0, f1], "stack_pop")
        raise AssertionError("different candidates must be refused")
    except ValueError:
        pass
    # the exploratory table (supermodel chosen on the plain validation) next to the pre-registered rule, with the bases each run chose
    c0, t0 = T18._fake_suite({"super_llm": base, "super_base": base}), T18._fake_suite({"super_llm": base + 0.002, "super_base": base})
    cp, tp = T18._fake_suite({"super_llm": base + 0.001, "super_base": base}), T18._fake_suite({"super_llm": base + 0.006, "super_base": base + 0.004})
    for s_, b_ in ((c0, "cat:um_knn_rag"), (t0, "cat:um_knn_rag"), (cp, "cat:um_knn_rag"), (tp, "catplus:um_knn_rag")):
        s_.kv = {"supermodel": {"base": b_, "groups": {"reader": ["catplus:um_frozen"]}}}
    emd, eout = FO.explore_table({"ctrl_s0": c0, "treat_s0": t0, "ctrl_cw0": cp, "treat_cw0": tp}, os.path.join(root, "explore.md"))
    assert abs(eout["super_llm"]["d_treat"][0] - 0.004) < 1e-9 and abs(eout["super_llm"]["d_plain"][0] - 0.005) < 1e-9 and abs(eout["super_base"]["d_ctrl"][0]) < 1e-12
    assert "EXPLORATORY" in emd and "treatment, plain: base `catplus:um_knn_rag`" in emd and "super_zs" not in eout and os.path.exists(os.path.join(root, "explore.md"))
    assert FO.explore_table({})[1] == {}, "no arm ran: an empty table"
    ok("(68) fusion_only: inputs found (full before QUICK), arms laid out with links (control without the tower, treatment with it and with the readers trained "
       "harder; supermodel run tags right), an export replaces the link not the source, missing inputs build nothing, copies compared, seed table and 2-seed means, "
       "exploratory plain-validation table")


def run_all(tmp, mind, enc_dir, dec_dir, cache_dir, run_args, texts):
    test_stack_gate(tmp, mind, enc_dir, cache_dir, run_args)
    test_time_boxed_training()
    test_encode_parallel(dec_dir, texts)
    test_h5_full_split(tmp, mind, dec_dir, cache_dir, run_args)
    test_gate_bests()
    test_fusion_inputs(tmp)
