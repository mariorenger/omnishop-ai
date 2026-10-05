"""Reference baselines as suite stages.

Headline rows (family ``ref-baseline``) -- NRMS, NAML, Fastformer trained with the usual published recipe on V10's split:
``ref_nrms``, ``ref_naml``, ``ref_ff``.
Ablation ladder (family ``ref-ladder``) -- NRMS moved step by step from V10's recipe to the published one, to attribute V10's gap to the
literature: ``ref_l0`` (V10 recipe re-implemented; must land near V10's own NRMS, otherwise the harness is suspect), ``ref_l1`` (+ title-only
regex tokens, padding mask, negatives re-drawn each epoch), ``ref_l2`` (+ published width / heads / lr, random word vectors); the next rung
(+ GloVe) is ``nrms_ref``.

Each model is its own stage, so a crash or an exhausted budget loses at most one model and a re-run resumes at the next one.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

import common
import rep_eval as R
import refdata as RD
import refmodels as RM
from common import avg_results, log

SPECS = {
    "nrms": dict(name="nrms_ref", family="ref-baseline", preset="nrms", pop=True),
    "naml": dict(name="naml_ref", family="ref-baseline", preset="naml", pop=True),
    "ff": dict(name="fastformer_ref", family="ref-baseline", preset="fastformer", pop=True),
    "nrms_refit": dict(name="nrms_ref_refit", family="ref-baseline", preset="nrms", pop=False, refit="nrms_ref"),     # train_core + validation, epochs fixed by nrms_ref
    "nrms1": dict(name="nrms_ref@1", family="ref-baseline", preset="nrms", pop=False, seed0=1),       # same recipe, other seed: the training-noise floor
    "l0": dict(name="ladder0_v10recipe", family="ref-ladder", preset="v10", pop=False),
    "l1": dict(name="ladder1_data", family="ref-ladder", preset="data", pop=False),
    "l2": dict(name="ladder2_recipe", family="ref-ladder", preset="recipe", pop=False),
}
LADDER = ("ladder0_v10recipe", "ladder1_data", "ladder2_recipe", "nrms_ref")


def resources(S, args):
    """Token tables, GloVe and training sets, built once per process (``S.store``)."""
    if "ref" in S.store:
        return S.store["ref"]
    t0 = time.time()
    text = RD.RefText(S.meta, title_len=args.ref_title_len, body_len=args.ref_body_len, use_glove=bool(args.ref_glove),
                      glove_dim=_dims(args)[0], glove_source=args.ref_glove_path or None, log=log)
    core = RD.read_train_core(args.mind_train, S.meta["ids"], S.data)
    full = RD.TrainSet(core)
    RD.check_train_alignment(full, S.data)                              # always on the whole set: it is the check that the article ids line up
    dyn = full if args.ref_train_frac >= 1 else RD.TrainSet(RD.subsample(core, args.ref_train_frac))
    val_imps = [(r[1], list(r[2]), list(r[3]), list(r[4]), "") for r in S.data.validation]                    # (user, history, candidates, labels, day)
    static = RD.StaticTrainSet(RD.subsample(S.data.train_core, args.ref_train_frac))
    log(f"  reference data: {text.stats} | {dyn.n_samples:,} dynamic-negative samples from {dyn.n_imp:,} impressions "
        f"(V10 static: {static.n_samples:,}) in {time.time() - t0:.0f}s")
    S.store["ref"] = {"text": text, "dyn": dyn, "static": static, "core": core, "val_imps": RD.subsample(val_imps, args.ref_train_frac)}
    return S.store["ref"]


def _dims(args):
    """``--ref-dims emb,heads,head_dim,naml_filters,ff_layers`` (published-style sizes 300,16,16,400,1; the self-test shrinks them)."""
    d = [int(x) for x in args.ref_dims.split(",")]
    if len(d) != 5:
        raise ValueError("--ref-dims needs 5 integers: emb,heads,head_dim,naml_filters,ff_layers")
    return d


def _cfg(spec, args, text):
    cfg = RM.preset(spec["preset"])
    if spec["preset"] not in ("v10", "data"):                          # the V10 rungs keep V10's 64-d / 2-head sizes by definition
        emb, heads, hd, filt, layers = _dims(args)
        cfg = replace(cfg, emb_dim=emb, heads=heads, head_dim=hd, naml_filters=filt, ff_layers=layers)
    if args.ref_max_epochs:
        cfg = replace(cfg, max_epochs=min(cfg.max_epochs, args.ref_max_epochs), min_epochs=min(cfg.min_epochs, args.ref_max_epochs))
    if args.ref_bs:
        cfg = replace(cfg, bs=args.ref_bs)
    if cfg.init == "glove" and text.glove_mat is None:
        cfg = replace(cfg, init="random")                              # no GloVe source worked: same model, random word vectors
    return cfg


def _fit_with_oom_fallback(S, cfg, trainset, val_fn, seed, deadline, text, **kw):
    """Train; on a CUDA out-of-memory error rebuild the network and retry with half / quarter the batch size (reported in the row)."""
    last = None
    for bs in dict.fromkeys([cfg.bs, max(cfg.bs // 2, 8), max(cfg.bs // 4, 8)]):
        net = None
        try:
            net = RM.build_net(cfg, S.device, seed, text, S.data)
            net, info = RM.train_ref(net, trainset, val_fn, replace(cfg, bs=bs), seed=seed, device=S.device, deadline=deadline, log=log, **kw)
            info["bs_used"] = bs
            return net, info
        except torch.cuda.OutOfMemoryError as e:
            last = e
            log(f"    out of memory at batch size {bs}: retrying with a smaller batch")
            del net
            torch.cuda.empty_cache()
    raise last


def stage_refit(S, args, spec, res, cfg, name):
    """Returns False when it could not run.  Refit on train_core + validation for the epoch count that ``nrms_ref`` selected on validation, then test once.  The validation columns
    of the row repeat the epoch-selection run (the refit model has seen those days), so only the test columns are meaningful."""
    base_name = spec["refit"] + ("_noglove" if res["text"].glove_mat is None else "")
    try:
        base = S.row(base_name)
    except KeyError:
        log(f"  {name}: skipped ({base_name} has not run: its validated epoch count is needed)")
        return False
    epochs = int(base["best_epochs"][0])
    trainset = RD.TrainSet(res["core"] + res["val_imps"])
    t0 = time.time()
    net, info = _fit_with_oom_fallback(S, cfg, trainset, None, 0, S.deadline(0.45, cap_s=args.ref_model_minutes * 60), res["text"], fixed_epochs=epochs)
    sc = RM.make_scorer(net)
    test = S.run(sc, "test")
    log(f"    refit on {trainset.n_samples:,} samples for {info['epochs']} of {epochs} epochs ({time.time() - t0:.0f}s)")
    S.record(name, spec["family"], {**cfg.__dict__, "refit_epochs": epochs, "refit_on": "train_core+validation"}, S.cache[base_name]["val"], test, "frozen",
             {"params": info["params"], "train_s": info["train_s"], "init": base.get("init"), "refit_of": base_name, "val_columns": "epoch-selection run (not held out)",
              "partial": info["epochs"] < epochs, "bs_used": info["bs_used"]})


def stage_ref(S, args, which):
    spec = SPECS[which]
    res = resources(S, args)
    text = res["text"]
    cfg = _cfg(spec, args, text)
    name = spec["name"]
    if SPECS[which]["preset"] in ("nrms", "naml", "fastformer") and cfg.init == "random":
        name = name.replace("@1", "") + "_noglove" + ("@1" if "@1" in spec["name"] else "")
    if which == "l2" and text.glove_mat is None:
        log("  ladder2_recipe is identical to nrms_ref_noglove here (no GloVe): skipped")
        return False
    if S.time_left() < args.ref_min_minutes * 60:
        log(f"  {name}: skipped (wall-clock budget)")
        return False
    log(f"\n== reference baseline {name}: {cfg.kind} text={cfg.text} dim={cfg.emb_dim}/{cfg.heads}x{cfg.head_dim} init={cfg.init} "
        f"dyn_neg={cfg.dyn_neg} lr={cfg.lr} bs={cfg.bs} ==")
    if spec.get("refit"):
        return stage_refit(S, args, spec, res, cfg, name)
    trainset = res["dyn"] if cfg.dyn_neg else res["static"]
    twin = spec["pop"] and S.want_pop("frozen")
    vals, tests, pv, pt, infos = [], [], [], [], []
    for sd in range(spec.get("seed0", 0), spec.get("seed0", 0) + args.ref_seeds):
        t0 = time.time()
        val_fn = lambda n: S.run(RM.make_scorer(n), "val")["mean"]["ndcg@10"]
        net, info = _fit_with_oom_fallback(S, cfg, trainset, val_fn, sd, S.deadline(0.45 / args.ref_seeds, cap_s=args.ref_model_minutes * 60 / args.ref_seeds), text)
        sc = RM.make_scorer(net)
        vals.append(S.run(sc, "val"))
        tests.append(S.run(sc, "test"))
        if twin:
            fz = common.fuse([(sc, 1.0), (common.pop_scorer, 1.0)])
            pv.append(S.run(fz, "val"))
            pt.append(S.run(fz, "test"))
        infos.append(info)
        log(f"    seed {sd}: {info['epochs']} epochs (best {info['best_epoch']}), {info['params']:,} parameters, val nDCG@10={info['best_val']:.4f} "
            f"({time.time() - t0:.0f}s)")
        del net, sc
        if S.device == "cuda":
            torch.cuda.empty_cache()
    extra = {"seeds": args.ref_seeds, "epochs": [i["epochs"] for i in infos], "best_epochs": [i["best_epoch"] for i in infos],
             "params": infos[0]["params"], "train_s": sum(i["train_s"] for i in infos), "init": text.glove_label if cfg.init == "glove" else cfg.init,
             "ndcg10_per_seed": [t["mean"]["ndcg@10"] for t in tests],
             "partial": any(h["partial"] for i in infos for h in i["history"]), "bs_used": infos[0]["bs_used"]}
    if args.ref_seeds > 1:
        extra["ndcg10_std"] = float(np.std(extra["ndcg10_per_seed"]))
    row = S.record(name, spec["family"], {**{k: v for k, v in cfg.__dict__.items()}}, avg_results(vals), avg_results(tests), "frozen", extra)
    if twin:
        S.record(f"{name}+pop", f"{spec['family']}+pop", {"base": name}, avg_results(pv), avg_results(pt), "frozen+pop", extra)
    if which == "l0" and args.ref_repro:
        auc_ref, nd_ref = (float(x) for x in args.ref_repro.split(","))
        ok = abs(row["auc"] - auc_ref) < 0.015 and abs(row["ndcg@10"] - nd_ref) < 0.015
        log(f"  harness check vs V10's own NRMS (AUC {auc_ref}, nDCG@10 {nd_ref}): "
            f"{'MATCH (within 0.015)' if ok else 'DIFFERS - a single V10 seed is noisy, but a gap > 0.015 means this harness or V10 trains differently: read the epoch log'}")


def ladder_table(S, repro=None, path=None):
    """Markdown: V10's recipe -> published recipe, one change set per rung, paired Δ nDCG@10 vs the previous rung (test split)."""
    order = [n for n in LADDER if n in S.cache]
    if "nrms_ref" not in S.cache and "nrms_ref_noglove" in S.cache:
        order.append("nrms_ref_noglove")
    rows = [(n, S.cache[n]["test"]) for n in order]
    if len(rows) < 2:
        return None
    what = {"ladder0_v10recipe": "V10 recipe (64-d, 2 heads, N(0,1) word vectors, 20 tokens of title+abstract, rare words -> PAD, no title mask, 4 negatives drawn once, lr 1e-3)",
            "ladder1_data": "+ title-only regex tokens (30), UNK instead of PAD, padding mask, negatives re-drawn every epoch",
            "ladder2_recipe": "+ published NRMS size (300-d input, 16 heads x 16), lr 1e-4, no output projection / attention dropout, N(0,0.1) word vectors",
            "nrms_ref": "+ GloVe-300 word vectors", "nrms_ref_noglove": "(GloVe unavailable: same model with random word vectors)"}
    lines = ["| rung | change | test AUC | test nDCG@10 | Δ nDCG@10 vs previous rung [95% CI] | Δ cold AUC [95% CI] |", "|---|---|---|---|---|---|"]
    if repro:
        a, n = (float(x) for x in repro.split(","))
        lines.append(f"| V10 (reported, its own code) | NRMS as run in V10 | {a:.4f} | {n:.4f} | - | - |")
    prev = None
    for n, t in rows:
        d = R.compare(t, prev, "ndcg@10") if prev is not None else None
        c = R.compare_slice(t, prev, "cold") if prev is not None else None
        fmt = lambda x: "-" if x is None else f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"
        lines.append(f"| {n} | {what.get(n, '')} | {t['mean']['auc']:.4f} | {t['mean']['ndcg@10']:.4f} | {fmt(d)} | {fmt(c)} |")
        prev = t
    md = "\n".join(lines)
    log("\n================ ABLATION LADDER: V10's NRMS recipe -> published recipe (same split, same evaluator) ================\n" + md)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# NRMS ablation ladder (MINDsmall_dev test)\n\n" + md + "\n")
    return md


STAGES = {"ref_nrms": lambda S, a: stage_ref(S, a, "nrms"), "ref_naml": lambda S, a: stage_ref(S, a, "naml"),
          "ref_ff": lambda S, a: stage_ref(S, a, "ff"), "ref_nrms_s1": lambda S, a: stage_ref(S, a, "nrms1"),
          "ref_nrms_refit": lambda S, a: stage_ref(S, a, "nrms_refit"), "ref_l0": lambda S, a: stage_ref(S, a, "l0"),
          "ref_l1": lambda S, a: stage_ref(S, a, "l1"), "ref_l2": lambda S, a: stage_ref(S, a, "l2")}
