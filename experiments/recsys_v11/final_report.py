"""FINAL_REPORT.md of a single-model run: the supermodel, the stacker (v15), what every technique adds on its own, and the reference baselines.

Nothing here is measured again: every number comes from the rows / caches of the finished ``Suite`` (``S.records`` / ``S.cache``) and from
``S.kv["supermodel"]`` (weights, rungs, leave-one-group-out) and ``S.kv["stack"]`` (backend, objective, importances).  The text only names what a table shows; it draws no conclusion that the numbers do not
carry (the verdict words are the ones ``rep_eval.verdict`` and the Bonferroni rule of ``suite.representatives`` already give).

The reference baselines (NRMS / NAML / Fastformer) are NOT trained by the notebooks that call this module: their numbers were measured in the
earlier full runs (``RESULTS_v7.md``, ``RESULTS_v13.md``) on the same V10 split of MIND-small with the same evaluator, and are only printed when the
run is on that split (73,152 test impressions).  Differences to them are therefore a *cross-run* comparison: the confidence interval printed is
the un-paired normal one (wider than a paired interval), and a single trained baseline moves by about 0.007 nDCG@10 from seed to seed.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common
import rep_eval as R

# Measured earlier on MIND-small (V10 split, this evaluator), content only unless the name says +pop.  Source: RESULTS_v7.md (one seed unless
# stated), RESULTS_v13.md (Fastformer: mean of 2 seeds; refit = train_core + validation days).  (AUC, MRR, nDCG@5, nDCG@10, cold AUC, warm AUC)
EARLIER_BASELINES = (
    ("NRMS, seed 0 (v7)", (0.6392, 0.3037, 0.3293, 0.3937, 0.6471, 0.6400)),
    ("NRMS, seed 1 (v7)", (0.6501, 0.3091, 0.3372, 0.4005, 0.6630, 0.6152)),
    ("NRMS refit (v7)", (0.6555, 0.3140, 0.3443, 0.4078, 0.6700, 0.6284)),
    ("NAML (v7)", (0.6471, 0.3079, 0.3364, 0.4006, 0.6698, 0.5640)),
    ("Fastformer (v7)", (0.6455, 0.2993, 0.3288, 0.3926, 0.6674, 0.5785)),
    ("Fastformer, 2 seeds (v13)", (0.6426, 0.3000, 0.3289, 0.3927, 0.6623, 0.5869)),
    ("Fastformer refit (v13)", (0.6550, 0.3097, 0.3387, 0.4033, 0.6767, 0.5883)),
)
EARLIER_BASELINES_POP = (("NRMS + pop (v7)", 0.4383), ("NAML + pop (v7)", 0.4366), ("Fastformer + pop (v13)", 0.4381))
MIND_SMALL_TEST = 73_152
MIND_LARGE_DEV = 376_471
# Published numbers on MIND-large, taken from search results / survey snippets (the papers' PDFs were NOT opened): (model, AUC, nDCG@10).  They are on the
# leaderboard's own test set, not on MINDlarge_dev, so a difference of ~0.01 means nothing.  None = not seen.
PUBLISHED_LARGE = (
    ("NRMS (GloVe), range over papers", "0.670 - 0.691", "0.418 - 0.434"),
    ("PLM-NR", "0.7064", "0.4438"),
    ("UNBERT", "0.7068", "0.4478"),
    ("leaderboard top, ensemble Fastformer + PLM-NR", "0.7268", None),
)

DIRECTIONS = {
    "H8-pooling": "late-interaction / top-k pooling over the frozen vectors",
    "F-text": "more article fields in the text (category, entities, views), mean-pool", "F-text-lse": "more article fields in the text, late interaction",
    "F-history": "longer history (100 / 200 clicks), mean-pool", "F-history-lse": "longer history, late interaction",
    "H11-multi-interest": "zero-shot multi-interest (k-means interests)", "H10-user-model": "learned user model on the frozen vectors (+ multi-interest reader)",
    "H5-LLM-user": "LLM as user encoder (LoRA tower)", "H2-adapt": "contrastive adaptation (head / LoRA)",
    "H3-RAG": "retrieval-augmented article vectors (entity / kNN)", "H4-GraphRAG": "GraphRAG-lite (communities, similar users)",
    "H4b-GraphRAG-LLM": "GraphRAG with LLM-written community summaries", "H6-LLM-aug": "LLM article expansion / reader profiles (KAR-style)",
    "H7-SemanticID": "Semantic IDs + generative ranker", "H9-LLM-rerank": "zero-shot LLM judge re-ranking the top-k",
    "H1-text-user": "history-as-text instruction query", "enc-sweep": "encoder sweep",
    "H12-finetune": "top layers of the encoder fine-tuned under the learned reader (PLM-NR recipe)",
}


def f4(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def ci(t):
    return "-" if not t or np.isnan(t[0]) else f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def _row(S, name):
    return next((r for r in reversed(S.records) if r["name"] == name), None)


def _metrics_line(label, r, extra=()):
    return "| " + " | ".join([label, f"{r['n_test']:,}", f4(r["auc"]), f4(r["mrr"]), f4(r["ndcg@5"]), f4(r["ndcg@10"]), f4(r.get("cold_auc")), f4(r.get("warm_auc")), *extra]) + " |"


def health(S, extra_suites=()):
    out = ["## 0. Run health", ""]
    checks = [(k, v, "this run") for k, v in (S.kv.get("checks") or {}).items()]
    for X in extra_suites:                                        # e.g. the 1-2 minute BGE-small run that verifies the evaluator against V10
        checks += [(k, v, f"protocol check with {os.path.basename(X.args.encoder.rstrip('/'))}") for k, v in (X.kv.get("checks") or {}).items()]
    if checks:
        out += ["| check | result | detail | run |", "|---|---|---|---|"] + [f"| {k} | {'MATCH' if v['ok'] else 'MISMATCH'} | {v['detail']} | {where} |" for k, v, where in checks] + [""]
        if any(not v["ok"] for _k, v, _w in checks):
            out += ["**A check MISMATCHES: do not trust any number below until it is explained.**", ""]
    else:
        out += ["No reproduction check against V10 was run for this encoder / dataset (V10 published numbers exist only for MIND-small and BGE-small).", ""]
    status = S.kv.get("stage_status") or {}
    bad = {k: v for k, v in status.items() if v != "done"}
    out += [f"Stages: {sum(v == 'done' for v in status.values())} done" + (f"; **not done: {bad}**" if bad else "; none failed or skipped") + ".", ""]
    tm = S.kv.get("timings")
    if tm:
        out += ["Stage timings (s): " + ", ".join(f"{k}={v}" for k, v in tm.items()) + f" (sum {sum(tm.values()) / 3600:.2f} h).", ""]
    mem = S.kv.get("memory")
    if mem:
        out += ["Memory after each stage (RAM: resident now, the stage's own peak, the peak of the whole run so far; the GPU figure is the peak since the process started):", ""] + [f"* `{k}`: {v}" for k, v in mem.items()] + [""]
    a = S.kv.get("data_audit")
    if a:
        out += [f"Data: {a['articles']:,} articles, {a['news_columns']} columns in news.tsv (no article body); abstract {a['abstract_words']:.0f} words on average; "
                f"{a['categories']} categories; test readers have {a['test_history_mean']:.1f} clicks on average ({a['test_history_over_50']:.1%} more than 50).", ""]
    caps = []                                                       # what was shrunk to fit a small machine (MIND-large on Kaggle)
    if getattr(S.args, "lean_data", 0):
        caps.append("memory-lean loader (same split / popularity / negatives as the V10 loader)")
    if len(S.data.validation) < S.data.validation_impressions:
        caps.append(f"validation {len(S.data.validation):,} of {S.data.validation_impressions:,} impressions (random, fixed seed)")
    if len(S.data.test) < S.data.test_impressions:
        caps.append(f"test {len(S.data.test):,} of {S.data.test_impressions:,} impressions (random, fixed seed)")
    ts = S.kv.get("train_samples")                                  # recorded by the stacker when it frees the samples
    if ts and ts["used"] < ts["all"]:
        caps.append(f"training samples of the learned stages {ts['used']:,} of {ts['all']:,}")
    elif getattr(S.data, "train_core_all", None) is not None:
        caps.append(f"training samples of the learned stages {len(S.data.train_core):,} of {len(S.data.train_core_all):,}")
    if getattr(S.args, "skip_user_rag", 0):
        caps.append("row `user_rag` skipped")
    if caps:
        out += ["Scale settings: " + "; ".join(caps) + ".  The cold / warm definition and the day split are computed on the full data.", ""]
    cf = S.cache["frozen"]["test"].get("cold_frac") if "frozen" in S.cache else None
    out += [f"Test: {len(S.data.test):,} impressions" + (f"; {cf:.1%} of the clicked articles are cold (unseen in the training rows)." if cf else "."), ""]
    return out


def main_table(S):
    sm = S.kv.get("supermodel") or {}
    names = ["frozen", "popularity", "frozen+pop"] + [r["name"] for r in sm.get("rungs", [])]
    out = ["## 1. The supermodel and its ladder", "",
           "Content-only rows first, then the same rows fused with V10's online popularity (streaming feedback: not comparable with offline papers).  "
           "Δ is paired over the common impressions against the row named in the last column.", "",
           "| row | n test | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC | Δ nDCG@10 vs reference [95% CI] | Δ cold AUC | reference |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    if not sm:
        out += ["| (the supermodel stage did not run: see section 0) |"] + [""]
    for name in names:
        r = _row(S, name)
        if r is None:
            continue
        ref = r["ref"] if name.startswith("super") else ("frozen+pop" if name.endswith("+pop") else "frozen")
        if name in ("frozen", "frozen+pop"):
            ref = name
        d = R.compare(S.cache[name]["test"], S.cache[ref]["test"], "ndcg@10") if ref in S.cache and ref != name else None
        dc = R.compare_slice(S.cache[name]["test"], S.cache[ref]["test"], "cold") if ref in S.cache and ref != name else None
        out.append(_metrics_line(f"`{name}`", r, (ci(d), ci(dc), ref if d else "-")))
    out += [""]
    if sm:
        out += [f"Base (the single LLM-side model with the best cold-weighted validation nDCG@10): `{sm['base']}`.  Groups added, in this order: "
                f"zero-shot extras {sm['groups']['zs']}, learned readers {sm['groups']['reader']}"
                + (f", trained baselines {sm['groups']['baseline']}" if sm["groups"]["baseline"] else "") + (f", LLM tower {sm['groups']['tower']} (its rows are on the subset it scored)" if sm["groups"]["tower"] else "") + ".", ""]
        cw = S.kv.get("cold_weights") or {}
        out += [f"Weights fitted on validation by coordinate ascent on the nDCG@10 re-weighted to {f4(cw.get('target'))} cold clicks (this run's validation: {f4(cw.get('cold_frac'))} cold; the base keeps 1; "
                "components with weight 0 are omitted):", "", "| rung | n val | n test | val nDCG@10 | val nDCG@10 (cold-weighted) | test nDCG@10 | weights |", "|---|---|---|---|---|---|---|"]
        for r in sm["rungs"]:
            out.append(f"| `{r['name']}` | {r['n_val']:,} | {r['n_test']:,} | {r['val_ndcg10']:.4f} | {f4(r.get('val_ndcg10_w'))} | {r['test_ndcg10']:.4f} | "
                       + ", ".join(f"{k.split(':', 1)[-1]}={v:g}" for k, v in r["weights"].items() if v) + " |")
        out += [""]
        plain = [r["name"] for r in sm["rungs"] if not r["pop"]]
        steps = list(zip(plain, plain[1:])) + [(r["name"][:-4], r["name"]) for r in sm["rungs"] if r["pop"]]       # each rung vs the one before; "+pop" vs the same rung without
        full_cov = [n for n in plain if "apex" not in n]
        if len(full_cov) > 2:
            steps.append((full_cov[0], full_cov[-1]))                         # the whole ladder at once: the last full-coverage rung vs the base alone
        pairs = [(a, b) for a, b in steps if a in S.cache and b in S.cache]
        if pairs:
            out += ["Rung-to-rung (paired on the common impressions):", "", "| from | to | Δ nDCG@10 [95% CI] | Δ cold AUC [95% CI] |", "|---|---|---|---|"]
            for a, b in pairs:
                out.append(f"| `{a}` | `{b}` | {ci(R.compare(S.cache[b]['test'], S.cache[a]['test'], 'ndcg@10'))} | {ci(R.compare_slice(S.cache[b]['test'], S.cache[a]['test'], 'cold'))} |")
            out += [""]
        if sm.get("loo"):
            out += [f"Leave-one-group-out from `{sm['loo'][0]['of']}` (the other weights re-fitted on validation; test paired with the full model):", "",
                    "| group removed | test nDCG@10 without it | full model | Δ [95% CI] |", "|---|---|---|---|"]
            for l in sm["loo"]:
                out.append(f"| {l['removed']} | {l['test_ndcg10']:.4f} | {l['full_test_ndcg10']:.4f} | {ci(l['d_vs_full'])} |")
            out += [""]
    return out


def stack_section(S):
    """Section 1b: the stacker's three variants (and the v15 recipe as a control), the configurations compared on the forward holdout, how far the validation score is
    from the test score, the paired differences against the best supermodel rung chosen on VALIDATION, and what its inputs contributed."""
    st = S.kv.get("stack")
    if "stack" not in (S.kv.get("stage_status") or {}):
        return []
    out = ["## 1b. The stacker: a gradient-boosted ranker over the exported scores and label-free features", ""]
    if not st:
        return out + ["(the stack stage did not run: see section 0)", ""]
    cv = st.get("cv", "user")
    folds_txt = (f"{st['folds']} contiguous time blocks of the validation day (a buffer of {st.get('buffer_s', 0) / 60:.0f} minutes left out on each side of the held-out block)"
                 if cv == "time" else f"{st['folds']} folds split by user")
    members = st.get("members") or []
    members_txt = ", ".join("`%s`/`%s`" % (m["config"], m["objective"]) for m in members)
    out += [f"Backend {st['backend']}, {folds_txt}, {st['n_features']} features from {len(st['components'])} exported components"
            + (" (including the entity overlap with the reader's clicks" + (" and the windowed click-through of the last 1 / 6 / 24 hours" if st.get("windowed_ctr") else "") + ")" if st.get("entity_features") else "")
            + ".  Trained on validation impressions only; the validation column is OUT-OF-FOLD; the test column is the mean over the fold models"
            + (f" of {len(members)} configuration(s) ({members_txt})" if members else "") + " of their per-impression z-scores.", "",
            "* `stack_llm`: exported scores (zero-shot, readers, history-free click prior, auxiliary scorers; every encoder run given) + history / slate features of the frozen vectors;",
            "* `stack_meta`: + how often the training rows showed / clicked the article, its category's CTR, and its exposure in the impression log (hours since first shown, shown "
            "before, shown in the last 1 / 6 / 24 completed hours) -- everything available offline, the variant comparable with leaderboards;",
            "* `stack_pop`: + V10's online popularity (and its windowed twin), which count the clicks of earlier impressions of the test day itself (streaming feedback: NOT comparable with offline papers);",
            "* `stack_meta_v15cfg` (control): the v15 recipe on the same components -- folds by user, every feature at full resolution, deep trees -- so that the effect of the v16 changes is a "
            "paired comparison.", "",
            "| row | n test | val nDCG@10 (out-of-fold) | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC |", "|---|---|---|---|---|---|---|---|---|"]
    names = [n for n in ("stack_llm", "stack_meta", "stack_pop") if n in st.get("variants", {}) and _row(S, n)]
    ctrl = "stack_meta_v15cfg" if "stack_meta_v15cfg" in st.get("variants", {}) and _row(S, "stack_meta_v15cfg") else None
    for n in names + ([ctrl] if ctrl else []):
        r = _row(S, n)
        out.append("| " + " | ".join([f"`{n}`", f"{r['n_test']:,}", f4(r["val_ndcg@10"]), f4(r["auc"]), f4(r["mrr"]), f4(r["ndcg@5"]), f4(r["ndcg@10"]),
                                      f4(r.get("cold_auc")), f4(r.get("warm_auc"))]) + " |")
    out += [""]
    sel = st.get("selection") or []
    if sel:
        fw = st.get("forward") or {}
        out += [f"Configurations compared once on a FORWARD holdout of the validation day (train on the first 60% in time = {fw.get('train', 0):,} impressions, stop on the next 10% = "
                f"{fw.get('stop', 0):,}, evaluate on the rest = {fw.get('eval', 0):,}, {fw.get('eval_with_new_click', 0):,} of which have a click on an article no training impression showed; "
                f"features of `{fw.get('variant', '?')}`).  Ranking metric: {st.get('selection_metric', '?')}.  The best {len(members)} are trained in every fold and averaged (marked).", "",
                "| configuration | objective | nDCG@10 on new-article clicks | nDCG@10 all evaluation impressions | rounds | columns | trained in the folds |", "|---|---|---|---|---|---|---|"]
        chosen = {(m["config"], m["objective"]) for m in members}
        for r in sel:
            out.append(f"| `{r['config']}` | `{r['objective']}` | {f4(r['new_ndcg10'])} | {f4(r['ndcg10'])} | {r['rounds']} | {r['features']} | {'yes' if (r['config'], r['objective']) in chosen else ''} |")
        out += [""]
    gap_rows = [(n, _row(S, n)) for n in ([ctrl] if ctrl else []) + names]
    sm = S.kv.get("supermodel") or {}
    ok_rung = lambda r: r["name"] in S.cache and "apex" not in r["name"] and r["n_test"] == len(S.data.test)
    vkey = lambda r: r.get("val_ndcg10_w", r["val_ndcg10"])                           # v16 chooses on the cold-weighted validation mean
    best = {pop: max((r for r in sm.get("rungs", []) if r["pop"] == pop and ok_rung(r)), key=vkey, default=None) for pop in (False, True)}
    gap_rows += [(b["name"], _row(S, b["name"])) for b in best.values() if b and _row(S, b["name"])]
    if gap_rows:
        cw = S.kv.get("cold_weights") or {}
        ft = _row(S, "frozen") or {}
        out += ["How far the validation score is from the test score (nDCG@10).  Cold share of the clicks in this run: validation " + f"{f4(cw.get('cold_frac'))}, test {f4(ft.get('cold_frac'))}; "
                f"the columns marked 'cold-weighted' re-weight the validation impressions to {f4(cw.get('target'))}.  In v15 (MIND-large, `RESULTS_v15_large.md` 2.2) the fixed-recipe rows "
                "(supermodel rungs, unlearned scorers) had the test ABOVE the validation (+0.006 ... +0.04: the validation day is warmer), while the learned stacker had it 0.03-0.04 BELOW -- "
                "a stacker whose test score is far below its validation score is fitting the validation day.  'On new-article clicks' = the out-of-fold score restricted (by weights) to the "
                "impressions whose clicked article none of the training impressions of the row's own folds showed (impressions without any click count 0 in the plain columns and are "
                "absent from this one, so it is not directly comparable with them):", "",
                "| row | validation (out-of-fold / fitted) | validation cold-weighted | validation on new-article clicks | test | test - validation | test - cold-weighted validation |", "|---|---|---|---|---|---|---|"]
        for n, r in gap_rows:
            vw = r.get("val_ndcg@10_w")
            out.append(f"| `{n}` | {f4(r['val_ndcg@10'])} | {f4(vw)} | {f4(r.get('val_new_ndcg10'))} | {f4(r['ndcg@10'])} | {r['ndcg@10'] - r['val_ndcg@10']:+.4f} | "
                       + (f"{r['ndcg@10'] - vw:+.4f}" if vw is not None else "-") + " |")
        out += [""]
    pairs = []
    for n, pop in (("stack_llm", False), ("stack_meta", False), ("stack_pop", True)):
        if n in S.cache and best[pop]:
            pairs.append((n, best[pop]["name"], f"best supermodel rung on validation ({'with' if pop else 'without'} popularity)"))
    for a, b, why in (("stack_meta", "stack_llm", "what the article counts and exposure add"), ("stack_pop", "stack_meta", "what the streaming popularity adds"),
                      ("stack_meta", "stack_meta_v15cfg", "what the v16 changes add (time folds, new-article selection, cleaned features, blend) on the same components")):
        if a in S.cache and b in S.cache:
            pairs.append((a, b, why))
    for ref in ("txt_cta_lse", "um_knn_rag", "pool_lse", "frozen"):                    # single components the stacker is made of
        if "stack_llm" in S.cache and ref in S.cache:
            pairs.append(("stack_llm", ref, "single component"))
    if pairs:
        out += ["Paired differences on the common test impressions (A - B):", "", "| A | B | why this pair | Δ nDCG@10 [95% CI] | Δ AUC [95% CI] | Δ cold AUC [95% CI] |", "|---|---|---|---|---|---|"]
        for a, b, why in pairs:
            ta, tb = S.cache[a]["test"], S.cache[b]["test"]
            out.append(f"| `{a}` | `{b}` | {why} | {ci(R.compare(ta, tb, 'ndcg@10'))} | {ci(R.compare(ta, tb, 'auc'))} | {ci(R.compare_slice(ta, tb, 'cold'))} |")
        out += [""]
    if any(st["variants"][n].get("group_gain_share") for n in names):
        out += ["Share of the trees' total split gain by input group (a description of the fitted model, not a causal effect; the scikit-learn fallback reports none):", "",
                "| variant | " + " | ".join(("scores", "ctx", "meta", "pop")) + " | most-used features |", "|---|---|---|---|---|---|"]
        for n in names + ([ctrl] if ctrl else []):
            v = st["variants"][n]
            g = v.get("group_gain_share") or {}
            out.append(f"| `{n}` | " + " | ".join(f"{g[k]:.2f}" if k in g else "-" for k in ("scores", "ctx", "meta", "pop")) + " | "
                       + ", ".join(f"{k} ({x:.2f})" for k, x in (v.get("top_features") or [])[:6]) + " |")
        out += [""]
    out += ["How to read it: the ranker is fitted on the validation split.  Only the test column and the paired differences above are the result; the validation numbers are what the "
            "model was fitted and stopped on.  If `stack_meta` beats `stack_meta_v15cfg` on the test pairs AND has the smaller test - validation gap, the v16 changes did what they were "
            "built for; if only one of the two holds, the hypothesis that the v15 gap came from fitting the validation day is not settled by this run.", ""]
    return out


def published_large(S):
    """Published MIND-large numbers, printed only for a run on MINDlarge_dev, flagged as secondary."""
    if len(S.data.test) != MIND_LARGE_DEV:
        return []
    out = ["## 4. Published MIND-large numbers (secondary: from search results, the papers were not opened)", "",
           "They are measured on the benchmark's own test set, not on `MINDlarge_dev`, with other training data and tuning; a difference of about 0.01 AUC is not a statement about the models.  "
           "Offline rows only: the `+pop` rows use streaming feedback and have no counterpart here.", "",
           "| model | AUC | nDCG@10 |", "|---|---|---|"]
    out += [f"| {m} | {a} | {n or '-'} |" for m, a, n in PUBLISHED_LARGE]
    mine = [(n, _row(S, n)) for n in ("super_zs", "super_llm", "stack_llm", "stack_meta")]
    mine = [(n, r) for n, r in mine if r is not None and r["n_test"] == len(S.data.test)]
    if mine:
        out += ["", "This run, same 376,471 impressions, offline rows only:", "", "| row | AUC | nDCG@10 |", "|---|---|---|"] + [f"| `{n}` | {f4(r['auc'])} | {f4(r['ndcg@10'])} |" for n, r in mine]
    return out + [""]


def technique_table(S):
    import suite
    out = ["## 2. What each technique does on its own (one row per direction, chosen on validation, Bonferroni over the directions)", "",
           "| direction | representative | n test | nDCG@10 | Δ vs its control [95% CI] | control | Bonferroni verdict | Δ cold AUC [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    reps = suite.representatives(S, pop=False)
    for best, adj, m, n_rows in sorted(reps, key=lambda t: -t[0]["val_ndcg@10"]):
        out.append(f"| {DIRECTIONS.get(best['family'], best['family'])} | `{best['name']}` | {best['n_test']:,} | {f4(best['ndcg@10'])} | {ci(best['d_ndcg@10'])} | {best['ref']} | "
                   f"{best['verdict']} / {adj} (m={m}) | {ci(best.get('d_cold_auc'))} |")
    if not reps:
        out.append("| (no direction rows) |")
    out += [""]
    txt = [(n, _row(S, n)) for n in ("txt_t_lse", "pool_lse", "txt_cta_lse", "txt_ctae_lse", "txt_views_lse", "txt_ctr_lse")]
    txt = [(n, r) for n, r in txt if r is not None]
    if len(txt) > 1:
        out += ["Which part of the text matters (late-interaction rows; Δ paired with `pool_lse`, the title + abstract text):", "", "| text | nDCG@10 | val nDCG@10 | Δ vs `pool_lse` [95% CI] |", "|---|---|---|---|"]
        for n, r in txt:
            d = R.compare(S.cache[n]["test"], S.cache["pool_lse"]["test"], "ndcg@10") if "pool_lse" in S.cache and n != "pool_lse" else None
            out.append(f"| `{n}` | {f4(r['ndcg@10'])} | {f4(r['val_ndcg@10'])} | {ci(d)} |")
        if _row(S, "txt_ctr_lse") and _row(S, "txt_cta_lse"):
            d = R.compare(S.cache["txt_cta_lse"]["test"], S.cache["txt_ctr_lse"]["test"], "ndcg@10")
            out += ["", f"`txt_ctr_lse` is the control: the same template with the category of a random article of ANOTHER category.  `txt_cta_lse - txt_ctr_lse` = {ci(d)}: "
                        + ("the true category beats the wrong one, so the gain is not just the extra words." if d[1] > 0 else
                           "not distinguishable from zero or negative: the gain may be the extra words / format rather than the topic.")]
        out += [""]
    return out


def earlier_baselines(S, name="super_llm"):
    if len(S.data.test) != MIND_SMALL_TEST:
        return ["## 3. Reference baselines", "", "Not printed: the earlier NRMS / NAML / Fastformer numbers exist only for the MIND-small V10 split (73,152 test impressions); "
                "this run's test split is a different one.  No baseline was trained in this notebook.", ""]
    out = ["## 3. NRMS / NAML / Fastformer (measured earlier on this split, not trained here)", "",
           "Same V10 split of MIND-small and the same evaluator, trained with the published recipe (GloVe-300, early stopping on validation; "
           "`refit` = trained again on the validation days too).  They come from `RESULTS_v7.md` / `RESULTS_v13.md`, are NOT from this run, and one trained model moves by "
           "about 0.007 nDCG@10 between seeds.", "",
           "| baseline | AUC | MRR | nDCG@5 | nDCG@10 | cold AUC | warm AUC |", "|---|---|---|---|---|---|---|"]
    for lab, v in EARLIER_BASELINES:
        out.append(f"| {lab} | " + " | ".join(f"{x:.4f}" for x in v) + " |")
    out += [""]
    rows = [(n, _row(S, n)) for n in (name, name + "+pop")]
    rows = [(n, r) for n, r in rows if r is not None and name in S.cache and r["n_test"] == MIND_SMALL_TEST]
    if not rows:
        return out + ["The supermodel row on the full test split is missing, so no comparison is printed.", ""]
    out += ["Supermodel minus each baseline (raw difference of the means; the interval is the UN-paired normal one, ±1.96·sqrt(se_a² + se_b²), taking the "
            "baseline's per-impression spread equal to the supermodel's -- wider than a paired interval, and it ignores the baseline's seed noise):", "",
            "| supermodel row | baseline | Δ AUC | Δ MRR | Δ nDCG@5 | Δ nDCG@10 [≈95% CI] | Δ cold AUC | Δ warm AUC |", "|---|---|---|---|---|---|---|---|"]
    for n, r in rows:
        a = S.cache[n]["test"]["arr"]["ndcg@10"]
        se = float(np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a))))
        half = 1.96 * np.sqrt(2) * se
        pool = EARLIER_BASELINES if not n.endswith("+pop") else ()
        for lab, v in pool:
            d = [r["auc"] - v[0], r["mrr"] - v[1], r["ndcg@5"] - v[2], r["ndcg@10"] - v[3], r.get("cold_auc", np.nan) - v[4], r.get("warm_auc", np.nan) - v[5]]
            out.append(f"| `{n}` | {lab} | " + " | ".join(f"{x:+.4f}" if i != 3 else f"{x:+.4f} [{x - half:+.4f}, {x + half:+.4f}]" for i, x in enumerate(d)) + " |")
        if n.endswith("+pop"):
            for lab, v in EARLIER_BASELINES_POP:
                x = r["ndcg@10"] - v
                out.append(f"| `{n}` | {lab} | - | - | - | {x:+.4f} [{x - half:+.4f}, {x + half:+.4f}] | - | - |")
    out += [""]
    return out


def build(S, path=None, json_path=None, extra_suites=()):
    """Write FINAL_REPORT.md (and a JSON with the supermodel facts) from a finished run; returns the markdown.
    ``extra_suites``: finished runs whose V10 reproduction checks belong in section 0 (the protocol check of the notebook)."""
    extra_suites = tuple(x for x in extra_suites if x is not None)
    title = f"# Final report: one model ({os.path.basename(S.args.encoder.rstrip('/'))}) with every technique applied -- {len(S.data.test):,} test impressions"
    parts = [[title, "", "Everything below is read from the finished run; nothing is measured again.  Confidence intervals are paired over the common impressions and "
              "cover the sampling of the test impressions only, not the training randomness of the learned readers / tower (a seed ensemble is used for them).", ""]]
    for fn in (lambda s: health(s, extra_suites), main_table, stack_section, technique_table, earlier_baselines, published_large):
        try:
            parts.append(fn(S))
        except Exception as e:                                  # a report section must never cost the run
            parts.append([f"## (a report section failed: {repr(e)[:200]})", ""])
    md = "\n".join(line for p in parts for line in p)
    if path:
        common.atomic_write(path, md)
    if json_path:
        checks = {**{f"run: {k}": v for k, v in (S.kv.get("checks") or {}).items()},
                  **{f"{os.path.basename(x.args.encoder.rstrip('/'))}: {k}": v for x in extra_suites for k, v in (x.kv.get("checks") or {}).items()}}
        common.atomic_write(json_path, json.dumps({"supermodel": S.kv.get("supermodel"), "stack": S.kv.get("stack"), "stage_status": S.kv.get("stage_status"), "checks": checks},
                                                  indent=1, default=str))
    return md
