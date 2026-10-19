"""Reports of the two v18 notebooks (MIND-large).

v18a puts a control and a treatment in ONE notebook: the v16b pipeline runs first (the control: supermodel + stacker over the BGE-large, Qwen3 + BGE-large and
Qwen3 runs), then two runs add Qwen3-Embedding-4B vectors, and the supermodel and the stacker run again with those components (the treatment).  Both read the
same validation / test impressions, so every row of the treatment is compared with the same row of the control by a paired CI.  ``fusion_table`` writes that
table, ``gate_v18a`` applies the criteria written down before the run.

v18b trains the readers of the Qwen3 + BGE-large run harder and adds the reader ``sca``; its fused rows can only be compared with v16b, another run on the
same test split (no paired CI, training noise not included): ``vs_v16b_table``.

The reference numbers are the measured rows of v16b on MINDlarge_dev (``RESULTS_v16b_large.md`` sections 1 and 2).
"""
from __future__ import annotations

import numpy as np

import rep_eval as R

# test nDCG@10 of v16b on MINDlarge_dev (376,471 impressions), RESULTS_v16b_large.md
V16B = {"super_base": 0.4260, "super_llm": 0.4373, "stack_llm": 0.4369, "stack_meta": 0.4516, "stack_meta_v15cfg": 0.4543,
        "super_llm+pop": 0.4638, "stack_pop": 0.4731,
        "qwen3:um_frozen": 0.4050, "qwen3:um_knn_rag": 0.4095, "qwen3:um_text": 0.4092,
        "bge:um_frozen": 0.4152, "bge:um_knn_rag": 0.4189, "cat:um_frozen": 0.4223, "cat:um_knn_rag": 0.4226}
BEST_OFFLINE = max(V16B["stack_meta"], V16B["stack_meta_v15cfg"])          # 0.4543: the best offline row measured so far
BEST_ONLINE = V16B["stack_pop"]                                            # 0.4731 (streaming popularity)
BEST_CONTENT = V16B["super_llm"]                                           # 0.4373: content + history only
BEST_V18 = {"super_llm": 0.4373, "stack_meta": 0.4570, "stack_pop": 0.4731}  # after v18a (RESULTS_v18a_large.md): offline best is stack_meta_v15cfg 0.4570
FUSED = ("super_base", "super_zs", "super_llm", "stack_llm", "stack_meta", "stack_meta_v15cfg", "super_llm+pop", "stack_pop")


def fmt_ci(d):
    return "-" if d is None else f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"


def paired(sa, na, sb, nb, metric="ndcg@10"):
    """Paired difference A - B of a test metric on the common impressions -> (mean, lo, hi) or None when a row is missing."""
    try:
        a, b = sa.cache[na]["test"], sb.cache[nb]["test"]
    except (KeyError, AttributeError, TypeError):
        return None
    return R.compare(a, b, metric)


def row_val(s, name, key="ndcg@10"):
    try:
        return float(s.row(name)[key])
    except Exception:
        return None


def _f(x):
    return "-" if x is None else f"{x:.4f}"


def fusion_table(ctrl, treat, path=None, names=FUSED, title="v18a: control (v16b pipeline, this notebook) vs treatment (+ Qwen3-Embedding-4B components), same impressions"):
    """Every fused row of the control next to the same row of the treatment, with the paired Δ (treatment - control) and the v16b number."""
    lines = [f"# {title}", "",
             "| row | control nDCG@10 | treatment nDCG@10 | Δ treatment - control [95% CI] | Δ AUC [95% CI] | v16b (other run) |", "|---|---|---|---|---|---|"]
    out = {}
    for n in names:
        c, t = row_val(ctrl, n), row_val(treat, n)
        if c is None and t is None:
            continue
        d, da = paired(treat, n, ctrl, n), paired(treat, n, ctrl, n, "auc")
        out[n] = {"control": c, "treatment": t, "d_ndcg@10": None if d is None else [float(x) for x in d], "d_auc": None if da is None else [float(x) for x in da]}
        lines.append(f"| `{n}` | {_f(c)} | {_f(t)} | {fmt_ci(d)} | {fmt_ci(da)} | {_f(V16B.get(n))} |")
    lines += ["", "The CI is paired over the test impressions (sampling noise of the test day only); it does not include training noise (two seeds of a reader differ by "
              "0.001 - 0.007 nDCG@10 on MIND-large; XGBoost on a GPU is not bit-exact).  The control is a re-run of v16b, so `control` vs `v16b` shows the run-to-run noise."]
    md = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    return md, out


def gate_v18a(ctrl, treat, path=None, title="v18a gates (written before the run; decided on the paired test CI of the treatment against the in-run control)", bests=None):
    """The criteria written down before the run (also in the notebook's introduction).

    G1 content: treatment super_llm - control super_llm > 0 with the paired CI above 0.
    G2 offline system: treatment stack_meta - control stack_meta > 0 with the CI above 0; it beats the best offline number so far when, in addition, the treatment's
       best offline stack row (stack_meta or stack_meta_v15cfg) is above 0.4543.
    G3 online system: same for stack_pop (best so far 0.4731).
    Verdict per gate: PASS (CI above 0), NO GAIN (CI contains 0), WORSE (CI below 0), NOT RUN (a row is missing)."""
    def verdict(d):
        if d is None:
            return "NOT RUN"
        return "PASS" if d[1] > 0 else ("WORSE" if d[2] < 0 else "NO GAIN")
    bests = bests or {"super_llm": BEST_CONTENT, "stack_meta": BEST_OFFLINE, "stack_pop": BEST_ONLINE}
    gates = []
    for g, n, best in (("G1 content (super_llm)", "super_llm", bests["super_llm"]), ("G2 offline (stack_meta)", "stack_meta", bests["stack_meta"]),
                       ("G3 online (stack_pop)", "stack_pop", bests["stack_pop"])):
        d = paired(treat, n, ctrl, n)
        t = row_val(treat, n)
        if n == "stack_meta":
            alt = row_val(treat, "stack_meta_v15cfg")
            t_best = max([x for x in (t, alt) if x is not None], default=None)
        else:
            t_best = t
        gates.append({"gate": g, "row": n, "verdict": verdict(d), "d": None if d is None else [float(x) for x in d], "treatment": t, "treatment_best": t_best,
                      "best_so_far": best, "beats_best_so_far": None if t_best is None else bool(t_best > best)})
    lines = [f"# {title}", "",
             "| gate | Δ treatment - control [95% CI] | verdict | treatment | best so far | above it? |", "|---|---|---|---|---|---|"]
    for g in gates:
        lines.append(f"| {g['gate']} | {fmt_ci(g['d'])} | **{g['verdict']}** | {_f(g['treatment_best'])} | {g['best_so_far']:.4f} | "
                     f"{'-' if g['beats_best_so_far'] is None else ('yes' if g['beats_best_so_far'] else 'no')} |")
    lines += ["", "PASS = the CI of the paired difference lies above 0.  'above it?' compares with a number of ANOTHER run (v16b): differences below ~0.003 are within the "
              "run-to-run noise that the control row shows.  G2 counts stack_meta_v15cfg too, because it was the best offline row of v16b (0.4543)."]
    md = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    return md, gates


def vs_v16b_table(rows, path=None, title="v18b vs v16b (another run on the same test split; no paired CI, training noise not included)"):
    """v18b: ``rows`` = [(label, suite, row name, v16b key)] -> table of the measured number next to v16b's and their difference."""
    lines = [f"# {title}", "", "| row | v18b nDCG@10 | v16b nDCG@10 | difference | AUC | per seed |", "|---|---|---|---|---|---|"]
    out = []
    for label, s, name, key in rows:
        v = row_val(s, name)
        if v is None:
            continue
        ref = V16B.get(key)
        try:
            seeds = s.row(name).get("ndcg10_per_seed") or []
        except Exception:
            seeds = []
        auc = row_val(s, name, "auc")
        out.append({"row": label, "v18b": v, "v16b": ref, "diff": None if ref is None else v - ref})
        lines.append(f"| `{label}` | {v:.4f} | {_f(ref)} | {'-' if ref is None else f'{v - ref:+.4f}'} | {_f(auc)} | {', '.join(f'{x:.4f}' for x in seeds) or '-'} |")
    lines += ["", f"Best so far: offline {BEST_OFFLINE:.4f} (v16b stack_meta_v15cfg; the pre-registered stack_meta was {V16B['stack_meta']:.4f}), online {BEST_ONLINE:.4f} (stack_pop), "
              f"content only {BEST_CONTENT:.4f} (super_llm).  A difference to another run below ~0.003 (fused rows) or ~0.005 (a reader, two seeds) is within run-to-run noise."]
    md = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    return md, out


def epochs_table(suites_tags, path=None):
    """Epochs run / best epoch of every learned reader (are the readers still improving at the cap?)."""
    lines = ["# Learned readers: epochs run and best epoch per seed", "", "| run | row | epochs | best epoch | val nDCG@10 | test nDCG@10 |", "|---|---|---|---|---|---|"]
    for s, tag in suites_tags:
        if s is None:
            continue
        for r in s.records:
            if not r["name"].startswith("um_") or "epochs" not in r:
                continue
            lines.append(f"| {tag} | {r['name']} | {r.get('epochs')} | {r.get('best_epochs', '-')} | {r.get('val_ndcg@10', float('nan')):.4f} | {r.get('ndcg@10', float('nan')):.4f} |")
    md = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    return md
