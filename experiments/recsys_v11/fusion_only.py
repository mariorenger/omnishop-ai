"""Fusion-only runs (v19c): the supermodel and the stacker over the components of notebooks that already ran, without training any component again.

The v19 notebooks wrote every run's exported component scores to their output (``results_v19x/scores/<run>/*.npz``).  On Kaggle another notebook's output is
mounted read-only under ``/kaggle/input``; ``find_scores`` locates it, ``link_run`` lays the files out as the run directories the supermodel and the stacker read
(``<dir>/scores/*.npz``, the run's tag is the directory's name) with symbolic links, so nothing is copied, and ``same_scores`` checks that two notebooks' copies of
the same run agree (the pipeline is deterministic on the same GPU type: v18a's control reproduced v16b to 4 decimals).  ``prepare`` builds the arms of v19c: the
shared runs and the readers trained harder come from v19a, the main run (with the H5 tower) from v19b, so the control and the treatment read the SAME file for
every component they share.

The stacker subsamples rows and columns (0.7 / 0.7), so its seed is a source of training noise that the paired CI of a row does not include (v18a: adding columns
moved the stacker rows by -0.0019 ... +0.0026).  ``seed_table`` puts the rows of two stacker seeds side by side, measures that noise directly (control seed 1 -
control seed 0: same components) and adds the mean of the two seeds' test scores (an ensemble of two stackers) as rows of their own.

``explore_table`` reports an EXPLORATORY arm chosen after v19a's test result: the supermodel with its base and weights chosen on the plain validation mean
(``--cold-target 0``) instead of the cold-weighted one, which ranked v19a's readers trained harder below the old ones while the test ranked them above.
"""
from __future__ import annotations

import glob
import os

import numpy as np

import common
import rep_eval as R

CONTROL_RUNS = ("bge", "cat")            # the v16b pipeline's other runs (BGE-large; Qwen3-0.6B + BGE-large side by side)
NEW_READERS = ("catplus", "bgeplus")     # v19a: the readers trained harder on the same vectors
MAIN = "qwen3"                           # the main run (Qwen3-Embedding-0.6B); v19b's copy also holds the H5 tower
STACK_ROWS = ("stack_llm", "stack_meta", "stack_meta_v15cfg", "stack_pop")
SUPER_ROWS = ("super_base", "super_zs", "super_llm", "super_llm+pop")


def _is_comp(fn):
    return fn.endswith(".npz") and not fn.endswith(".tmp.npz")


def components(run_dir):
    """Names of the component files of a run directory (the file stem; ``common.save_scores`` keeps names file-safe)."""
    return sorted(fn[:-4] for fn in os.listdir(run_dir) if _is_comp(fn))


def find_scores(tag, roots=("/kaggle/input",), override=None):
    """The ``scores`` directory that notebook v19<tag> wrote (``results_v19<tag>[_quick]/scores``), searched under ``roots``; ``override`` (a path) wins.
    A full run's directory is preferred to a QUICK one.  -> path or None."""
    if override:
        return override if os.path.isdir(override) else None
    hits = set()
    for r in roots:
        if r and os.path.isdir(r):
            hits.update(os.path.normpath(h) for h in glob.glob(os.path.join(r, "**", f"results_v19{tag}*", "scores"), recursive=True) if os.path.isdir(h))
    return sorted(hits, key=lambda h: ("_quick" in h, len(h), h))[0] if hits else None


def runs_in(root):
    """``{run tag: directory}`` of the runs whose component files are under ``root`` (``root/<run>/*.npz``)."""
    if not root or not os.path.isdir(root):
        return {}
    out = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and any(_is_comp(f) for f in os.listdir(p)):
            out[d] = p
    return out


def same_scores(dir_a, dir_b):
    """Compare the component files two run directories share -> ``{"shared", "identical", "differ": [names], "max_abs_diff"}`` (``inf`` when the impressions or
    the candidates of a component differ)."""
    shared = sorted(set(components(dir_a)) & set(components(dir_b)))
    differ, worst = [], 0.0
    for nm in shared:
        za, zb = common.load_scores(os.path.join(dir_a, nm + ".npz")), common.load_scores(os.path.join(dir_b, nm + ".npz"))
        same = True
        for sp in ("val", "test"):
            if not (np.array_equal(za[sp]["ids"], zb[sp]["ids"]) and np.array_equal(za[sp]["ptr"], zb[sp]["ptr"])):
                same, worst = False, float("inf")
                break
            fa, fb = np.asarray(za[sp]["flat"], np.float64), np.asarray(zb[sp]["flat"], np.float64)
            d = float(np.max(np.abs(fa - fb))) if len(fa) else 0.0
            worst = max(worst, d)
            same = same and d == 0.0
        if not same:
            differ.append(nm)
    return {"shared": len(shared), "identical": len(shared) - len(differ), "differ": differ, "max_abs_diff": worst}


def link_run(src_dir, run_dir, exclude=()):
    """``run_dir/scores/<file>`` -> a symbolic link to every component file of ``src_dir`` except those whose name starts with one of ``exclude``.  A stage that
    exports into ``run_dir/scores`` (the controls stage writes ``frozen``) replaces the link by its own file (``common.save_scores`` renames over it), never the
    source.  -> the linked component names."""
    sd = os.path.join(run_dir, "scores")
    os.makedirs(sd, exist_ok=True)
    names = []
    for nm in components(src_dir):
        if any(nm.startswith(x) for x in exclude if x):
            continue
        dst = os.path.join(sd, nm + ".npz")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(os.path.join(src_dir, nm + ".npz")), dst)
        names.append(nm)
    return names


def prepare(work, a_root, b_root, arms=("ctrl_s0", "treat_s0"), tower_prefix="ut_"):
    """Lay out the inputs of the fusion-only arms under ``work``.  Control arms (``ctrl*``) get the main run without the H5 tower and the super-dirs of the v16b
    pipeline; treatment arms (``treat*``) get the tower and, in addition, v19a's readers trained harder.  -> dict: ``missing`` (what is not there; the arms are
    not built then), ``tower`` (its component names), ``checks`` (v19a's copy vs v19b's copy of every shared run), ``arms`` ({arm: {work, super, components,
    treat}})."""
    A, B = runs_in(a_root), runs_in(b_root)
    missing = [f"v19a:{t}" for t in CONTROL_RUNS + NEW_READERS if t not in A] + ([] if MAIN in B else [f"v19b:{MAIN}"])
    tower = [nm for nm in components(B[MAIN]) if nm.startswith(tower_prefix)] if MAIN in B else []
    if MAIN in B and not tower:
        missing.append(f"v19b:{MAIN}:{tower_prefix}* (the H5 tower)")
    out = {"a": a_root, "b": b_root, "missing": missing, "tower": tower, "checks": {}, "arms": {}}
    if missing:
        return out
    for t in CONTROL_RUNS + (MAIN,):                               # v19a's main run has no tower: the comparison covers the control components only
        if t in A and t in B:
            out["checks"][t] = same_scores(A[t], B[t])
    runs = os.path.join(work, "runs")
    for t in CONTROL_RUNS + NEW_READERS:
        link_run(A[t], os.path.join(runs, t))
    super_ctrl = [os.path.join(runs, t) for t in CONTROL_RUNS]
    super_treat = super_ctrl + [os.path.join(runs, t) for t in NEW_READERS]   # the order of v19a's treatment: the stacker's columns follow the directories
    for arm in arms:
        treat = arm.startswith("treat")
        d = os.path.join(work, arm, MAIN)
        names = link_run(B[MAIN], d, exclude=() if treat else (tower_prefix,))
        out["arms"][arm] = {"work": d, "super": super_treat if treat else super_ctrl, "components": len(names), "treat": treat}
    return out


# ----------------------------------------------------------------------------------------------- reports
def _f(x):
    return "-" if x is None else f"{x:.4f}"


def _ci(d):
    return "-" if d is None else f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"


def _row(s, name):
    try:
        return float(s.row(name)["ndcg@10"])
    except Exception:
        return None


def _paired(sa, na, sb, nb):
    try:
        return tuple(float(x) for x in R.compare(sa.cache[na]["test"], sb.cache[nb]["test"], "ndcg@10"))
    except Exception:
        return None


def seed_mean(S_eval, files, variant):
    """Mean of the seeds' test scores of one stacker variant (each is already a mean of within-impression z-scores, ``stack_scores.npz``) -> the V10 evaluator's
    result on the test split (per-impression arrays), or None when a file or the variant is missing.  ``S_eval``: a run whose data are still loaded."""
    arrs, ptr = [], None
    for f in files:
        if not f or not os.path.exists(f):
            return None
        z = np.load(f)
        if f"{variant}__test" not in z.files:
            return None
        if ptr is None:
            ptr = np.asarray(z["test_ptr"])
        elif not np.array_equal(ptr, z["test_ptr"]):
            raise ValueError(f"{variant}: the seeds' files hold different test candidates")
        arrs.append(np.asarray(z[f"{variant}__test"], np.float64))
    if not arrs:
        return None
    flat = np.mean(arrs, 0).astype(np.float32)
    n = len(ptr) - 1
    return S_eval.run(common.ragged_scorer({"ids": np.arange(n), "ptr": ptr, "flat": flat}, np.arange(n), S_eval.device), "test")


def seed_table(arms, files, S_eval=None, path=None, bests=None, rows=STACK_ROWS):
    """The stacker rows of two seeds side by side.  ``arms`` = {"ctrl_s0", "ctrl_s1", "treat_s0", "treat_s1": run or None}; ``files`` = {"ctrl": [stack_scores.npz
    of seed 0, of seed 1], "treat": [...]}; ``S_eval``: a run whose data are still loaded (for the 2-seed means; None = not computed).
    -> (markdown, {row: {...}})."""
    c0, c1, t0, t1 = (arms.get(k) for k in ("ctrl_s0", "ctrl_s1", "treat_s0", "treat_s1"))
    bests = bests or {}
    lines = ["# v19c: the stacker trained with two seeds (row and column subsampling 0.7 / 0.7)", "",
             "| row | control s0 | control s1 | Δ control s1 - s0 [95% CI] (seed noise) | treatment s0 | treatment s1 | Δ treatment - control, s0 | Δ treatment - control, s1 "
             "| control 2-seed mean | treatment 2-seed mean | Δ treatment - control, 2-seed means |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for n in rows:
        r = {"ctrl_s0": _row(c0, n), "ctrl_s1": _row(c1, n), "treat_s0": _row(t0, n), "treat_s1": _row(t1, n),
             "d_seed_ctrl": _paired(c1, n, c0, n), "d_seed_treat": _paired(t1, n, t0, n), "d_s0": _paired(t0, n, c0, n), "d_s1": _paired(t1, n, c1, n),
             "ctrl_mean": None, "treat_mean": None, "d_mean": None}
        if S_eval is not None:
            try:
                cm, tm = seed_mean(S_eval, files.get("ctrl", ()), n), seed_mean(S_eval, files.get("treat", ()), n)
            except Exception as e:                                  # a broken file must not cost the other rows
                cm = tm = None
                r["error"] = repr(e)[:200]
            if cm is not None:
                r["ctrl_mean"] = float(cm["mean"]["ndcg@10"])
            if tm is not None:
                r["treat_mean"] = float(tm["mean"]["ndcg@10"])
            if cm is not None and tm is not None:
                r["d_mean"] = tuple(float(x) for x in R.compare(tm, cm, "ndcg@10"))
        if all(v is None for k, v in r.items() if k != "error"):
            continue
        out[n] = r
        lines.append(f"| `{n}` | {_f(r['ctrl_s0'])} | {_f(r['ctrl_s1'])} | {_ci(r['d_seed_ctrl'])} | {_f(r['treat_s0'])} | {_f(r['treat_s1'])} | {_ci(r['d_s0'])} | "
                     f"{_ci(r['d_s1'])} | {_f(r['ctrl_mean'])} | {_f(r['treat_mean'])} | {_ci(r['d_mean'])} |")
    lines += ["", "Every CI is paired over the test impressions.  'seed noise' is the same stacker on the same components with another seed: a treatment effect smaller "
              "than it is not distinguishable from re-training the stacker.  A 2-seed mean averages the two stackers' test scores (each already a mean of "
              "within-impression z-scores); it is a new row (an ensemble of two stackers), compared with the control's 2-seed mean."]
    robust = []
    for n, r in out.items():
        ok = [d is not None and d[1] > 0 for d in (r["d_s0"], r["d_s1"], r["d_mean"])]
        if r["d_s0"] is not None and r["d_s1"] is not None:
            robust.append(f"* `{n}`: treatment above control at seed 0: {'yes' if ok[0] else 'no'}, at seed 1: {'yes' if ok[1] else 'no'}, "
                          f"2-seed means: {'yes' if ok[2] else ('no' if r['d_mean'] is not None else '-')}")
    if robust:
        lines += ["", "Above = the paired CI lies above 0."] + robust
    key = {"stack_meta": "stack_meta", "stack_meta_v15cfg": "stack_meta", "stack_pop": "stack_pop"}
    best_lines = []
    for n, r in out.items():
        b = bests.get(key.get(n, ""))
        vals = [(lab, r[k]) for lab, k in (("seed 0", "treat_s0"), ("seed 1", "treat_s1"), ("2-seed mean", "treat_mean")) if r[k] is not None]
        if b is not None and vals:
            best_lines.append(f"* `{n}` (best so far {b:.4f}): " + ", ".join(f"{lab} {v:.4f} ({'above' if v > b else 'not above'})" for lab, v in vals))
    if best_lines:
        lines += ["", "Treatment against the best numbers measured before (other runs: differences below ~0.003 are within the run-to-run noise):"] + best_lines
    md = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    return md, out


def supermodel_table(S, path=None, title="Supermodel of the treatment: components and weights chosen on validation"):
    """What the supermodel of a run picked: its base, the members of every group and the weights of every rung (``S.kv['supermodel']``)."""
    st = (getattr(S, "kv", None) or {}).get("supermodel") if S is not None else None
    if not st:
        return ""
    lines = [f"# {title}", "", f"base: `{st['base']}`", ""]
    for g, keys in st.get("groups", {}).items():
        lines.append(f"* {g}: " + (", ".join(f"`{k}`" for k in keys) or "-"))
    lines += ["", "| rung | validation nDCG@10 (cold-weighted) | test nDCG@10 | weights |", "|---|---|---|---|"]
    for r in st.get("rungs", []):
        w = ", ".join(f"{k}={v:g}" for k, v in r["weights"].items() if v)
        lines.append(f"| `{r['name']}` | {r.get('val_ndcg10_w', float('nan')):.4f} | {r['test_ndcg10']:.4f} | {w} |")
    if st.get("loo"):
        lines += ["", "| left out | test nDCG@10 without | with | Δ [95% CI] |", "|---|---|---|---|"]
        for x in st["loo"]:
            lines.append(f"| {x['removed']} | {x['test_ndcg10']:.4f} | {x['full_test_ndcg10']:.4f} | {_ci(x['d_vs_full'])} |")
    md = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    return md


def stack_table(S, path=None, title="Stacker of the treatment: features and what the trees use"):
    """Size of the stacker's table and the gain share of every feature group / the top features of every variant (``S.kv['stack']``)."""
    st = (getattr(S, "kv", None) or {}).get("stack") if S is not None else None
    if not st:
        return ""
    lines = [f"# {title}", "", f"{len(st.get('components', []))} components, {st.get('n_features')} features, groups {st.get('groups')}, backend {st.get('backend')}", "",
             "| variant | test nDCG@10 | gain share by group | top features |", "|---|---|---|---|"]
    for v, info in st.get("variants", {}).items():
        gs = ", ".join(f"{g}={s:.2f}" for g, s in info.get("group_gain_share", {}).items())
        top = ", ".join(f"{n} {s:.2f}" for n, s in info.get("top_features", [])[:6])
        lines.append(f"| `{v}` | {info.get('test_ndcg10', float('nan')):.4f} | {gs} | {top} |")
    md = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    return md


def explore_table(arms, path=None, rows=SUPER_ROWS):
    """The supermodel rows with the base and weights chosen on the cold-weighted validation (``ctrl_s0`` / ``treat_s0``, the pre-registered rule) next to the
    same rows chosen on the plain validation (``ctrl_cw0`` / ``treat_cw0``), with the paired differences.  EXPLORATORY: the plain rule was picked after
    v19a's test result, so a gain here is a lead for a pre-registered run, not a result.  -> (markdown, {row: {...}})."""
    c0, t0, cp, tp = (arms.get(k) for k in ("ctrl_s0", "treat_s0", "ctrl_cw0", "treat_cw0"))
    lines = ["# v19c, EXPLORATORY: supermodel chosen on the plain validation mean (--cold-target 0) instead of the cold-weighted one", "",
             "The rule was picked AFTER v19a's test result (the cold-weighted validation ranked the readers trained harder below the old ones, the test above), "
             "so these rows are a lead for a pre-registered run, not a result.", ""]
    bases = []
    for lab, s in (("control, cold-weighted", c0), ("treatment, cold-weighted", t0), ("control, plain", cp), ("treatment, plain", tp)):
        st = (getattr(s, "kv", None) or {}).get("supermodel") if s is not None else None
        if st:
            bases.append(f"* {lab}: base `{st.get('base')}`; readers {', '.join(f'`{k}`' for k in st.get('groups', {}).get('reader', [])) or '-'}")
    if bases:
        lines += ["Base and readers chosen:"] + bases + [""]
    lines += ["| row | control, cold-weighted | control, plain | Δ plain - cold-weighted (control) | treatment, cold-weighted | treatment, plain "
              "| Δ plain - cold-weighted (treatment) | Δ treatment - control (plain) |", "|---|---|---|---|---|---|---|---|"]
    out = {}
    for n in rows:
        r = {"ctrl": _row(c0, n), "ctrl_plain": _row(cp, n), "treat": _row(t0, n), "treat_plain": _row(tp, n),
             "d_ctrl": _paired(cp, n, c0, n), "d_treat": _paired(tp, n, t0, n), "d_plain": _paired(tp, n, cp, n)}
        if all(v is None for v in r.values()):
            continue
        out[n] = r
        lines.append(f"| `{n}` | {_f(r['ctrl'])} | {_f(r['ctrl_plain'])} | {_ci(r['d_ctrl'])} | {_f(r['treat'])} | {_f(r['treat_plain'])} | {_ci(r['d_treat'])} | "
                     f"{_ci(r['d_plain'])} |")
    md = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    return md, out
