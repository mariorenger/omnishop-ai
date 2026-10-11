"""Supermodel: the best single model plus every technique that helps, fused with weights fitted on validation.

Every stage that produces a candidate component keeps its raw per-candidate scores (``<work>/scores/<name>.npz``, see ``Suite.export_scores``):
zero-shot late-interaction scorers over the BGE / Qwen3 vectors with the category / entity text variants, the learned readers (mean of their
seeds' z-scores), the trained Fastformer / NRMS / NAML baselines (same), the LoRA-tuned encoder, and the LLM user tower (a validation /
test *subset*).  This module reads them back and builds the combined model in a way that cannot touch the test split before the end:

  1. components are z-scored inside every impression (the recipe of V10's ``bge_zs_pop`` and of ``common.fuse``);
  2. the BASE is the single LLM-side model (zero-shot or learned reader) with the best COLD-WEIGHTED validation nDCG@10 (v16: the validation impressions are
     re-weighted so that their cold / warm click mix is the test split's -- ``Suite.val_weights``; v15 chose ``um_text``, strong on warm articles, over ``um_knn_rag``);
  3. rungs add one group of techniques at a time -- the other zero-shot scorers, the learned readers, the trained baselines (hybrid),
     the LLM tower (on its subset) -- and a ``+pop`` twin adds V10's online popularity; at every rung the weights are re-fitted on
     VALIDATION by coordinate ascent on the same weighted nDCG@10 (the base keeps weight 1, a small grid, a few sweeps);
  4. the test split is evaluated once per rung with the validation weights, and a leave-one-group-out table says what each group contributes
     to the final model.

The "LLM-only" rungs never see a trained recommender baseline; only the ``hybrid`` rung does, so that "the LLM side beats NAML / NRMS /
Fastformer" can be told apart from "an ensemble that contains Fastformer beats Fastformer".  Rows go into the table as family ``SUPER`` (never a
candidate "direction" in the decision summary).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

import common
import metrics as ref_metrics
import rep_eval as R
from common import log

TOP = {"zs": 4, "reader": 3, "baseline": 3}            # components kept per group (ranked by their own validation nDCG@10)
GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)             # weights tried per coordinate (the base keeps weight 1)
SWEEPS = 3


# --------------------------------------------------------------------------------------------- components
class Comp:
    """One exported component: raw and z-scored ragged scores on the validation and test impressions it was evaluated on."""

    def __init__(self, run, name, kind, val, test, meta):
        self.run, self.name, self.kind, self.meta = run, name, kind, meta
        self.key = f"{run}:{name}"
        for sp, rag in (("val", val), ("test", test)):
            if not np.all(np.diff(rag["ids"]) > 0):
                raise ValueError(f"{self.key}: impression ids of the {sp} scores are not strictly increasing")
        self.val, self.test = common.ragged_zscore(val), common.ragged_zscore(test)

    def covers(self, n_val, n_test):
        return len(self.val["ids"]) == n_val and len(self.test["ids"]) == n_test


def load_components(dirs):
    """All components under ``<dir>/scores`` of every run directory (the run's tag is the directory name)."""
    comps = []
    for d in dirs:
        sd = os.path.join(d, "scores")
        if not os.path.isdir(sd):
            log(f"  no exported scores under {sd}")
            continue
        run = os.path.basename(os.path.normpath(d))
        for fn in sorted(os.listdir(sd)):
            if not fn.endswith(".npz") or fn.endswith(".tmp.npz"):
                continue
            z = common.load_scores(os.path.join(sd, fn))
            kind = z["meta"].get("kind") or ("tower" if z["name"].startswith("ut_") else "zs")
            comps.append(Comp(run, z["name"], kind, z["val"], z["test"], z["meta"]))
    return comps


def merge_towers(comps):
    """The tower runs of one run directory (``ut_causal``, ``ut_causal@1`` ...) become one component: the mean of their z-scores."""
    towers = [c for c in comps if c.kind == "tower"]
    rest = [c for c in comps if c.kind != "tower"]
    by_run = {}
    for c in towers:
        by_run.setdefault(c.run, []).append(c)
    for run, cs in by_run.items():
        ok = [c for c in cs if np.array_equal(c.val["ids"], cs[0].val["ids"]) and np.array_equal(c.test["ids"], cs[0].test["ids"])]
        m = Comp(run, "ut_causal[ensemble]" if len(ok) > 1 else ok[0].name, "tower", common.ragged_mean_z([c.val for c in ok]),
                 common.ragged_mean_z([c.test for c in ok]), {"members": [c.name for c in ok]})
        rest.append(m)
    return rest


# --------------------------------------------------------------------------------------------- fusion on a fixed set of impressions
def ndcg_at_k(S, Y, M, k=10):
    """nDCG@k per impression for padded scores (the formula of ``rep_eval.batch_metrics``, without the AUC part and without tie repair)."""
    Sm = S.masked_fill(~M, float("-inf"))
    pos = ((Y > 0.5) & M).float()
    order = Sm.argsort(dim=1, descending=True)
    ys = pos.gather(1, order)
    C = S.shape[1]
    ranks = torch.arange(1, C + 1, device=S.device).float()
    disc = 1.0 / torch.log2(ranks + 1.0)
    dcg = (ys[:, :k] * disc[:k]).sum(1)
    P = pos.sum(1)
    cum = torch.cumsum(disc[:min(k, C)], 0)
    kk = torch.clamp(P, max=min(k, C)).long()
    ideal = torch.where(kk > 0, cum[(kk - 1).clamp(min=0)], torch.zeros_like(dcg))
    return torch.where(ideal > 0, dcg / ideal.clamp(min=1e-12), torch.zeros_like(dcg))


def ndcg_tie_aware(S, Y, M, k=10):
    """nDCG@k exactly as ``rep_eval.evaluate`` reports it: an impression whose scores contain exact ties (users with an empty history, popularity
    counts that coincide) is scored with V10's ``metrics.py`` ordering convention instead of the arbitrary order of the sort."""
    out = ndcg_at_k(S, Y, M, k)
    Sm = S.masked_fill(~M, float("-inf"))
    eq = (Sm[:, :, None] == Sm[:, None, :]) & M[:, :, None] & M[:, None, :]
    tie = (eq.sum((1, 2)) > M.sum(1)).cpu().numpy()
    if tie.any():
        n_valid, sc, lb = M.sum(1).cpu().numpy(), Sm.cpu().numpy(), Y.cpu().numpy()
        out = out.clone()
        for r in np.nonzero(tie)[0]:
            n = int(n_valid[r])
            out[r] = float(ref_metrics.ndcg_score(lb[r, :n], sc[r, :n].astype(np.float32), k))
    return out


class Fusion:
    """Z-scored component scores of ``rows[ids]`` (+ optionally V10's online popularity as the last component), laid out batch by batch."""

    def __init__(self, rows, comps, split, ids, device, with_pop=False, es=None, imp_w=None):
        """``imp_w``: optional weight of every impression of the split (indexed by original row id, e.g. ``Suite.val_weights()``); ``ndcg10`` is then the weighted mean."""
        self.rows, self.comps, self.split, self.device, self.with_pop = rows, comps, split, device, with_pop
        self.ids = np.asarray(ids, np.int64)
        self.imp_w = None if imp_w is None else np.asarray(imp_w, np.float64)[self.ids]
        full = len(self.ids) == len(rows) and np.array_equal(self.ids, np.arange(len(rows)))
        if full and es is not None and len(es.rows) == len(rows):          # the Suite's own EvalSet: same batches, no second copy on the device
            self.es = es
        else:
            self.es = R.EvalSet(rows, device=device) if full else common.subset_evalset(rows, self.ids, device)
        want = np.array([len(rows[i][3]) for i in self.ids])
        self.idx = []
        for c in comps:
            rag = getattr(c, split)
            ix = np.searchsorted(rag["ids"], self.ids)
            if len(ix) and (ix.max() >= len(rag["ids"]) or not np.array_equal(rag["ids"][ix], self.ids)):
                raise ValueError(f"{c.key}: has no scores for some of the {split} impressions of this fusion")
            if not np.array_equal(np.diff(rag["ptr"])[ix], want):
                raise ValueError(f"{c.key}: number of scored candidates differs from the data (scores exported for another split / cap?)")
            self.idx.append(ix)
        self.K = len(comps) + int(with_pop)
        self.names = [c.key for c in comps] + (["pop"] if with_pop else [])
        self._batches = None

    def _assemble(self, b):
        rows = b["rows"]
        B, C = b["cand_mask"].shape
        Z = np.zeros((self.K, B, C), np.float32)
        for k, (c, ix) in enumerate(zip(self.comps, self.idx)):
            rag = getattr(c, self.split)
            flat, ptr = rag["flat"], rag["ptr"]
            for j, r in enumerate(rows):
                i = ix[r]
                v = flat[ptr[i]:ptr[i + 1]]
                Z[k, j, :len(v)] = v
        Zt = torch.from_numpy(Z).to(self.device)
        if self.with_pop:
            Zt[-1] = common.zscore(b["pop"].float(), b["cand_mask"])
        return Zt

    def batches(self):
        if self._batches is None:
            self._batches = [(b["labels"], b["cand_mask"], self._assemble(b),
                              None if self.imp_w is None else torch.as_tensor(self.imp_w[b["rows"]], dtype=torch.float32, device=self.device)) for b in self.es]
        return self._batches

    @torch.no_grad()
    def ndcg10(self, w):
        """nDCG@10 of the weighted sum of the components, averaged over the impressions (weighted by ``imp_w`` when there is one)."""
        wt = torch.as_tensor(np.asarray(w, np.float32), device=self.device)
        tot, n = 0.0, 0.0
        for Y, M, Z, W in self.batches():
            m = ndcg_tie_aware(torch.einsum("k,kbc->bc", wt, Z), Y, M)
            if W is None:
                tot += float(m.sum())
                n += len(m)
            else:
                tot += float((m * W).sum())
                n += float(W.sum())
        return tot / max(n, 1e-12)

    def scorer(self, w):
        wt = torch.as_tensor(np.asarray(w, np.float32), device=self.device)
        return lambda b: torch.einsum("k,kbc->bc", wt, self._assemble(b))


def fit_weights(fusion, init, free, grid=GRID, sweeps=SWEEPS):
    """Coordinate ascent on validation nDCG@10 over the weights in ``free`` (all others stay at ``init``)."""
    w = np.array(init, np.float64)
    best = fusion.ndcg10(w)
    for _ in range(sweeps):
        moved = False
        for k in free:
            cur = w[k]
            for g in grid:
                if g == cur:
                    continue
                w2 = w.copy()
                w2[k] = g
                v = fusion.ndcg10(w2)
                if v > best + 1e-9:
                    best, w, moved = v, w2, True
        if not moved:
            break
    return w, best


# --------------------------------------------------------------------------------------------- the stage
def _pick(comps, kind, n, ranks, allowed=None):
    cs = [c for c in comps if c.kind == kind and (allowed is None or any(c.name == a or c.name.startswith(a) for a in allowed))]
    return sorted(cs, key=lambda c: -ranks[c.key])[:n]


def stage_supermodel(S, args):
    log("\n== S supermodel: the best single LLM-side model + every technique that helps, fused with weights fitted on validation ==")
    dirs = [S.work] + [d for d in args.super_dirs.split(",") if d]
    comps = merge_towers(load_components(dirs))
    if not comps:
        log("  no exported component scores found (stages not run, or --export-scores 0): supermodel not run")
        return False
    n_val, n_test = len(S.data.validation), len(S.data.test)
    full = [c for c in comps if c.covers(n_val, n_test) and c.kind in ("zs", "reader", "baseline")]       # "aux" / "feature" scores are the stacker's inputs
    towers = [c for c in comps if c.kind == "tower"]
    if sum(c.kind in ("zs", "reader") for c in full) < 1:
        log("  no LLM-side component (zero-shot scorer or reader) covers the whole validation / test split: supermodel not run")
        return False
    dev = S.device
    allv = np.arange(n_val)
    wv = S.val_weights()                           # validation impressions re-weighted to the test split's cold share: the base, the grid and the weights are chosen on this
    probe = Fusion(S.data.validation, full, "val", allv, dev, es=S.es["val"], imp_w=wv)
    ranks = {}
    for k, c in enumerate(full):
        e = np.zeros(probe.K)
        e[k] = 1.0
        ranks[c.key] = probe.ndcg10(e)
    cw = S.kv.get("cold_weights", {})
    log(f"  single components on validation (nDCG@10, cold-weighted to {cw.get('target', 0):.0%}): " + ", ".join(f"{c.key}={ranks[c.key]:.4f}" for c in sorted(full, key=lambda c: -ranks[c.key])[:12]))
    baselines_ok = [x for x in args.super_baselines.split(",") if x]
    llm_side = [c for c in full if c.kind in ("zs", "reader")]
    base = max(llm_side, key=lambda c: ranks[c.key])
    zs = [c for c in _pick(full, "zs", TOP["zs"] + 1, ranks) if c is not base][:TOP["zs"]]
    readers = [c for c in _pick(full, "reader", TOP["reader"] + 1, ranks) if c is not base][:TOP["reader"]]
    trained = _pick(full, "baseline", TOP["baseline"], ranks, allowed=baselines_ok or None)
    log(f"  base (best single LLM-side model on validation): {base.key}; zero-shot extras {[c.key for c in zs]}; readers {[c.key for c in readers]}; "
        f"trained baselines {[c.key for c in trained]}; tower {[c.key for c in towers]}")
    ref, ref_pop = ("frozen" if "frozen" in S.cache else None), ("frozen+pop" if "frozen+pop" in S.cache else None)
    state = {"base": base.key, "single_val": {k: float(v) for k, v in ranks.items()}, "rungs": [], "loo": [], "groups": {
        "zs": [c.key for c in zs], "reader": [c.key for c in readers], "baseline": [c.key for c in trained], "tower": [c.key for c in towers]}}
    S.kv["supermodel"] = state

    def run_rung(name, rung_comps, pop, init_w=None, only_new=None, tower=False):
        """Fit on validation, evaluate val + test once, record the row.  ``init_w``: {key: weight} warm start.  ``only_new``: keys whose weights are fitted
        (default: all but the base)."""
        t0 = time.time()
        if tower:                                   # the tower scores a subset: validation / test restricted to the impressions every component covers
            vid = rung_comps[0].val["ids"]
            tid = rung_comps[0].test["ids"]
            for c in rung_comps[1:]:
                vid, tid = np.intersect1d(vid, c.val["ids"]), np.intersect1d(tid, c.test["ids"])
        else:
            vid, tid = allv, np.arange(n_test)
        fv = Fusion(S.data.validation, rung_comps, "val", vid, dev, with_pop=pop, es=S.es["val"], imp_w=wv)
        ft = Fusion(S.data.test, rung_comps, "test", tid, dev, with_pop=pop, es=S.es["test"])
        w0 = np.zeros(fv.K)
        for i, nm in enumerate(fv.names):
            w0[i] = 1.0 if nm == base.key else (init_w or {}).get(nm, 1.0 if nm == "pop" else 0.0)
        free = [i for i, nm in enumerate(fv.names) if nm != base.key and (only_new is None or nm in only_new)]
        w, best = fit_weights(fv, w0, free)
        vres = S.run(fv.scorer(w), "val", es={"val": fv.es})
        tres = S.run(ft.scorer(w), "test", es={"test": ft.es})
        weights = {nm: float(x) for nm, x in zip(fv.names, w)}
        row = S.record(name + ("+pop" if pop else ""), "SUPER+pop" if pop else "SUPER",
                       {"base_component": base.key, "weights": weights, "components": [c.key for c in rung_comps], "fit": f"coordinate ascent on the cold-weighted validation nDCG@10, grid {list(GRID)}, {SWEEPS} sweeps"},
                       vres, tres, ref_pop if pop else ref,
                       {"val_fit_ndcg10": best, "n_components": len(rung_comps) + int(pop), "fit_s": time.time() - t0})
        state["rungs"].append({"name": row["name"], "weights": weights, "val_ndcg10": row["val_ndcg@10"], "val_ndcg10_w": row["val_ndcg@10_w"], "test_ndcg10": row["ndcg@10"],
                               "n_val": row["n_val"], "n_test": row["n_test"], "pop": pop})
        log("    weights: " + ", ".join(f"{k}={v:g}" for k, v in weights.items() if v))
        return weights

    # rung 0: the base alone (the reference every other rung is compared with)
    run_rung("super_base", [base], False)
    chain, w_prev, w_pop = [base], {}, {}
    rungs = [("super_zs", zs), ("super_llm", readers), ("super_hybrid", trained)]
    for name, group in rungs:
        if not group:
            log(f"  {name}: no components in this group -> rung skipped")
            continue
        chain = chain + group
        w_prev = run_rung(name, chain, False, init_w=w_prev)
        if name in ("super_llm", "super_hybrid"):
            w_pop = run_rung(name, chain, True, init_w=w_prev)
    # rung 4: + the LLM tower, on the subset it scored; the other weights are the ones fitted on the full validation split.  The popularity twin starts from
    # the weights of the last popularity rung (tower weight 0), so on the tower's subset it can only improve on that rung (v14 started it from the no-pop weights
    # and lost 0.0065 nDCG@10 to super_llm+pop on the same impressions)
    if towers and chain:
        w_apex = run_rung("super_apex", chain + towers, False, init_w=w_prev, only_new={towers[0].key}, tower=True)
        run_rung("super_apex", chain + towers, True, init_w=w_pop or w_apex, only_new={towers[0].key, "pop"}, tower=True)

    # leave-one-group-out from the last full-coverage model with popularity
    last = next((r for r in reversed(state["rungs"]) if r["pop"] and "apex" not in r["name"]), None)
    if last is not None:
        used = [c for c in chain if c.key in last["weights"]]
        groups = {"zero-shot extras": [c for c in used if c in zs], "learned readers": [c for c in used if c in readers],
                  "trained baselines": [c for c in used if c in trained], "popularity": []}
        fv = Fusion(S.data.validation, used, "val", allv, dev, with_pop=True, es=S.es["val"], imp_w=wv)
        ft = Fusion(S.data.test, used, "test", np.arange(n_test), dev, with_pop=True, es=S.es["test"])
        w_full = np.array([last["weights"].get(nm, 0.0) for nm in fv.names])
        t_full = S.run(ft.scorer(w_full), "test", es={"test": ft.es})
        for gname, members in groups.items():
            keys = {c.key for c in members} | ({"pop"} if gname == "popularity" else set())
            if not keys:
                continue
            w0 = np.array([0.0 if nm in keys else w_full[i] for i, nm in enumerate(fv.names)])
            free = [i for i, nm in enumerate(fv.names) if nm != base.key and nm not in keys]
            w, best = fit_weights(fv, w0, free, sweeps=2)
            tres = S.run(ft.scorer(w), "test", es={"test": ft.es})
            d = R.compare(tres, t_full, "ndcg@10")
            state["loo"].append({"removed": gname, "members": sorted(keys), "val_ndcg10": float(best), "test_ndcg10": float(tres["mean"]["ndcg@10"]),
                                 "d_vs_full": [float(x) for x in d], "full_test_ndcg10": float(t_full["mean"]["ndcg@10"]), "of": last["name"]})
            log(f"  without {gname:18s}: test nDCG@10 {tres['mean']['ndcg@10']:.4f} vs {t_full['mean']['ndcg@10']:.4f} with it (Δ {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}])")
    return True
