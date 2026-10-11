"""Stacker (v15): a gradient-boosted ranker over the LLM-side scores and a few label-free features, trained on the validation split.

Why.  ``supermodel.py`` fuses the components linearly with ONE global weight per component, fitted on validation.  Measured on MIND-large
(``RESULTS_v14_large.md``) that validation split is warmer than the test split (82% of the test clicks are on articles no training row showed):
the weights over-trust what wins on warm articles (the trained readers, RAG) and the fused model lost to a smaller one on test.  A tree ensemble sees
how cold / how new / how widely shown a candidate is next to the scores, so it can weigh them differently per situation instead of once for all.

What it reads, per candidate (nothing here reads a test label before the final evaluation):

  ``scores``  every exported component that covers the whole validation and test split -- of this run and of the runs in ``--super-dirs`` (a second
              encoder): zero-shot scorers, the learned readers (``um_*``), the history-free click prior (``um_prior_*``) and the auxiliary scorers
              (plain mean-pool, top-k, max-sim, RAG ...) -- as the z-score inside the impression, plus the raw value for a handful of scorers;
  ``ctx``     history / slate features of the frozen vectors (``featscore.py``);
  ``meta``    counts of the article over everything labelled before the split's first day (train_core for validation impressions, train_core + the
              validation days for test ones: shown / clicked / smoothed CTR, "never shown", "unseen" as the cold/warm slices define it) and the
              pooled CTR of its category and sub-category,
              and its exposure in the impression log -- hours since it first appeared in an EARLIER impression, how many earlier impressions showed it,
              and how many showed it in the last 1 / 6 / 24 completed hours (label-free; ``data_lean.exposure_features_lean``);
  ``pop``     V10's online smoothed log-CTR -- counts the clicks of earlier impressions of the same day, i.e. streaming feedback; kept apart.

Variants: ``stack_llm`` = scores + ctx, ``stack_meta`` = + meta (all of it available offline: the one comparable with leaderboards), ``stack_pop`` = + pop.

Protocol.  The ranker is trained on the VALIDATION impressions only, K folds split by user (an impression never sits in the training part of a fold that
scores it); the out-of-fold scores are the validation result; the test result is the mean over the fold models of their per-impression z-scored
predictions, computed once at the end.  Early stopping uses a held-out tenth of each fold's training users.  The objective (listwise ``rank:ndcg`` vs
pointwise ``binary:logistic``) is chosen on the held-out part of fold 0.  Backends: XGBoost (GPU when there is one) -> LightGBM -> scikit-learn.
"""
from __future__ import annotations

import gc
import os
import re
import time

import numpy as np
import torch

import common
import featscore as FS
import rep_eval as R
from common import log

GROUPS = ("scores", "ctx", "meta", "pop")
VARIANTS = {"stack_llm": ("scores", "ctx"), "stack_meta": ("scores", "ctx", "meta"), "stack_pop": ("scores", "ctx", "meta", "pop")}
RAW_RE = re.compile(r"^(frozen|pool_(maxsim|topk|lse)|txt_cta(_lse)?|txt_views_lse|hist100_lse|knn_rag)$")        # scorers whose raw value is also a column
META_ITEM = ("m_log_shown", "m_log_pos", "m_ctr", "m_never_shown", "m_unseen", "m_cat_ctr", "m_sub_ctr")
META_EXPO = ("m_age_h", "m_log_prior_shown", "m_expo_1h", "m_expo_6h", "m_expo_24h")
META_REL = ("m_ctr_z", "m_age_z", "m_expo_z", "m_expo6_z")


# ----------------------------------------------------------------------------------------------- small helpers
def expand(ptr, lens, imp):
    """Candidate-row positions of the impressions ``imp`` (sorted) in the flat layout described by ``ptr`` / ``lens``."""
    imp = np.asarray(imp, np.int64)
    n = lens[imp]
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    first = np.cumsum(n) - n
    return np.repeat(ptr[imp] - first, n) + np.arange(tot)


def user_folds(users, k, seed=0):
    """Deterministic fold of every user (hash), so an impression is scored by models that never saw its user."""
    h = (np.asarray(users, np.uint64) * np.uint64(2654435761) + np.uint64(seed * 7919 + 1)) % np.uint64(2 ** 32)
    return (h % np.uint64(k)).astype(np.int64)


def es_mask(users, seed=0):
    """About a tenth of the users, for early stopping."""
    h = (np.asarray(users, np.uint64) * np.uint64(40503) + np.uint64(seed * 104729 + 17)) % np.uint64(2 ** 32)
    return (h % np.uint64(10)) == 0


def make_seg_z(ptr):
    """Within-impression z-score of flat float arrays laid out by ``ptr`` (the formula of ``common.ragged_zscore``; the segment index is built once)."""
    lens = np.diff(ptr)
    seg = np.repeat(np.arange(len(lens)), lens)
    n = np.maximum(lens, 1)

    def f(flat):
        v = np.asarray(flat, np.float64)
        mu = np.bincount(seg, weights=v, minlength=len(lens)) / n
        sd = np.sqrt(np.bincount(seg, weights=(v - mu[seg]) ** 2, minlength=len(lens)) / n)
        return ((v - mu[seg]) / (sd[seg] + 1e-6)).astype(np.float32)
    return f


def seg_z(flat, ptr):
    return make_seg_z(ptr)(flat)


def ragged_of(split, flat, imp):
    """Ragged scores of the impressions ``imp`` cut out of a flat array in the split's layout."""
    imp = np.asarray(imp, np.int64)
    sel = expand(split.ptr, split.lens, imp)
    return {"ids": imp, "ptr": np.concatenate([[0], np.cumsum(split.lens[imp])]).astype(np.int64), "flat": flat[sel]}


# ----------------------------------------------------------------------------------------------- one split's layout and per-impression sources
class Split:
    def __init__(self, rows, aux=None):
        self.rows, self.aux = rows, aux
        self.n = len(rows)
        self.lens = np.fromiter((len(r[3]) for r in rows), np.int64, self.n)
        self.ptr = np.concatenate([[0], np.cumsum(self.lens)]).astype(np.int64)
        self.N = int(self.ptr[-1])
        self.users = np.fromiter((r[1] for r in rows), np.int64, self.n)

    def cand(self, a, b):
        return np.concatenate([np.asarray(r[3], np.int64) for r in self.rows[a:b]]) if b > a else np.zeros(0, np.int64)

    def labels(self, a, b):
        return np.concatenate([np.asarray(r[4], np.float32) for r in self.rows[a:b]]) if b > a else np.zeros(0, np.float32)

    def pop(self, a, b):
        return np.concatenate([np.asarray(r[5], np.float32) for r in self.rows[a:b]]) if b > a else np.zeros(0, np.float32)

    def exposure(self, a, b):
        return np.concatenate(self.aux[a:b]).astype(np.float32) if b > a else np.zeros((0, len(META_EXPO)), np.float32)


class Component:
    def __init__(self, key, kind, name, run, val, test):
        self.key, self.kind, self.name, self.run = key, kind, name, run
        self.flat = {"val": val, "test": test}
        self.raw = bool(RAW_RE.match(name)) and kind in ("zs", "aux")


def load_components(dirs, val, test, kinds=("zs", "reader", "aux"), baselines=()):
    """Every exported component of the run directories that covers the whole validation and test split.  A component exported for ANOTHER
    candidate layout (other caps) is an error, as in the supermodel; one that covers only a subset (the LLM tower) is skipped.  Trained baselines are
    used only when their name starts with one of ``baselines``."""
    out, taken, skipped = [], set(), []
    for d in dirs:
        sd = os.path.join(d, "scores")
        if not os.path.isdir(sd):
            log(f"  no exported scores under {sd}")
            continue
        run = os.path.basename(os.path.normpath(d))
        tag, i = run, 1
        while tag in taken:
            i += 1
            tag = f"{run}{i}"
        taken.add(tag)
        for fn in sorted(os.listdir(sd)):
            if not fn.endswith(".npz") or fn.endswith(".tmp.npz"):
                continue
            z = common.load_scores(os.path.join(sd, fn))
            kind = z["meta"].get("kind") or common.default_kind(z["name"])
            if kind == "baseline" and any(z["name"].startswith(x) for x in baselines if x):
                pass
            elif kind not in kinds:
                skipped.append(f"{tag}:{z['name']} ({kind})")
                continue
            covered = True
            for sp, sh in (("val", val), ("test", test)):
                rag = z[sp]
                if len(rag["ids"]) != sh.n or not np.array_equal(rag["ids"], np.arange(sh.n)):
                    covered = False
                    break
                if not np.array_equal(rag["ptr"], sh.ptr):
                    raise ValueError(f"{tag}:{z['name']}: scored candidates differ from the data (scores exported for another split / cap?)")
            if not covered:
                skipped.append(f"{tag}:{z['name']} (does not cover the whole split)")
                continue
            out.append(Component(f"{tag}:{z['name']}", kind, z["name"], tag, np.asarray(z["val"]["flat"], np.float32), np.asarray(z["test"]["flat"], np.float32)))
    if skipped:
        log(f"  not used by the stacker: {', '.join(skipped[:12])}{' ...' if len(skipped) > 12 else ''}")
    return out


# ----------------------------------------------------------------------------------------------- feature table
class Builder:
    """Column layout and the code that fills the feature table of any run of consecutive impressions of a split.  Columns are ordered by group
    (scores, ctx, meta, pop), so every variant is a prefix of the table."""

    def __init__(self, S, comps, groups):
        self.S, self.comps, self.groups = S, comps, set(groups)
        data = S.data
        self.device = S.device
        n_news = data.n_news
        self.cat, self.sub = FS.code_arrays(S.meta, n_news)
        self.cat_t = torch.from_numpy(self.cat).to(self.device)
        self.sub_t = torch.from_numpy(self.sub).to(self.device)
        self.E = S.tensor(S.E0)
        self.has_meta = "meta" in self.groups
        self.has_pop = "pop" in self.groups
        if self.has_meta:                                                       # what is labelled before the split's first day: train_core for validation, + validation for test
            st = data.item_stats
            seen = {"val": np.asarray(S.seen_tr, bool), "test": np.asarray(S.seen_tr, bool) | np.asarray(S.seen_va, bool)}
            self.item = {}
            for sp, (kp, ks) in (("val", ("train_pos", "train_shown")), ("test", ("trainval_pos", "trainval_shown"))):
                pos, shown = np.asarray(st[kp], np.float64), np.asarray(st[ks], np.float64)
                self.item[sp] = np.stack([np.log1p(shown), np.log1p(pos), self.ctr(pos, shown), (shown == 0).astype(np.float64), (~seen[sp]).astype(np.float64),
                                          self.group_ctr(self.cat, pos, shown), self.group_ctr(self.sub, pos, shown)], 1).astype(np.float32)
        self.names, group_of = [], []
        for c in comps:
            self.names.append(c.key)
            group_of.append("scores")
            if c.raw:
                self.names.append(c.key + "#raw")
                group_of.append("scores")
        self.names += ["ctx_" + n for n in FS.NAMES]
        group_of += ["ctx"] * len(FS.NAMES)
        if self.has_meta:
            self.names += list(META_ITEM + META_EXPO + META_REL)
            group_of += ["meta"] * len(META_ITEM + META_EXPO + META_REL)
        if self.has_pop:
            self.names += ["pop", "pop_z"]
            group_of += ["pop", "pop"]
        self.group_of = np.array(group_of)

    @staticmethod
    def ctr(pos, shown):
        """The smoothed log-CTR of ``common.pop_scorer`` / V10's popularity, from training counts."""
        return np.log((pos + 1.0) / (shown + 20.0))

    @classmethod
    def group_ctr(cls, codes, pos, shown):
        """Smoothed log-CTR of every article's category (sub-category), pooled over the training impressions.  Stands in for the raw category code: a tree can
        memorise a code's behaviour on the one validation day it is fitted on, but not a rate pooled over the whole training period."""
        n = int(codes.max()) + 1
        gp, gs = np.bincount(codes, weights=pos, minlength=n), np.bincount(codes, weights=shown, minlength=n)
        return cls.ctr(gp, gs)[codes]

    @property
    def n_features(self):
        return len(self.names)

    def prefix(self, groups):
        """Number of leading columns that make up exactly ``groups`` (the variants are prefixes of the table)."""
        m = int(np.isin(self.group_of, list(groups)).sum())
        assert set(self.group_of[:m]) == set(g for g in groups if g in self.groups) and np.isin(self.group_of[:m], list(groups)).all(), (groups, m)
        return m

    def build(self, sp, split, a, b):
        """Feature table of the impressions ``a:b`` of ``split`` ('val' / 'test') -> float32 [rows, n_features] in the split's candidate order.
        Reads no label."""
        lptr = split.ptr[a:b + 1] - split.ptr[a]
        N = int(lptr[-1])
        X = np.zeros((N, self.n_features), np.float32)
        z = make_seg_z(lptr)
        j = 0
        for c in self.comps:
            flat = c.flat[sp][split.ptr[a]:split.ptr[b]]
            X[:, j] = z(flat)
            j += 1
            if c.raw:
                X[:, j] = flat
                j += 1
        self._ctx(X, j, split, a, b, lptr)
        j += len(FS.NAMES)
        if self.has_meta:
            it = self.item[sp][split.cand(a, b)]
            ex = split.exposure(a, b)
            X[:, j:j + len(META_ITEM)] = it
            j += len(META_ITEM)
            X[:, j:j + len(META_EXPO)] = ex
            j += len(META_EXPO)
            X[:, j], X[:, j + 1], X[:, j + 2], X[:, j + 3] = z(it[:, 2]), z(ex[:, 0]), z(ex[:, 1]), z(ex[:, 3])
            j += len(META_REL)
        if self.has_pop:
            p = split.pop(a, b)
            X[:, j], X[:, j + 1] = p, z(p)
            j += 2
        assert j == self.n_features
        return X

    def _ctx(self, X, j0, split, a, b, lptr):
        es = R.EvalSet(split.rows[a:b], device=self.device)
        k = len(FS.NAMES)
        for bt in es:
            f = FS.batch_features(bt, self.E, self.cat_t, self.sub_t)
            arr = torch.stack([f[n] for n in FS.NAMES], -1).cpu().numpy()
            rr, cc = np.nonzero(bt["cand_mask"].cpu().numpy())
            X[lptr[bt["rows"][rr]] + cc, j0:j0 + k] = arr[rr, cc]


# ----------------------------------------------------------------------------------------------- gradient-boosting backends
LGB_OBJ = {"rank:ndcg": ("lambdarank", {}),                                         # lambdarank on the top of the list (LightGBM's truncation level 30)
           "rank:pairwise": ("lambdarank", {"lambdarank_truncation_level": 300}),     # the same over (practically) the whole list: the AUC-like one
           "binary:logistic": ("binary", {})}


class Backend:
    name = "?"
    device = "cpu"

    def fit(self, X, y, g, Xe, ye, ge, objective, p):
        raise NotImplementedError

    def predict(self, model, X):
        raise NotImplementedError

    def gain(self, model, n):
        return np.zeros(n)


class XGBBackend(Backend):
    name = "xgboost"

    def __init__(self, device):
        import xgboost as xgb
        self.xgb, self.device = xgb, device
        self.major = int(xgb.__version__.split(".")[0])

    def _dm(self, X, y, g, ref=None):
        xgb = self.xgb
        try:
            d = xgb.QuantileDMatrix(X, label=y, max_bin=128, ref=ref) if ref is not None else xgb.QuantileDMatrix(X, label=y, max_bin=128)
        except Exception:
            d = xgb.DMatrix(X, label=y)
        d.set_group(g)
        return d

    def _params(self, objective, p):
        params = {"objective": objective, "eval_metric": "ndcg@10", "eta": p["eta"], "max_depth": p["depth"], "min_child_weight": 20, "subsample": 0.7,
                  "colsample_bytree": 0.7, "reg_lambda": 5.0, "tree_method": "hist", "max_bin": 128, "seed": p["seed"], "verbosity": 0}
        if self.device == "cuda":
            if self.major >= 2:
                params["device"] = "cuda"
            else:
                params["tree_method"] = "gpu_hist"
        return params

    def fit(self, X, y, g, Xe, ye, ge, objective, p):
        xgb = self.xgb
        for attempt in (0, 1):
            params = self._params(objective, p)
            try:
                dtr = self._dm(X, y, g)
                dev = self._dm(Xe, ye, ge, ref=dtr)
                bst = xgb.train(params, dtr, p["rounds"], evals=[(dev, "es")], early_stopping_rounds=p["patience"], verbose_eval=False)
                break
            except Exception as e:
                if self.device != "cuda" or attempt:                          # no usable GPU (driver / memory): the CPU histogram method
                    raise
                log(f"    xgboost on the GPU failed ({repr(e)[:120]}) -> CPU")
                self.device = "cpu"
        return bst, int(getattr(bst, "best_iteration", p["rounds"] - 1)) + 1

    def predict(self, model, X):
        bst, k = model
        try:
            return np.asarray(bst.inplace_predict(X, iteration_range=(0, k)), np.float32)
        except Exception:
            return bst.predict(self.xgb.DMatrix(X), iteration_range=(0, k)).astype(np.float32)

    def gain(self, model, n):
        out = np.zeros(n)
        for key, v in model[0].get_score(importance_type="total_gain").items():
            out[int(key[1:])] = v                                                # features are named f0, f1, ...
        return out


class LGBBackend(Backend):
    name = "lightgbm"

    def __init__(self, device="cpu"):
        import lightgbm as lgb
        self.lgb = lgb

    def fit(self, X, y, g, Xe, ye, ge, objective, p):
        lgb = self.lgb
        obj, extra = LGB_OBJ.get(objective, LGB_OBJ["rank:ndcg"])
        params = {**extra, "objective": obj, "metric": "ndcg", "eval_at": [10], "learning_rate": p["eta"],
                  "num_leaves": min(2 ** p["depth"], 127), "min_data_in_leaf": 200, "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
                  "lambda_l2": 5.0, "max_bin": 127, "verbosity": -1, "seed": p["seed"], "num_threads": os.cpu_count() or 1, "force_row_wise": True}
        dtr = lgb.Dataset(X, label=y, group=g, free_raw_data=True)
        dev = lgb.Dataset(Xe, label=ye, group=ge, reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=p["rounds"], valid_sets=[dev], callbacks=[lgb.early_stopping(p["patience"], verbose=False)])
        return bst, max(int(bst.best_iteration), 1)

    def predict(self, model, X):
        return model[0].predict(X, num_iteration=model[1]).astype(np.float32)

    def gain(self, model, n):
        return np.asarray(model[0].feature_importance(importance_type="gain"), np.float64)


class SKBackend(Backend):
    """Pointwise logistic boosting; the last resort when neither XGBoost nor LightGBM can be imported."""
    name = "sklearn"

    def __init__(self, device="cpu"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.cls = HistGradientBoostingClassifier

    def fit(self, X, y, g, Xe, ye, ge, objective, p):
        clf = self.cls(max_iter=min(p["rounds"], 300), learning_rate=p["eta"], max_depth=p["depth"], max_leaf_nodes=63, early_stopping=True,
                       validation_fraction=0.1, n_iter_no_change=p["patience"], random_state=p["seed"])
        clf.fit(X, y > 0.5)
        return clf, int(clf.n_iter_)

    def predict(self, model, X):
        return model[0].decision_function(X).astype(np.float32)


def make_backend(pref, device):
    order = {"xgboost": ["xgboost"], "lightgbm": ["lightgbm"], "sklearn": ["sklearn"], "auto": ["xgboost", "lightgbm", "sklearn"]}[pref]
    last = None
    for nm in order:
        try:
            return {"xgboost": XGBBackend, "lightgbm": LGBBackend, "sklearn": SKBackend}[nm](device)
        except Exception as e:                                                   # not installed / broken
            last = e
            log(f"  backend {nm} unavailable ({repr(e)[:100]})")
    raise RuntimeError(f"no gradient-boosting backend could be imported: {last!r}")


# ----------------------------------------------------------------------------------------------- training
def predict_rows(be, model, X, m, rows, chunk=1_500_000):
    out = np.empty(len(rows), np.float32)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = be.predict(model, X[rows[s:s + chunk], :m])
    return out


def fold_split(k, folds, esm, usable, inside):
    """Impressions of fold ``k``: ``(train, early-stopping, held-out)``.  Folds are by user, the early-stopping users are a tenth of the others', and
    only impressions with a click and a competitor teach a ranker anything."""
    held = np.flatnonzero(inside & (folds == k))
    rest = usable & (folds != k)
    tr, es = np.flatnonzero(rest & ~esm), np.flatnonzero(rest & esm)
    if len(es) < 2 or len(tr) < 2:                                                  # tiny data: half of the rest stops the training
        idx = np.flatnonzero(rest)
        tr, es = idx[::2], idx[1::2]
    return tr, es, held


def fit_fold(be, X, y, split, tr_imp, es_imp, m, objective, params, cap_rows=0):
    """Train on the impressions ``tr_imp`` (early stopping on ``es_imp``) with the first ``m`` feature columns.  A CPU backend (also an XGBoost run that
    fell back from the GPU in the middle of the stage) trains on a fixed random subset of the impressions when they hold more than ``cap_rows`` candidate rows."""
    n_rows = int(split.lens[tr_imp].sum())
    if be.device != "cuda" and cap_rows and n_rows > cap_rows:
        keep = np.random.default_rng(params["seed"]).permutation(len(tr_imp))[:max(int(len(tr_imp) * cap_rows / n_rows), 100)]
        log(f"    CPU backend: training on {len(keep):,} of {len(tr_imp):,} impressions ({cap_rows:,} candidate rows at most)")
        tr_imp = np.sort(tr_imp[keep])
    rt, re_ = expand(split.ptr, split.lens, tr_imp), expand(split.ptr, split.lens, es_imp)
    return be.fit(X[rt, :m], y[rt], split.lens[tr_imp], X[re_, :m], y[re_], split.lens[es_imp], objective, params)


def held_ndcg(S, be, model, X, m, split, held):
    """nDCG@10 (V10 evaluator) of a model over the impressions ``held`` of the validation table."""
    flat = np.zeros(split.N, np.float32)
    rows = expand(split.ptr, split.lens, held)
    flat[rows] = predict_rows(be, model, X, m, rows)
    es = common.subset_evalset(split.rows, held, S.device)
    return S.run(common.ragged_scorer(ragged_of(split, flat, held), held, S.device), "val", es={"val": es})["mean"]["ndcg@10"]


# ----------------------------------------------------------------------------------------------- the stage
def stage_stack(S, args):
    log("\n== T stacker: a gradient-boosted ranker over the LLM-side scores + label-free features, trained on validation (K-fold by user), test touched once ==")
    t_all = time.time()
    data = S.data
    if getattr(args, "stack_free_memory", 0):                                       # the stacker is the last consumer of the training samples and the retrieval memories
        S.kv["train_samples"] = {"used": len(data.train_core), "all": len(getattr(data, "train_core_all", None) or data.train_core)}
        for k in ("um_pack", "memo", "comm", "entity_A"):
            S.store.pop(k, None)
        data.train_core, data.train_core_all = [], None
        gc.collect()
        log(f"  freed the training samples and retrieval memories ({common.mem_note() or 'memory not reported'})")
    val, test = Split(data.validation, getattr(data, "aux_validation", None)), Split(data.test, getattr(data, "aux_test", None))
    dirs = [S.work] + [d for d in args.super_dirs.split(",") if d]
    comps = load_components(dirs, val, test, baselines=[x for x in args.stack_baselines.split(",") if x])
    if not comps:
        log("  no exported component scores cover the whole split (stages not run, or --export-scores 0): stacker not run")
        return False
    have_meta = getattr(data, "item_stats", None) is not None and val.aux is not None and test.aux is not None
    wanted = [x for x in args.stack_variants.split(",") if x]
    variants = {k: v for k, v in VARIANTS.items() if k.split("_", 1)[1] in wanted}
    if not have_meta:
        dropped = [k for k, v in variants.items() if "meta" in v and "pop" not in v]
        variants = {k: tuple(g for g in v if g != "meta") for k, v in variants.items() if "meta" not in v or "pop" in v}
        log(f"  no exposure features / item counts on the data (the lean loader builds them: --lean-data 1): {dropped or 'the meta group'} left out")
    if not variants:
        log("  no variant to run")
        return False
    groups_used = sorted({g for v in variants.values() for g in v}, key=GROUPS.index)
    be = make_backend(args.stack_backend, S.device)
    log(f"  backend: {be.name}{' (GPU)' if be.device == 'cuda' else ''}; {len(comps)} components from {len(dirs)} run(s); feature groups {groups_used}; variants {list(variants)}")
    bld = Builder(S, comps, groups_used)
    log(f"  {bld.n_features} features: " + ", ".join(f"{g}={int((bld.group_of == g).sum())}" for g in groups_used))
    if S.device == "cuda":
        torch.cuda.empty_cache()

    # ---- training table: the validation split
    n = val.n
    t0 = time.time()
    X = bld.build("val", val, 0, n)
    y = val.labels(0, n)
    for c in comps:
        c.flat["val"] = None                                                         # the validation columns live in X now
    log(f"  validation table: {X.shape[0]:,} rows x {X.shape[1]} ({X.nbytes / 1e9:.1f} GB), {time.time() - t0:.0f}s")
    folds = user_folds(val.users, args.stack_folds, args.stack_seed)
    esm = es_mask(val.users, args.stack_seed)
    base = {"eta": args.stack_eta, "depth": args.stack_depth, "rounds": args.stack_rounds, "patience": 50, "seed": args.stack_seed}
    inside = np.ones(n, bool)
    usable = inside & np.array([float(np.sum(r[4])) > 0 for r in val.rows]) & (val.lens > 1)       # no click or no competitor: nothing to learn a ranking from
    state = {"backend": be.name, "n_features": bld.n_features, "components": [c.key for c in comps], "groups": groups_used, "folds": args.stack_folds,
             "train_impressions": int(usable.sum()), "variants": {}}
    S.kv["stack"] = state

    split_fold = lambda k: fold_split(k, folds, esm, usable, inside)

    # ---- objective: chosen on the held-out part of fold 0 with the richest label-free variant
    objectives = [o for o in args.stack_objectives.split(",") if o] or ["rank:ndcg"]
    main_var = "stack_meta" if "stack_meta" in variants else next(iter(variants))
    m_main = bld.prefix(variants[main_var])
    objective = objectives[0]
    if len(objectives) > 1:
        tr, es, held = split_fold(0)
        res = {}
        for o in objectives:
            if S.time_left() < 900:
                break
            t1 = time.time()
            fm = fit_fold(be, X, y, val, tr, es, m_main, o, base, args.stack_max_rows_cpu)        # (model, number of rounds kept)
            res[o] = held_ndcg(S, be, fm, X, m_main, val, held)
            log(f"    objective {o}: held-out fold 0 nDCG@10 = {res[o]:.4f} ({fm[1]} rounds, {time.time() - t1:.0f}s)")
            del fm
        if res:
            objective = max(res, key=res.get)
        state["objective_scan"] = {k: float(v) for k, v in res.items()}
    state["objective"] = objective
    log(f"  objective: {objective}")

    # ---- K-fold training of every variant
    models, oof = {}, {}
    for vname, grp in variants.items():
        if S.time_left() < 900:
            log(f"  {vname}: skipped (wall-clock budget)")
            continue
        m = bld.prefix(grp)
        ms, rounds_, gain = [], [], np.zeros(m)
        flat_oof = np.zeros(val.N, np.float32)
        have = np.zeros(n, bool)
        t1 = time.time()
        for k in range(args.stack_folds):
            tr, es, held = split_fold(k)
            fm = fit_fold(be, X, y, val, tr, es, m, objective, base, args.stack_max_rows_cpu)
            ms.append(fm)
            rounds_.append(fm[1])
            gain += be.gain(fm, m)
            rows = expand(val.ptr, val.lens, held)
            flat_oof[rows] = predict_rows(be, fm, X, m, rows)
            have[held] = True
        models[vname] = (ms, m)
        oof[vname] = (flat_oof, have)
        share = gain / max(gain.sum(), 1e-12)
        gshare = {g: float(share[bld.group_of[:m] == g].sum()) for g in grp}
        top = sorted(((bld.names[c], float(s)) for c, s in enumerate(share)), key=lambda x: -x[1])[:12]
        state["variants"][vname] = {"rounds": rounds_, "group_gain_share": gshare, "top_features": top, "fit_s": round(time.time() - t1)}
        log(f"  {vname}: {args.stack_folds} folds, {be.name} {objective}, rounds {rounds_}, gain by group " + ", ".join(f"{g}={s:.2f}" for g, s in gshare.items())
            + f" ({time.time() - t1:.0f}s)")
        log("    top features: " + ", ".join(f"{nm}={s:.3f}" for nm, s in top[:8]))
    del X, y
    gc.collect()
    if not models:
        log("  no variant could be trained (wall-clock budget)")
        return False

    # ---- test: built chunk by chunk (never held whole), predicted by every fold model of every variant
    t0 = time.time()
    preds = {v: [np.zeros(test.N, np.float32) for _ in ms] for v, (ms, _) in models.items()}
    step = max(int(args.stack_chunk), 1)
    for a in range(0, test.n, step):
        b = min(a + step, test.n)
        Xc = bld.build("test", test, a, b)
        r0, r1 = test.ptr[a], test.ptr[b]
        for v, (ms, m) in models.items():
            for k, mod in enumerate(ms):
                preds[v][k][r0:r1] = be.predict(mod, Xc[:, :m])
        del Xc
    log(f"  test features + predictions: {test.N:,} candidate rows, {time.time() - t0:.0f}s")
    gc.collect()
    ref, ref_pop = ("frozen" if "frozen" in S.cache else None), ("frozen+pop" if "frozen+pop" in S.cache else None)
    zt = make_seg_z(test.ptr)
    final = {}
    for vname in models:
        flat_oof, have = oof[vname]
        sub = np.flatnonzero(have)
        es_val = common.subset_evalset(val.rows, sub, S.device) if len(sub) != n else S.es["val"]
        vres = S.run(common.ragged_scorer(ragged_of(val, flat_oof, sub), sub, S.device), "val", es={"val": es_val})
        final[vname] = np.mean([zt(p) for p in preds[vname]], 0).astype(np.float32)
        tres = S.run(common.ragged_scorer({"ids": np.arange(test.n), "ptr": test.ptr, "flat": final[vname]}, np.arange(test.n), S.device), "test")
        pop = "pop" in variants[vname]
        cfg = {"backend": be.name, "objective": objective, "folds": args.stack_folds, "groups": list(variants[vname]), "features": int(models[vname][1]),
               "components": [c.key for c in comps], "fit": "validation only, K-fold by user; val = out-of-fold, test = mean over the fold models"}
        row = S.record(vname, "STACK+pop" if pop else "STACK", cfg, vres, tres, ref_pop if pop else ref,
                       {"rounds": state["variants"][vname]["rounds"], "n_features": cfg["features"]})
        state["variants"][vname].update({"val_ndcg10_oof": row["val_ndcg@10"], "test_ndcg10": row["ndcg@10"], "test_auc": row["auc"]})
    if args.stack_save_scores:                                                       # the scores themselves, for later analysis (and the tests)
        out = {"val_ptr": val.ptr, "test_ptr": test.ptr, "val_users": val.users, "folds": folds}
        for vname in models:
            out[f"{vname}__val_oof"], out[f"{vname}__val_have"] = oof[vname]
            out[f"{vname}__test"] = final[vname]
            if args.stack_save_scores > 1:
                for k, p in enumerate(preds[vname]):
                    out[f"{vname}__test_fold{k}"] = p
        np.savez_compressed(os.path.join(S.work, "stack_scores.npz"), **out)
    del preds
    gc.collect()
    state["seconds"] = round(time.time() - t_all)
    return True
