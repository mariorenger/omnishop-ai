"""Stacker (v15, v16): a gradient-boosted ranker over the LLM-side scores and a few label-free features, trained on the validation split.

Why.  ``supermodel.py`` fuses the components linearly with ONE global weight per component, fitted on validation.  Measured on MIND-large
(``RESULTS_v14_large.md``) that validation split is warmer than the test split (82% of the test clicks are on articles no training row showed):
the weights over-trust what wins on warm articles (the trained readers, RAG) and the fused model lost to a smaller one on test.  A tree ensemble sees
how cold / how new / how widely shown a candidate is next to the scores, so it can weigh them differently per situation instead of once for all.

What it reads, per candidate (nothing here reads a test label before the final evaluation):

  ``scores``  every exported component that covers the whole validation and test split -- of this run and of the runs in ``--super-dirs`` (a second
              encoder, the composite one): zero-shot scorers, the learned readers (``um_*``), the history-free click prior (``um_prior_*``) and the
              auxiliary scorers (plain mean-pool, top-k, max-sim, RAG ...) -- as the z-score inside the impression, plus the raw value for a handful of scorers;
  ``ctx``     history / slate features of the frozen vectors (``featscore.py``), and (v16) the overlap of the candidate's entities with the entities of the
              reader's clicks (IDF-weighted; all clicks and the three most recent);
  ``meta``    counts of the article over everything labelled before the split's first day (train_core for validation impressions, train_core + the
              validation days for test ones: shown / clicked / smoothed CTR, "never shown", "unseen" as the cold/warm slices define it) and the
              pooled CTR of its category and sub-category,
              and its exposure in the impression log -- hours since it first appeared in an EARLIER impression, how many earlier impressions showed it,
              and how many showed it in the last 1 / 6 / 24 completed hours (label-free; ``data_lean.exposure_features_lean``);
  ``pop``     V10's online smoothed log-CTR -- counts the clicks of earlier impressions of the same day, i.e. streaming feedback -- and (v16) the same
              click-through over the last 1 / 6 / 24 completed hours; kept apart.

Variants: ``stack_llm`` = scores + ctx, ``stack_meta`` = + meta (all of it available offline: the one comparable with leaderboards), ``stack_pop`` = + pop.

Protocol.  The ranker is trained on the VALIDATION impressions only; the out-of-fold scores are the validation result; the test result is the mean over
the fold models of their per-impression z-scored predictions, computed once at the end.  The test split is never used to choose anything.

v16 changes, all of them answers to what the v15 run measured (``RESULTS_v15_large.md`` 2.2): the out-of-fold validation score was 0.03-0.04 nDCG@10 above
the test score (the linear fusion had test >= validation), the objectives were indistinguishable on the held-out fold and ``ctx_n_cand`` -- constant inside
an impression -- carried the largest share of the gain.  Hypothesis (to be tested by this very run, see the report): with folds by user the model is scored
on articles whose clicks of the SAME day it saw from other users, which the test day does not offer.  Hence

  * folds are contiguous TIME BLOCKS of the validation day (``--stack-cv time``), with a buffer on both sides of the held-out block; early stopping uses the
    training impressions nearest in time to it;
  * what the model is stopped and CHOSEN on is the nDCG@10 of the impressions whose clicked article is NEW to the stacker's training impressions
    (``new_fraction``): the situation of the 82% cold test clicks.  Configurations (``CONFIGS``: deep trees on every feature as in v15, columns that are
    constant inside an impression dropped and near-identifying ones coarsened, shallower trees, without the history-free prior) are compared once on a
    FORWARD holdout (first 60% of the day train, the next 10% stop, the last ~30% evaluate); the best ``--stack-blend`` are trained in every fold and averaged;
  * the v15 recipe (user folds, every feature) is trained next to it for the 'meta' variant (row ``stack_meta_v15cfg``), so the effect of the changes is a
    paired comparison on the same components.

Backends: XGBoost (GPU when there is one) -> LightGBM -> scikit-learn.
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
POPW_NAMES = ("popw_ctr_1h", "popw_ctr_6h", "popw_ctr_24h", "popw_clk_1h", "popw_clk_6h", "popw_clk_24h", "popw_ctr_6h_z")    # v16: click-through of the last 1 / 6 / 24 completed hours

# Stacker configurations compared on the forward holdout.  ``clean``: drop the columns that are constant inside an impression (``DROP_CLEAN``: a ranking cannot use them
# except through interactions, and a pointwise objective spends its trees on the per-impression base rate) and coarsen the columns that, at full resolution, pin down
# one article on one day (``COARSE``); ``noprior``: also leave out the history-free click-prior readers (an article-level score); ``depth`` / ``mcw``: tree depth and
# minimum child weight (XGBoost; LightGBM's min_data_in_leaf is 10 x mcw).
CONFIGS = {"v15": {"clean": False, "noprior": False, "depth": 7, "mcw": 20},
           "clean": {"clean": True, "noprior": False, "depth": 7, "mcw": 20},
           "clean_d5": {"clean": True, "noprior": False, "depth": 5, "mcw": 100},
           "clean_d3": {"clean": True, "noprior": False, "depth": 3, "mcw": 300},
           "noprior_d5": {"clean": True, "noprior": True, "depth": 5, "mcw": 100}}
DROP_CLEAN = ("ctx_n_cand",)
MIN_NEW_EVAL = 200                                       # evaluation impressions with a new-article click needed to choose configurations on that slice
COARSE = {"m_log_shown": ("lin", 0.25), "m_log_pos": ("lin", 0.25), "m_ctr": ("lin", 0.1), "m_age_h": ("log1p", 0.25), "m_log_prior_shown": ("lin", 0.25),
          "m_expo_1h": ("lin", 0.25), "m_expo_6h": ("lin", 0.25), "m_expo_24h": ("lin", 0.25), "pop": ("lin", 0.05),
          "popw_ctr_1h": ("lin", 0.1), "popw_ctr_6h": ("lin", 0.1), "popw_ctr_24h": ("lin", 0.1),
          "popw_clk_1h": ("lin", 0.25), "popw_clk_6h": ("lin", 0.25), "popw_clk_24h": ("lin", 0.25)}      # column -> (transform, step the value is rounded to)
REL_SRC = {"m_ctr_z": "m_ctr", "m_age_z": "m_age_h", "m_expo_z": "m_log_prior_shown", "m_expo6_z": "m_expo_6h", "pop_z": "pop", "popw_ctr_6h_z": "popw_ctr_6h"}   # z-columns follow their source


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
    def __init__(self, rows, aux=None, popw=None, times=None):
        """``aux``: exposure arrays per impression; ``popw``: windowed click-through arrays per impression (``data_lean.popwin_features_lean``, streaming);
        ``times``: impression times in seconds (None: the order of the rows stands for time)."""
        self.rows, self.aux, self.pw = rows, aux, popw
        self.n = len(rows)
        self.lens = np.fromiter((len(r[3]) for r in rows), np.int64, self.n)
        self.ptr = np.concatenate([[0], np.cumsum(self.lens)]).astype(np.int64)
        self.N = int(self.ptr[-1])
        self.users = np.fromiter((r[1] for r in rows), np.int64, self.n)
        self.times = None if times is None else np.asarray(times, np.int64)
        self._flat = None

    def flat_cand_label(self):
        """All candidates and labels in the split's flat layout (built once)."""
        if self._flat is None:
            self._flat = (self.cand(0, self.n), self.labels(0, self.n))
        return self._flat

    def popw(self, a, b):
        if self.pw is None:
            raise ValueError("this split carries no windowed click-through (build it with Split(rows, aux, popw, times) from data.aux_pop_*)")
        return np.concatenate(self.pw[a:b]).astype(np.float32) if b > a else np.zeros((0, 6), np.float32)

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
        self.ctx_names = list(FS.NAMES)
        self.ent = None
        ents = S.meta.get("ents") if isinstance(S.meta, dict) else None
        if ents is not None and len(ents) and any(len(e) for e in ents):                 # entity overlap of the candidate with the reader's clicks
            idx, idf = FS.entity_tables(ents, n_news)
            self.ent = (torch.from_numpy(idx).to(self.device), torch.from_numpy(idf).to(self.device))
            self.ctx_names += list(FS.ENT_NAMES)
        self.has_meta = "meta" in self.groups
        self.has_pop = "pop" in self.groups
        self.has_popw = self.has_pop and getattr(data, "aux_pop_validation", None) is not None and getattr(data, "aux_pop_test", None) is not None
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
        self.names += ["ctx_" + n for n in self.ctx_names]
        group_of += ["ctx"] * len(self.ctx_names)
        if self.has_meta:
            self.names += list(META_ITEM + META_EXPO + META_REL)
            group_of += ["meta"] * len(META_ITEM + META_EXPO + META_REL)
        if self.has_pop:
            self.names += ["pop", "pop_z"] + (list(POPW_NAMES) if self.has_popw else [])
            group_of += ["pop"] * (2 + (len(POPW_NAMES) if self.has_popw else 0))
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
        j += len(self.ctx_names)
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
            if self.has_popw:
                pw = split.popw(a, b)
                X[:, j:j + 6] = pw
                X[:, j + 6] = z(pw[:, 1])
                j += len(POPW_NAMES)
        assert j == self.n_features
        return X

    def _ctx(self, X, j0, split, a, b, lptr):
        es = R.EvalSet(split.rows[a:b], device=self.device)
        k = len(self.ctx_names)
        for bt in es:
            f = FS.batch_features(bt, self.E, self.cat_t, self.sub_t)
            if self.ent is not None:
                f.update(FS.batch_entity_features(bt, *self.ent))
            arr = torch.stack([f[n] for n in self.ctx_names], -1).cpu().numpy()
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
        params = {"objective": objective, "eval_metric": "ndcg@10", "eta": p["eta"], "max_depth": p["depth"], "min_child_weight": p.get("mcw", 20), "subsample": 0.7,
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
                  "num_leaves": min(2 ** p["depth"], 127), "min_data_in_leaf": int(10 * p.get("mcw", 20)), "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
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


# ----------------------------------------------------------------------------------------------- columns: cleaning and configurations
def coarsen(X, names, ptr):
    """In place: round the near-identifying columns of a feature table (``COARSE``: log-scale counts and ages to a quarter of a log unit, the smoothed CTRs to a
    tenth, ...) and recompute the within-impression z-columns that follow them (``REL_SRC``) from the rounded values.  At full resolution these columns pin one
    article down on one day (age since its first appearance to the minute, how often it was shown to the last count), which lets a tree memorise that article's
    clicks of the day from the other users' impressions; rounded, the same columns still say 'new / old / widely shown'."""
    idx = {n: j for j, n in enumerate(names)}
    for nm, (kind, step) in COARSE.items():
        j = idx.get(nm)
        if j is None:
            continue
        v = X[:, j].astype(np.float64)
        if kind == "log1p":
            v = np.log1p(np.maximum(v, 0.0))
        X[:, j] = (np.round(v / step) * step).astype(np.float32)
    z = make_seg_z(ptr)
    for zn, src in REL_SRC.items():
        if zn in idx and src in idx:
            X[:, idx[zn]] = z(X[:, idx[src]])
    return X


class ColumnSwap:
    """One feature table serves every configuration: keeps the full-resolution version of the columns ``coarsen`` changes and switches the table between
    'hi' (as built) and 'lo' (coarsened) in place."""

    def __init__(self, X, names, ptr):
        idx = {n: j for j, n in enumerate(names)}
        self.cols = np.array(sorted({idx[n] for n in list(COARSE) + list(REL_SRC) if n in idx}), np.int64)
        self.hi = X[:, self.cols].copy() if len(self.cols) else None
        self.names, self.ptr, self.mode = names, ptr, "hi"

    def set(self, X, mode):
        if mode == self.mode or self.hi is None:
            self.mode = mode
            return
        X[:, self.cols] = self.hi
        if mode == "lo":
            coarsen(X, self.names, self.ptr)
        self.mode = mode


def config_columns(bld, m, cfg):
    """Indices (< m) of the columns a configuration trains on."""
    drop = set(DROP_CLEAN) if cfg["clean"] else set()
    keep = [j for j in range(m) if bld.names[j] not in drop and not (cfg.get("noprior") and ":um_prior" in bld.names[j])]
    return np.asarray(keep, np.int64)


def _as_cols(cols):
    return np.arange(int(cols), dtype=np.int64) if np.isscalar(cols) else np.asarray(cols, np.int64)


# ----------------------------------------------------------------------------------------------- folds, forward holdout, NEW-article weights
def time_folds(times, k):
    """Block id (0..k-1) of every impression: k contiguous blocks of (nearly) equal size along the time axis."""
    n = len(times)
    order = np.argsort(np.asarray(times), kind="stable")
    block = np.empty(n, np.int64)
    block[order] = np.minimum((np.arange(n) * k) // max(n, 1), k - 1)
    return block


def buffer_seconds(times, minutes):
    """The buffer left out of the training part on each side of a held-out time block: ``minutes``, but at most 2% of the span of the split."""
    span = float(np.max(times) - np.min(times)) if len(times) else 0.0
    return float(min(minutes * 60.0, 0.02 * span))


def time_fold_split(k, block, times, usable, buffer_s, es_frac=0.12):
    """Impressions of time block ``k``: ``(train, early-stopping, held-out)``.  Held-out = the block; the training pool = the usable impressions of the other blocks
    that lie at least ``buffer_s`` outside it; early stopping = the ``es_frac`` of that pool NEAREST in time to the block (what the model must predict is the next
    stretch of time, not a random subset of what it trained on)."""
    held = np.flatnonzero(block == k)
    t_lo, t_hi = times[held].min(), times[held].max()
    pool = np.flatnonzero(usable & (block != k) & ((times <= t_lo - buffer_s) | (times >= t_hi + buffer_s)))
    dist = np.where(times[pool] < t_lo, t_lo - times[pool], times[pool] - t_hi)
    order = np.argsort(dist, kind="stable")
    n_es = int(len(pool) * es_frac)
    es, tr = np.sort(pool[order[:n_es]]), np.sort(pool[order[n_es:]])
    if len(es) < 2 or len(tr) < 2:                                                  # tiny data: half of the pool stops the training
        tr, es = pool[::2], pool[1::2]
    return tr, es, held


def forward_split(times, usable, buffer_s, fracs=(0.6, 0.1)):
    """``(train, early-stopping, evaluation)`` impressions of the forward holdout: train on the first 60% of the split in time, stop on the next 10%, evaluate on the
    rest -- each part ``buffer_s`` apart from the previous one.  Tiny data: three consecutive thirds of the order."""
    c1, c2 = np.quantile(times, [fracs[0], fracs[0] + fracs[1]])
    tr = np.flatnonzero(usable & (times < c1 - buffer_s))
    es = np.flatnonzero(usable & (times >= c1) & (times < c2 - buffer_s))
    ev = np.flatnonzero(times >= c2)
    if len(tr) < 20 or len(es) < 5 or len(ev) < 5:
        order = np.argsort(times, kind="stable")
        n = len(order)
        u = usable[order]
        tr, es, ev = order[:n // 2][u[:n // 2]], order[n // 2:(3 * n) // 4][u[n // 2:(3 * n) // 4]], order[(3 * n) // 4:]
        tr, es, ev = np.sort(tr), np.sort(es), np.sort(ev)
    return tr, es, ev


def shown_in(split, imp, n_news):
    """bool over article ids: the article is in a candidate list of the impressions ``imp``."""
    cand, _ = split.flat_cand_label()
    return np.bincount(cand[expand(split.ptr, split.lens, np.asarray(imp, np.int64))], minlength=n_news) > 0


def new_fraction(split, imp, shown):
    """For every impression of ``imp``: the share of its clicked articles that are NOT in ``shown`` (articles of the stacker's training impressions) -- 1.0 when every
    click is on an article the model never saw in a training slate, 0.0 for an impression without a click.  The test day's clicks are 82% such articles."""
    imp = np.asarray(imp, np.int64)
    cand, lab = split.flat_cand_label()
    rows = expand(split.ptr, split.lens, imp)
    seg = np.repeat(np.arange(len(imp)), split.lens[imp])
    pos = lab[rows] > 0.5
    new = pos & ~shown[cand[rows]]
    npos = np.bincount(seg[pos], minlength=len(imp))
    nnew = np.bincount(seg[new], minlength=len(imp))
    return np.where(npos > 0, nnew / np.maximum(npos, 1), 0.0)


def restrict_new(split, imp, shown, min_n=50, min_frac=0.2):
    """The impressions of ``imp`` with a click on an article not in ``shown``, if they are enough to stop a training on (else ``imp``)."""
    imp = np.asarray(imp, np.int64)
    if len(imp) == 0:
        return imp
    keep = imp[new_fraction(split, imp, shown) > 0]
    return keep if len(keep) >= max(min_n, int(min_frac * len(imp))) else imp


def make_folds(cv, K, seed, val, usable, times, n_news, buffer_s):
    """``(specs, fold_of, w_new)``: ``specs[k] = (train, early-stopping, held-out)`` impression indices, ``fold_of`` the fold of every impression, ``w_new`` the share
    of each impression's clicks that are on articles neither the training nor the early-stopping part of its fold showed (the 'new article' weight).  ``cv`` =
    'time' (contiguous blocks, ``time_fold_split``) or 'user' (hash of the user, v15's ``fold_split``).  The early-stopping part is cut down to impressions with a
    click on an article the training part never showed whenever there are enough of them."""
    n = val.n
    if cv == "user":
        fold_of = user_folds(val.users, K, seed)
        esm, inside = es_mask(val.users, seed), np.ones(n, bool)
        raw = [fold_split(k, fold_of, esm, usable, inside) for k in range(K)]
    else:
        fold_of = time_folds(times, K)
        raw = [time_fold_split(k, fold_of, times, usable, buffer_s) for k in range(K)]
    specs, w_new = [], np.zeros(n)
    for tr, es, held in raw:
        es_new = restrict_new(val, es, shown_in(val, tr, n_news))
        w_new[held] = new_fraction(val, held, shown_in(val, np.concatenate([tr, es]), n_news))
        specs.append((tr, es_new, held))
    return specs, fold_of, w_new


def eval_flat(S, val, flat, imp):
    """V10-evaluator results (per-impression arrays included) of per-candidate scores ``flat`` (the validation split's layout) on the impressions ``imp``."""
    imp = np.asarray(imp, np.int64)
    es = S.es["val"] if len(imp) == val.n else common.subset_evalset(val.rows, imp, S.device)
    return S.run(common.ragged_scorer(ragged_of(val, flat, imp), imp, S.device), "val", es={"val": es})


def new_metric(res, w_new):
    """``(nDCG@10 averaged with the NEW-article weights, total weight)`` of an ``eval_flat`` result; without any new-article click the plain mean and weight 0."""
    w = w_new[res["ids"]]
    if float(w.sum()) <= 0:
        return float(res["mean"]["ndcg@10"]), 0.0
    return common.weighted_mean(res["arr"]["ndcg@10"], w), float(w.sum())


# ----------------------------------------------------------------------------------------------- training
def predict_rows(be, model, X, cols, rows, chunk=1_500_000):
    """Scores of the table rows ``rows`` with the columns ``cols`` (an int = the first columns)."""
    cols = _as_cols(cols)
    out = np.empty(len(rows), np.float32)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = be.predict(model, X[np.ix_(rows[s:s + chunk], cols)])
    return out


def fold_split(k, folds, esm, usable, inside):
    """Impressions of fold ``k`` (folds by user, v15): ``(train, early-stopping, held-out)``.  The early-stopping users are a tenth of the others', and
    only impressions with a click and a competitor teach a ranker anything."""
    held = np.flatnonzero(inside & (folds == k))
    rest = usable & (folds != k)
    tr, es = np.flatnonzero(rest & ~esm), np.flatnonzero(rest & esm)
    if len(es) < 2 or len(tr) < 2:                                                  # tiny data: half of the rest stops the training
        idx = np.flatnonzero(rest)
        tr, es = idx[::2], idx[1::2]
    return tr, es, held


def fit_cols(be, X, y, split, tr_imp, es_imp, cols, objective, params, cap_rows=0):
    """Train on the impressions ``tr_imp`` (early stopping on ``es_imp``) with the columns ``cols``.  A CPU backend (also an XGBoost run that fell back from the GPU
    in the middle of the stage) trains on a fixed random subset of the impressions when they hold more than ``cap_rows`` candidate rows."""
    cols = _as_cols(cols)
    n_rows = int(split.lens[tr_imp].sum())
    if be.device != "cuda" and cap_rows and n_rows > cap_rows:
        keep = np.random.default_rng(params["seed"]).permutation(len(tr_imp))[:max(int(len(tr_imp) * cap_rows / n_rows), 100)]
        log(f"    CPU backend: training on {len(keep):,} of {len(tr_imp):,} impressions ({cap_rows:,} candidate rows at most)")
        tr_imp = np.sort(tr_imp[keep])
    rt, re_ = expand(split.ptr, split.lens, tr_imp), expand(split.ptr, split.lens, es_imp)
    return be.fit(X[np.ix_(rt, cols)], y[rt], split.lens[tr_imp], X[np.ix_(re_, cols)], y[re_], split.lens[es_imp], objective, params)


def fit_fold(be, X, y, split, tr_imp, es_imp, m, objective, params, cap_rows=0):
    """v15 signature: the first ``m`` columns."""
    return fit_cols(be, X, y, split, tr_imp, es_imp, m, objective, params, cap_rows)


def held_ndcg(S, be, model, X, m, split, held):
    """nDCG@10 (V10 evaluator) of a model over the impressions ``held`` of the validation table."""
    flat = np.zeros(split.N, np.float32)
    rows = expand(split.ptr, split.lens, held)
    flat[rows] = predict_rows(be, model, X, m, rows)
    return eval_flat(S, split, flat, held)["mean"]["ndcg@10"]


# ----------------------------------------------------------------------------------------------- the stage
def stage_stack(S, args):
    log("\n== T stacker: a gradient-boosted ranker over the LLM-side scores + label-free features, trained on validation "
        f"(folds: {args.stack_cv}; configurations chosen on a forward holdout), test touched once ==")
    t_all = time.time()
    data = S.data
    if getattr(args, "stack_free_memory", 0):                                       # the stacker is the last consumer of the training samples and the retrieval memories
        S.kv["train_samples"] = {"used": len(data.train_core), "all": len(getattr(data, "train_core_all", None) or data.train_core)}
        for k in ("um_pack", "memo", "comm", "entity_A"):
            S.store.pop(k, None)
        data.train_core, data.train_core_all = [], None
        gc.collect()
        log(f"  freed the training samples and retrieval memories ({common.mem_note() or 'memory not reported'})")
    val = Split(data.validation, getattr(data, "aux_validation", None), getattr(data, "aux_pop_validation", None), getattr(data, "time_validation", None))
    test = Split(data.test, getattr(data, "aux_test", None), getattr(data, "aux_pop_test", None), getattr(data, "time_test", None))
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
    cfg_names = [c for c in args.stack_configs.split(",") if c] or ["clean"]
    unknown = [c for c in cfg_names if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"--stack-configs: unknown {unknown}; choose from {sorted(CONFIGS)}")
    objectives = [o for o in args.stack_objectives.split(",") if o] or ["binary:logistic"]
    groups_used = sorted({g for v in variants.values() for g in v}, key=GROUPS.index)
    be = make_backend(args.stack_backend, S.device)
    log(f"  backend: {be.name}{' (GPU)' if be.device == 'cuda' else ''}; {len(comps)} components from {len(dirs)} run(s); feature groups {groups_used}; variants {list(variants)}")
    bld = Builder(S, comps, groups_used)
    log(f"  {bld.n_features} features: " + ", ".join(f"{g}={int((bld.group_of == g).sum())}" for g in groups_used)
        + f" (entity overlap {'on' if bld.ent is not None else 'off: no entities in the article metadata'}; windowed click-through {'on' if bld.has_popw else 'off'})")
    if S.device == "cuda":
        torch.cuda.empty_cache()

    # ---- training table: the validation split
    n = val.n
    n_news = data.n_news
    t0 = time.time()
    X = bld.build("val", val, 0, n)
    y = val.labels(0, n)
    for c in comps:
        c.flat["val"] = None                                                         # the validation columns live in X now
    log(f"  validation table: {X.shape[0]:,} rows x {X.shape[1]} ({X.nbytes / 1e9:.1f} GB), {time.time() - t0:.0f}s")
    times = val.times if val.times is not None else np.arange(n, dtype=np.int64)    # without time stamps the order of the rows stands for time
    buf = buffer_seconds(times, args.stack_buffer_min)
    swap = ColumnSwap(X, bld.names, val.ptr)
    base = {"eta": args.stack_eta, "rounds": args.stack_rounds, "patience": 50, "seed": args.stack_seed}
    cap = args.stack_max_rows_cpu
    usable = np.array([float(np.sum(r[4])) > 0 for r in val.rows]) & (val.lens > 1)  # no click or no competitor: nothing to learn a ranking from
    state = {"backend": be.name, "n_features": bld.n_features, "components": [c.key for c in comps], "groups": groups_used, "folds": args.stack_folds, "cv": args.stack_cv,
             "train_impressions": int(usable.sum()), "buffer_s": buf, "variants": {}, "cold_weights": S.kv.get("cold_weights"),
             "entity_features": bld.ent is not None, "windowed_ctr": bool(bld.has_popw)}
    S.kv["stack"] = state
    S.val_weights()

    # ---- (1) configurations compared once, on the forward holdout
    main_var = "stack_meta" if "stack_meta" in variants else next(iter(variants))
    m_main = bld.prefix(variants[main_var])
    pairs = [(c, o) for c in cfg_names for o in objectives]
    members = [{"config": pairs[0][0], "objective": pairs[0][1]}]
    if len(pairs) > 1:
        tr_f, es_f, ev_f = forward_split(times, usable, buf)
        es_use = restrict_new(val, es_f, shown_in(val, tr_f, n_news))
        w_new_f = np.zeros(n)
        w_new_f[ev_f] = new_fraction(val, ev_f, shown_in(val, np.concatenate([tr_f, es_f]), n_news))
        n_new = int((w_new_f[ev_f] > 0).sum())
        use_new = n_new >= MIN_NEW_EVAL                                              # too few new-article clicks to rank configurations on: the plain mean of all evaluation impressions
        state["forward"] = {"train": int(len(tr_f)), "stop": int(len(es_use)), "eval": int(len(ev_f)), "eval_with_new_click": n_new, "variant": main_var}
        state["selection_metric"] = ("nDCG@10 of the evaluation impressions weighted by their share of clicks on articles new to the stacker" if use_new
                                     else f"plain nDCG@10 of all evaluation impressions (only {n_new} with a new-article click, fewer than {MIN_NEW_EVAL})")
        log(f"  forward holdout on '{main_var}': train {len(tr_f):,} / stop {len(es_use):,} / evaluate {len(ev_f):,} impressions ({n_new:,} with a click on an article the model never saw)")
        sel = []
        for cn, obj in pairs:
            if S.time_left() < 900:
                log("  configuration scan stopped (wall-clock budget)")
                break
            cfg = CONFIGS[cn]
            swap.set(X, "lo" if cfg["clean"] else "hi")
            cols = config_columns(bld, m_main, cfg)
            t1 = time.time()
            fm = fit_cols(be, X, y, val, tr_f, es_use, cols, obj, {**base, "depth": cfg["depth"], "mcw": cfg["mcw"]}, cap)
            flat = np.zeros(val.N, np.float32)
            rows = expand(val.ptr, val.lens, ev_f)
            flat[rows] = predict_rows(be, fm, X, cols, rows)
            res = eval_flat(S, val, flat, ev_f)
            nd_new, w_sum = new_metric(res, w_new_f)
            score = nd_new if use_new else float(res["mean"]["ndcg@10"])
            sel.append({"config": cn, "objective": obj, "score": score, "new_ndcg10": nd_new, "ndcg10": float(res["mean"]["ndcg@10"]), "rounds": int(fm[1]), "features": int(len(cols)),
                        "fit_s": round(time.time() - t1)})
            log(f"    {cn:11s} {obj:16s} forward holdout: nDCG@10 on new-article clicks {nd_new:.4f} (all impressions {sel[-1]['ndcg10']:.4f}), {fm[1]} rounds, {len(cols)} columns, {time.time() - t1:.0f}s")
            del fm
        if sel:
            members = sorted(sel, key=lambda r: -r["score"])[:max(int(args.stack_blend), 1)]
        state["selection"] = sel
    state["members"] = [{"config": m["config"], "objective": m["objective"]} for m in members]
    log("  trained in every fold and averaged: " + ", ".join(f"{m['config']}/{m['objective']}" for m in members))

    # ---- (2) the folds; the v15 recipe next to it as a control
    specs, fold_of, w_new = make_folds(args.stack_cv, args.stack_folds, args.stack_seed, val, usable, times, n_news, buf)
    fits = {}                                                                         # (variant or 'ctrl', member) -> trained folds

    def kfold(specs_, cols, cfg, obj, tag):
        t1 = time.time()
        models, rounds_, flat_oof, gain = [], [], np.zeros(val.N, np.float32), np.zeros(len(cols))
        for tr, es, held in specs_:
            fm = fit_cols(be, X, y, val, tr, es, cols, obj, {**base, "depth": cfg["depth"], "mcw": cfg["mcw"]}, cap)
            models.append(fm)
            rounds_.append(int(fm[1]))
            gain += be.gain(fm, len(cols))
            rows = expand(val.ptr, val.lens, held)
            flat_oof[rows] = predict_rows(be, fm, X, cols, rows)
        log(f"  {tag}: {len(specs_)} folds, {be.name} {obj}, depth {cfg['depth']}, rounds {rounds_} ({time.time() - t1:.0f}s)")
        return {"models": models, "rounds": rounds_, "oof": flat_oof, "gain": gain, "cols": cols, "cfg": cfg, "objective": obj, "fit_s": round(time.time() - t1)}

    ctrl_name = "stack_meta_v15cfg"
    ctrl_on = bool(getattr(args, "stack_control", 1)) and "stack_meta" in variants
    if ctrl_on and S.time_left() > 900:
        c_specs, c_fold_of, c_w_new = make_folds("user", args.stack_folds, args.stack_seed, val, usable, times, n_news, buf)
        swap.set(X, "hi")
        m_c = bld.prefix(variants["stack_meta"])
        c_obj = "binary:logistic" if "binary:logistic" in objectives else objectives[0]
        fits[("ctrl", 0)] = kfold(c_specs, np.arange(m_c), CONFIGS["v15"], c_obj, f"{ctrl_name} (v15 recipe: user folds, every feature)")
    else:
        ctrl_on = False

    for mi, mem in enumerate(members):
        cfg = CONFIGS[mem["config"]]
        swap.set(X, "lo" if cfg["clean"] else "hi")
        for vname, grp in variants.items():
            if S.time_left() < 900:
                log(f"  {vname}/{mem['config']}: skipped (wall-clock budget)")
                continue
            m = bld.prefix(grp)
            fits[(vname, mi)] = kfold(specs, config_columns(bld, m, cfg), cfg, mem["objective"], f"{vname} / {mem['config']}")
    done_variants = [v for v in variants if any((v, mi) in fits for mi in range(len(members)))]
    del X, y, swap
    gc.collect()
    if not done_variants:
        log("  no variant could be trained (wall-clock budget)")
        return False

    # ---- (3) test: built chunk by chunk (never held whole), predicted by every fold model
    t0 = time.time()
    step = max(int(args.stack_chunk), 1)
    acc = {key: np.zeros(test.N, np.float32) for key in fits}
    raw = {(key, k): np.zeros(test.N, np.float32) for key, f in fits.items() for k in range(len(f["models"]))} if args.stack_save_scores > 1 else {}
    for a in range(0, test.n, step):
        b = min(a + step, test.n)
        Xc = bld.build("test", test, a, b)
        lptr = test.ptr[a:b + 1] - test.ptr[a]
        r0, r1 = test.ptr[a], test.ptr[b]
        zc = make_seg_z(lptr)
        Xl = None
        for key, f in fits.items():
            if f["cfg"]["clean"] and Xl is None:
                Xl = coarsen(Xc.copy(), bld.names, lptr)
            Xs = (Xl if f["cfg"]["clean"] else Xc)[:, f["cols"]]
            for k, mod in enumerate(f["models"]):
                pr = be.predict(mod, Xs)
                acc[key][r0:r1] += zc(pr)
                if raw:
                    raw[(key, k)][r0:r1] = pr
        del Xc, Xl
    log(f"  test features + predictions: {test.N:,} candidate rows, {len(fits)} fold sets, {time.time() - t0:.0f}s")
    gc.collect()

    # ---- (4) rows: validation = out-of-fold, test = mean over the fold models (and members) of their within-impression z-scores
    ref, ref_pop = ("frozen" if "frozen" in S.cache else None), ("frozen+pop" if "frozen+pop" in S.cache else None)
    zv = make_seg_z(val.ptr)
    all_val = np.arange(n)
    final, oof_final = {}, {}

    def finish(name, keys, w_new_, family, ref_name, cfg_info, extra):
        oof = np.mean([zv(fits[k]["oof"]) for k in keys], 0).astype(np.float32)
        test_s = (sum(acc[k] for k in keys) / sum(len(fits[k]["models"]) for k in keys)).astype(np.float32)
        vres = eval_flat(S, val, oof, all_val)
        tres = S.run(common.ragged_scorer({"ids": np.arange(test.n), "ptr": test.ptr, "flat": test_s}, np.arange(test.n), S.device), "test")
        nd_new, w_sum = new_metric(vres, w_new_)
        gain = np.zeros(len(bld.names))
        for k in keys:
            gain[fits[k]["cols"]] += fits[k]["gain"]
        m_full = bld.prefix(cfg_info["groups_tuple"])
        share = gain[:m_full] / max(gain[:m_full].sum(), 1e-12)
        gshare = {g: float(share[bld.group_of[:m_full] == g].sum()) for g in cfg_info["groups_tuple"]}
        top = sorted(((bld.names[c], float(s_)) for c, s_ in enumerate(share)), key=lambda x: -x[1])[:12]
        row = S.record(name, family, {k: v for k, v in cfg_info.items() if k != "groups_tuple"}, vres, tres, ref_name,
                       {**extra, "val_new_ndcg10": nd_new, "val_new_weight": w_sum, "n_features": int(cfg_info["features"])})
        info = {"rounds": [fits[k]["rounds"] for k in keys], "group_gain_share": gshare, "top_features": top, "fit_s": int(sum(fits[k]["fit_s"] for k in keys)),
                "val_ndcg10_oof": row["val_ndcg@10"], "val_ndcg10_oof_w": row["val_ndcg@10_w"], "val_new_ndcg10": nd_new, "test_ndcg10": row["ndcg@10"], "test_auc": row["auc"],
                "test_cold_auc": row.get("cold_auc"), "members": [f"{fits[k]['cfg_name']}/{fits[k]['objective']}" for k in keys]}
        state["variants"][name] = info
        final[name], oof_final[name] = test_s, oof
        log(f"  {name}: out-of-fold nDCG@10 {row['val_ndcg@10']:.4f} (on new-article clicks {nd_new:.4f}) | test AUC {row['auc']:.4f} nDCG@10 {row['ndcg@10']:.4f} | gain by group "
            + ", ".join(f"{g}={s_:.2f}" for g, s_ in gshare.items()))
        return row

    for key, f in fits.items():
        f["cfg_name"] = next((c for c, v in CONFIGS.items() if v == f["cfg"]), "?")
    base_cfg = {"backend": be.name, "folds": args.stack_folds, "components": [c.key for c in comps]}
    if ctrl_on:
        finish(ctrl_name, [("ctrl", 0)], c_w_new, "STACK-control", ref,
               {**base_cfg, "groups_tuple": variants["stack_meta"], "groups": list(variants["stack_meta"]), "features": int(len(fits[("ctrl", 0)]["cols"])), "objective": fits[("ctrl", 0)]["objective"],
                "cv": "user", "fit": "validation only, v15 recipe: 4 folds by user, every feature, deep trees; val = out-of-fold, test = mean over the fold models"},
               {"rounds": fits[("ctrl", 0)]["rounds"]})
    for vname in done_variants:
        keys = [(vname, mi) for mi in range(len(members)) if (vname, mi) in fits]
        pop = "pop" in variants[vname]
        finish(vname, keys, w_new, "STACK+pop" if pop else "STACK", ref_pop if pop else ref,
               {**base_cfg, "groups_tuple": variants[vname], "groups": list(variants[vname]), "features": int(max(len(fits[k]["cols"]) for k in keys)),
                "objective": members[0]["objective"], "cv": args.stack_cv, "members": [f"{members[mi]['config']}/{members[mi]['objective']}" for _, mi in keys],
                "fit": f"validation only, K-fold by {args.stack_cv}; val = out-of-fold, test = mean over the fold models of the members"},
               {"rounds": [fits[k]["rounds"] for k in keys]})
    state["objective"] = members[0]["objective"]
    if args.stack_save_scores:                                                       # the scores themselves, for later analysis (and the tests)
        out = {"val_ptr": val.ptr, "test_ptr": test.ptr, "val_users": val.users, "folds": fold_of, "val_times": times}
        for vname in final:
            out[f"{vname}__val_oof"], out[f"{vname}__val_have"] = oof_final[vname], np.ones(n, bool)
            out[f"{vname}__test"] = final[vname]
        for (key, k), p_ in raw.items():
            out[f"{'stack_meta_v15cfg' if key[0] == 'ctrl' else key[0]}__test_m{key[1]}_fold{k}"] = p_
        np.savez_compressed(os.path.join(S.work, "stack_scores.npz"), **out)
    del acc, raw
    gc.collect()
    state["seconds"] = round(time.time() - t_all)
    return True
