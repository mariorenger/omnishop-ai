"""V11 decision suite: which LLM-era direction is worth a thesis on news recommendation?

Every variant is evaluated on the exact V10 protocol (chronological validation split, MINDsmall_dev as the one-shot
test).  Hyper-parameters are picked on validation, the test split is touched once per selected configuration, and every
comparison carries a paired 95% CI plus a cold-article slice (clicked articles the training rows never saw).

Hypotheses (see README):
  H1  history-as-text, instruction-aware query                         (LLM as user encoder, zero-shot)
  H2  co-click contrastive adaptation: heads, LoRA/DoRA, MRL, hard neg  (LLM2Rec-style)
  H3  entity / kNN retrieval-augmented article representations          (RAG)
  H4  GraphRAG-lite (Louvain communities, similar-user retrieval)  +  H4b with LLM-written community summaries
  H5  LoRA-tuned LLM user tower, causal / bidirectional / soft attention mask
  H6  LLM-augmented articles and readers (KAR-style expansions, profiles)
  H7  Semantic IDs: RQ-KMeans profile + TIGER-style generative slate ranker
  H8  pooling rule: recency / late interaction
  H9  zero-shot LLM judge re-ranking the top-k
  H10 learned candidate-aware user model on top of each representation (downstream consistency check)
  +   encoder sweep (TF-IDF/LSA, BGE small/base/large, Qwen3-Embedding)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
import traceback

import numpy as np
import torch

import adapt
import common
import rep_eval as R
import semgraph as G
import stages_ext as X
from common import HistQueries, avg_results, fuse, log, query_prefix  # noqa: F401  (re-exported for callers)
from data import MindData

EQUIV_REFERENCE = None  # callable(texts)->np.ndarray of reference embeddings (e.g. Sentence-Transformers), set by the notebook

DEFAULT_STAGES = "controls,pooling,graph,head,histquery,sid,encoder,usermodel,sweep"
TECHNIQUE_FAMILIES_EXCLUDED = {"control", "H10-user-model", "enc-sweep"}      # not "techniques over the frozen embedding"


def _slim(r):
    """Keep what paired comparisons need, in compact dtypes (a full run caches ~100 variants)."""
    out = {"arr": {k: np.asarray(v, np.float32) for k, v in r["arr"].items()}, "ids": np.asarray(r["ids"], np.int64),
           "pos_news": np.asarray(r["pos_news"], np.int32), "pos_auc": np.asarray(r["pos_auc"], np.float32),
           "pos_row": np.asarray(r["pos_row"], np.int32), "mean": dict(r["mean"])}
    for k in ("cold", "cold_frac"):
        if k in r:
            out[k] = r[k]
    return out


class Suite:
    def __init__(self, data, meta, E0, device, work, seen_tr, seen_va, args=None):
        self.data, self.meta, self.E0, self.device, self.work = data, meta, E0, device, work
        self.seen_tr, self.seen_va, self.args = seen_tr, seen_va, args
        self.titles = [""] + meta["titles"]
        self.es = {"val": R.EvalSet(data.validation, device=device), "test": R.EvalSet(data.test, device=device)}
        self.records, self.cache = [], {}
        self.store, self.kv = {}, {}                      # in-memory artefacts / JSON-able facts that survive a resume
        self.cache_dir = (getattr(args, "cache_dir", None) or os.path.join(work, "cache"))
        self.stage, self._perm, self._sub = "", {}, {}

    # ---- plumbing
    def tensor(self, E):
        return torch.from_numpy(np.ascontiguousarray(E, dtype=np.float32)).to(self.device)

    def run(self, scorer, split, es=None):
        return R.evaluate(scorer, (es or self.es)[split], self.seen_tr)

    def val_ndcg(self, E):
        return self.run(R.mean_pool_scorer(self.tensor(E)), "val")["mean"]["ndcg@10"]

    def time_left(self):
        return common.time_left()

    def deadline(self, frac=1.0, cap_s=None):
        """Absolute time by which a long loop should stop: ``frac`` of the remaining budget (None = unlimited)."""
        left = common.time_left()
        if left == float("inf"):
            return None if cap_s is None else time.time() + cap_s
        t = left * frac
        return time.time() + max(min(t, cap_s) if cap_s is not None else t, 0.0)

    def perm(self, split):
        if split not in self._perm:
            self._perm[split] = np.random.default_rng(0).permutation(len(self.es[split].rows))
        return self._perm[split]

    def subset(self, split, n):
        """Nested random subset of a split -> ``(original row ids, EvalSet)``; ``n <= 0`` or ``>= N`` means the full split."""
        N = len(self.es[split].rows)
        if n is None or n <= 0 or n >= N:
            return np.arange(N), self.es[split]
        if (split, n) not in self._sub:
            ids = np.sort(self.perm(split)[:n])
            self._sub[(split, n)] = (ids, common.subset_evalset(self.data.validation if split == "val" else self.data.test, ids, self.device))
        return self._sub[(split, n)]

    def row(self, name):
        for r in reversed(self.records):
            if r["name"] == name:
                return r
        raise KeyError(name)

    # ---- shared heavy objects, built lazily so a resumed run can skip the stage that first needed them
    def memo(self):
        if "memo" not in self.store:
            A = G.entity_matrix(self.meta["ents"], self.data.n_news)
            mem = {"val": self.seen_tr, "test": self.seen_tr | self.seen_va}
            self.store["entity_A"] = A
            self.store["memo"] = {sp: G.Memory(self.E0, A, mem[sp], self.device) for sp in ("val", "test")}
        return self.store["memo"]

    def communities(self):
        if "comm" not in self.store:
            out = {}
            for sp, m in self.memo().items():
                t0 = time.time()
                out[sp] = m.communities(k=10, resolution=self.args.louvain_res, seed=0)
                log(f"  communities[{sp}]: {out[sp][1].shape[0]} via {out[sp][2]} ({time.time() - t0:.0f}s)")
            self.store["comm"] = out
        return self.store["comm"]

    def generator(self):
        import llm_gen
        if "gen" not in self.store:
            path = os.path.join(self.work, "gen_cache.jsonl")
            half = self.device == "cuda" and not self.args.gen_fp32
            g = llm_gen.Generator(self.args.gen_model, self.device, torch.float16 if half else torch.float32, path)
            if half and not g.self_test():                                    # fp16 overflow: fall back to fp32 (slower, the planner shrinks the subsets)
                log("  generator: fp16 output is broken (overflow?) -> reloading in fp32")
                g.free()
                g = llm_gen.Generator(self.args.gen_model, self.device, torch.float32, path + ".fp32")
            self.store["gen"] = g
        return self.store["gen"]

    def doc_encoder(self):
        return adapt.HFEncoder(self.args.encoder, max_len=self.args.max_len, device=self.device, prefix=self.args.doc_prefix)

    def query_encoder(self):
        return adapt.HFEncoder(self.args.encoder, max_len=self.args.q_max_len, device=self.device,
                               prefix=query_prefix(self.args.encoder, self.args.query_format, True))

    def close(self):
        g = self.store.pop("gen", None)
        if g is not None:
            g.free()

    # ---- results
    def record(self, name, family, cfg, val, test, ref_name, extra=None):
        """Store a row; Δ vs the reference is computed on the test split with paired CIs (common impressions only)."""
        ref = self.cache.get(ref_name)
        val, test = _slim(val), _slim(test)           # same float32 rounding as the cached reference: identical variants give exactly 0
        row = {"name": name, "family": family, "stage": self.stage, "cfg": cfg, "ref": ref_name,
               "n_val": int(len(val["ids"])), "n_test": int(len(test["ids"])),
               "val_ndcg@10": val["mean"]["ndcg@10"], "val_auc": val["mean"]["auc"], **test["mean"],
               "cold_frac": test.get("cold_frac"), **(extra or {})}
        if ref is not None and name != ref_name:
            rv, rt = ref["val"], ref["test"]
            row["val_delta"] = R.compare(val, rv, "ndcg@10")[0]
            row["d_ndcg@10"] = R.compare(test, rt, "ndcg@10")
            row["d_auc"] = R.compare(test, rt, "auc")
            row["d_cold_auc"] = R.compare_slice(test, rt, "cold")
            row["d_warm_auc"] = R.compare_slice(test, rt, "warm")
            row["verdict"] = R.verdict(row["d_ndcg@10"][1], row["d_ndcg@10"][2])
            _, ib = R._align(test, rt)
            row["ref_ndcg@10_same"] = float(np.mean(rt["arr"]["ndcg@10"][ib]))
            row["ref_auc_same"] = float(np.nanmean(rt["arr"]["auc"][ib]))
        self.records.append(row)
        self.cache[name] = {"val": val, "test": test}
        d = row.get("d_ndcg@10")
        log(f"  {name:34s} val nDCG@10={row['val_ndcg@10']:.4f} | test AUC={row['auc']:.4f} nDCG@10={row['ndcg@10']:.4f}"
            + (f" (n={row['n_test']:,})" if row["n_test"] != len(self.es["test"].rows) else "")
            + (f" | Δ={d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] {row['verdict']}" if d else "")
            + (f" | cold AUC={row.get('cold_auc', float('nan')):.4f}" if "cold_auc" in row else ""))
        return row

    def pop_twin(self, name, family, cfg, make, es, extra=None):
        """Same scorer + V10's online popularity (equal-weight z-score sum, exactly V10's ``bge_zs_pop`` recipe), reference ``frozen+pop``."""
        sc = lambda sp: common.fuse([(make(cfg, sp), 1.0), (common.pop_scorer, 1.0)])
        return self.record(f"{name}+pop", f"{family}+pop", {**cfg, "base": name}, self.run(sc("val"), "val", es=es),
                           self.run(sc("test"), "test", es=es), "frozen+pop", extra)

    def want_pop(self, ref_name):
        return bool(getattr(self.args, "with_pop", 0)) and ref_name == "frozen" and "frozen+pop" in self.cache

    def select(self, name, family, configs, make, ref_name, extra=None, es=None, pop=None):
        """Generic grid: choose the config on validation, report it on test once (plus its popularity twin)."""
        best = None
        for cfg in configs:
            v = self.run(make(cfg, "val"), "val", es=es)
            if len(configs) > 1:
                log(f"    {name} {cfg}: val nDCG@10={v['mean']['ndcg@10']:.4f}")
            if best is None or v["mean"]["ndcg@10"] > best[0]["mean"]["ndcg@10"]:
                best = (v, cfg)
        v, cfg = best
        t = self.run(make(cfg, "test"), "test", es=es)
        row = self.record(name, family, cfg, v, t, ref_name, extra)
        if self.want_pop(ref_name) if pop is None else pop:
            self.pop_twin(name, family, cfg, make, es, extra)
        return row

    def export(self, name, E, model_name):
        path = os.path.join(self.work, f"emb_{name}.npz")
        np.savez(path, ids=np.array(self.meta["ids"]), vecs=np.asarray(E[1:], np.float32), model_name=np.array(model_name))
        return path

    # ---- resume
    SIG_IGNORED = {"stages", "budget_hours", "min_stage_s", "no_resume", "work", "cache_dir", "device", "mind_train", "mind_dev"}

    def signature(self):
        """A saved state is reused only if the data AND every result-relevant argument are identical (so a QUICK run can never leak
        into a full run); the stage list, budget, device and paths do not matter."""
        a = {k: v for k, v in sorted(vars(self.args).items()) if k not in self.SIG_IGNORED}
        return {"args": hashlib.sha1(json.dumps(a, sort_keys=True, default=str).encode()).hexdigest(), "n_news": int(self.data.n_news),
                "n_val": len(self.data.validation), "n_test": len(self.data.test)}

    def save_state(self, done):
        path = os.path.join(self.work, "state.pkl")
        with open(path + ".tmp", "wb") as f:
            pickle.dump({"sig": self.signature(), "records": self.records, "cache": self.cache, "done": sorted(done), "kv": self.kv}, f, protocol=4)
        os.replace(path + ".tmp", path)

    def load_state(self):
        path = os.path.join(self.work, "state.pkl")
        if not os.path.exists(path):
            return set()
        try:
            with open(path, "rb") as f:
                st = pickle.load(f)
        except Exception as e:                                            # a truncated file must not block a fresh run
            log(f"state.pkl unreadable ({repr(e)[:80]}) - starting fresh")
            return set()
        if st["sig"] != self.signature():
            log("state.pkl belongs to a different configuration - starting fresh")
            return set()
        self.records, self.cache, self.kv = st["records"], st["cache"], st["kv"]
        return set(st["done"])


# ------------------------------------------------------------------- original stages
def stage_controls(S, args):
    log("\n== controls ==")
    E = S.tensor(S.E0)
    v, t = S.run(R.mean_pool_scorer(E), "val"), S.run(R.mean_pool_scorer(E), "test")
    S.record("frozen", "control", {}, v, t, "frozen")
    if args.repro:
        auc_ref, nd_ref = (float(x) for x in args.repro.split(","))
        ok = abs(t["mean"]["auc"] - auc_ref) < 0.003 and abs(t["mean"]["ndcg@10"] - nd_ref) < 0.003
        log(f"  reproduction check vs V10 bge_zeroshot (AUC {auc_ref}, nDCG@10 {nd_ref}): "
            f"{'MATCH' if ok else 'MISMATCH - check input text / truncation / embedding file'}")
    if t.get("cold_frac") is not None:
        log(f"  clicked articles that are cold (unseen in training rows): {t['cold_frac']:.1%}")
    mp = R.mean_pool_scorer(E)
    pv, pt = S.run(common.pop_scorer, "val"), S.run(common.pop_scorer, "test")
    rp = S.record("popularity", "control", {"note": "V10 online smoothed log-CTR, content-free"}, pv, pt, "frozen")
    fz = common.fuse([(mp, 1.0), (common.pop_scorer, 1.0)])
    rf = S.record("frozen+pop", "control", {"note": "V10 bge_zs_pop: z(BGE mean-pool) + z(popularity)"}, S.run(fz, "val"), S.run(fz, "test"), "frozen")
    if args.repro_pop:
        a1, n1, a2, n2 = (float(x) for x in args.repro_pop.split(","))
        ok = (abs(rp["auc"] - a1) < 0.003 and abs(rp["ndcg@10"] - n1) < 0.003 and abs(rf["auc"] - a2) < 0.003 and abs(rf["ndcg@10"] - n2) < 0.003)
        log(f"  reproduction check vs V10 popularity (AUC {a1}, nDCG@10 {n1}) and bge_zs_pop (AUC {a2}, nDCG@10 {n2}): "
            f"{'MATCH' if ok else 'MISMATCH - popularity plumbing differs from V10'}")
    Er = np.random.default_rng(0).standard_normal(S.E0.shape).astype(np.float32)
    Er[0] = 0
    Er[1:] /= np.linalg.norm(Er[1:], axis=1, keepdims=True)
    Et = S.tensor(Er)
    S.record("random_vec", "control", {"note": "random unit vectors: only 'already read' identity information"},
             S.run(R.mean_pool_scorer(Et), "val"), S.run(R.mean_pool_scorer(Et), "test"), "frozen")


def stage_graph(S, args):
    log("\n== H3/H4 retrieval-augmented representations (content-only, causal memory) ==")
    memo = S.memo()
    A = S.store["entity_A"]
    n_with = int((np.asarray(A.sum(1)).ravel() > 0).sum())
    log(f"  entity coverage: {n_with:,}/{S.data.n_news - 1:,} articles carry Wikidata entities")
    if n_with:
        S.select("entity_rag", "H3-RAG", [{"beta": b} for b in args.betas],
                 lambda c, sp: R.mean_pool_scorer(S.tensor(memo[sp].entity_view(c["beta"])[0])), "frozen")
    nn = {sp: memo[sp].neighbours(np.arange(1, S.data.n_news), max(args.knn_k)) for sp in ("val", "test")}

    def knn_view(c, sp):
        nbr, sim = nn[sp]
        k = c["k"]
        w = np.clip(sim[:, :k], 0, None)
        w = w / (w.sum(1, keepdims=True) + 1e-8)
        E = S.E0.copy()
        for s0 in range(0, len(nbr), 8192):                       # chunked: [rows,k,d] would be ~GBs at k=20
            sl = slice(s0, s0 + 8192)
            agg = (S.E0[nbr[sl, :k]] * w[sl, :, None]).sum(1)
            E[1 + s0:1 + s0 + len(agg)] = G._norm(S.E0[1 + s0:1 + s0 + len(agg)] + c["beta"] * G._norm(agg))
        return E
    S.select("knn_rag", "H3-RAG", [{"k": k, "beta": b} for k in args.knn_k for b in args.betas],
             lambda c, sp: R.mean_pool_scorer(S.tensor(knn_view(c, sp))), "frozen")
    comm = S.communities()
    S.select("community_graphrag", "H4-GraphRAG", [{"lam": l} for l in args.lams],
             lambda c, sp: G.community_scorer(S.E0, comm[sp][0], comm[sp][1], c["lam"], S.device), "frozen")
    um = G.UserMemory(S.data, S.E0, S.device)
    S.select("user_rag", "H4-GraphRAG", [{"gamma": g} for g in args.gammas],
             lambda c, sp: G.user_rag_scorer(S.E0, um, c["gamma"], S.device), "frozen")


def _seed_runs(S, name, family, cfg, train_fn, ref, seeds, trunc=None):
    """Train a stochastic adapter over seeds -> one averaged row; Matryoshka truncation reported."""
    vals, tests, Es, infos = [], [], [], []
    for sd in range(seeds):
        E, info = train_fn(sd)
        Et = S.tensor(E)
        vals.append(S.run(R.mean_pool_scorer(Et), "val"))
        tests.append(S.run(R.mean_pool_scorer(Et), "test"))
        Es.append(E); infos.append(info)
    extra = {"seeds": seeds, "best_epochs": [i["best_epoch"] for i in infos],
             "ndcg10_per_seed": [t["mean"]["ndcg@10"] for t in tests]}
    if seeds > 1:
        extra["ndcg10_std"] = float(np.std(extra["ndcg10_per_seed"]))
    for m in (trunc or []):
        Em = Es[0][:, :m].copy()
        Em = Em / (np.linalg.norm(Em, axis=1, keepdims=True) + 1e-8)
        extra[f"ndcg10@{m}d"] = S.run(R.mean_pool_scorer(S.tensor(Em)), "test")["mean"]["ndcg@10"]
    S.record(name, family, cfg, avg_results(vals), avg_results(tests), ref, extra)
    if S.want_pop("frozen"):
        pv, pt = [], []
        for E in Es:
            fz = common.fuse([(R.mean_pool_scorer(S.tensor(E)), 1.0), (common.pop_scorer, 1.0)])
            pv.append(S.run(fz, "val")), pt.append(S.run(fz, "test"))
        S.record(f"{name}+pop", f"{family}+pop", {**cfg, "base": name}, avg_results(pv), avg_results(pt), "frozen+pop", extra)
    return Es[0]


def stage_head(S, args):
    log("\n== H2 contrastive co-click adaptation: residual heads on frozen vectors ==")
    pairs = adapt.build_pairs(S.data, max_pairs=args.pairs)
    log(f"  {len(pairs[0]):,} (anchor, clicked, hard-negative) pairs from train_core")
    d = S.E0.shape[1]
    trunc = [m for m in (64, 128) if m < d]
    ref_t = S.cache["frozen"]["test"]
    for m in trunc:  # how good is plain truncation of the frozen vector? (the thing MRL should fix)
        Em = S.E0[:, :m] / (np.linalg.norm(S.E0[:, :m], axis=1, keepdims=True) + 1e-8)
        log(f"  frozen truncated to {m}d: test nDCG@10={S.run(R.mean_pool_scorer(S.tensor(Em)), 'test')['mean']['ndcg@10']:.4f}"
            f" (full {ref_t['mean']['ndcg@10']:.4f})")
    best = None
    for kind in ("lin", "mlp"):
        for tag, mrl, hn in (("", False, False), ("+hn", False, True), ("+hn+mrl", True, True)):
            name = f"head_{kind}{tag}"
            fn = lambda sd, kind=kind, mrl=mrl, hn=hn: adapt.train_head(
                S.E0, pairs, kind, mrl, hn, epochs=args.epochs_head, seed=sd, device=S.device,
                val_fn=S.val_ndcg, deadline=S.deadline(0.2), log=lambda *a: None)
            E = _seed_runs(S, name, "H2-adapt", {"kind": kind, "mrl": mrl, "hardneg": hn}, fn, "frozen", args.seeds, trunc)
            r = S.records[-1]
            if best is None or r["val_ndcg@10"] > best[0]:
                best = (r["val_ndcg@10"], name, E)
    S.kv["best_head"] = best[1]
    log(f"  best head on validation: {best[1]}  -> exported {S.export(best[1], best[2], args.encoder)}")


def stage_encoder(S, args):
    log("\n== H2 contrastive co-click adaptation: LoRA / DoRA with the encoder in the loop ==")
    texts = [""] + S.meta["text"]
    val_ids = np.unique(np.concatenate([np.asarray(r[2], np.int64).ravel() for r in S.data.validation]
                                        + [np.asarray(r[3], np.int64) for r in S.data.validation]))
    pairs = adapt.build_pairs(S.data, max_pairs=args.pairs_enc)
    log(f"  {len(pairs[0]):,} pairs, {len(val_ids):,} validation articles re-encoded per epoch")
    enc0 = adapt.HFEncoder(args.encoder, max_len=args.max_len, device=S.device, prefix=args.doc_prefix)
    Ef = np.zeros_like(S.E0)
    Ef[1:] = enc0.encode(S.meta["text"], bs=args.enc_bs)
    Et = S.tensor(Ef)
    S.record("frozen_hf", "control", {"max_len": args.max_len}, S.run(R.mean_pool_scorer(Et), "val"),
             S.run(R.mean_pool_scorer(Et), "test"), "frozen_hf")
    del enc0
    variants = (("lora", True), ("dora", True), ("lora", False))
    for i, (mode, mrl) in enumerate(variants):
        if S.time_left() < 1800:
            log(f"  {mode}: skipped (wall-clock budget)")
            continue
        name = f"{mode}{'+mrl' if mrl else ''}+hn"
        enc = adapt.HFEncoder(args.encoder, max_len=args.max_len, device=S.device, prefix=args.doc_prefix)
        t0 = time.time()
        E, info = adapt.train_encoder(enc, texts, pairs, Ef, mode=mode, mrl=mrl, hardneg=True,
                                      epochs=args.epochs_enc, bs=args.bs_enc, r=args.lora_r, val_ids=val_ids,
                                      val_fn=S.val_ndcg, max_steps=args.enc_steps, deadline=S.deadline(0.45 / (len(variants) - i)), log=log)
        Et = S.tensor(E)
        S.record(name, "H2-adapt", {"mode": mode, "mrl": mrl, "r": args.lora_r}, S.run(R.mean_pool_scorer(Et), "val"),
                 S.run(R.mean_pool_scorer(Et), "test"), "frozen_hf",
                 {"trainable": info["trainable"], "steps": info["steps"], "train_s": time.time() - t0,
                  "truncated": bool(info["steps"] < min(args.enc_steps, args.epochs_enc * (len(pairs[0]) // args.bs_enc)))})
        S.export(name, E, args.encoder)
        if S.want_pop("frozen"):
            fz = common.fuse([(R.mean_pool_scorer(S.tensor(E)), 1.0), (common.pop_scorer, 1.0)])
            S.record(f"{name}+pop", "H2-adapt+pop", {"mode": mode, "mrl": mrl, "r": args.lora_r, "base": name}, S.run(fz, "val"), S.run(fz, "test"), "frozen+pop")
        del enc
        if S.device == "cuda":
            torch.cuda.empty_cache()


def stage_histquery(S, args):
    log("\n== H1 history-as-text, instruction-aware query ==")
    K = args.hist_k
    hq = HistQueries(S.data, S.titles, K)
    n_val, n_test = (args.llm_val_n, args.llm_test_n) if args.hq_subset else (0, 0)
    sets = {sp: S.subset(sp, n) for sp, n in (("val", n_val), ("test", n_test))}
    es = {sp: sets[sp][1] for sp in sets}
    need = np.union1d(hq.needed("val", sets["val"][0]), hq.needed("test", sets["test"][0]))
    log(f"  {len(need):,} unique histories (last {K} headlines) to encode; evaluated on "
        f"{len(sets['val'][0]):,} validation / {len(sets['test'][0]):,} test impressions")
    Et = S.tensor(S.E0)
    mp = R.mean_pool_scorer(Et)
    for instruct in ((True, False) if args.hq_plain else (True,)):
        if S.time_left() < 900:
            log("  skipped (wall-clock budget)")
            break
        pre = query_prefix(args.encoder, args.query_format, instruct)
        enc = adapt.HFEncoder(args.encoder, max_len=args.q_max_len, device=S.device, prefix=pre)
        Q = hq.alloc(S.E0.shape[1])
        t0 = time.time()
        Q[need] = enc.encode(hq.texts(need), bs=args.enc_bs)
        log(f"  encoded queries ({'instruct' if instruct else 'plain'}) in {time.time() - t0:.0f}s, prefix={pre[:40]!r}")
        qs = lambda sp: hq.scorer(Q, Et, sp, S.device, ids=sets[sp][0])
        tag = "instr" if instruct else "plain"
        S.select(f"histq_{tag}", "H1-text-user", [{"w": 0.0}], lambda c, sp: qs(sp), "frozen", es=es)
        S.select(f"histq_{tag}+meanpool", "H1-text-user", [{"w": w} for w in (0.5, 1.0, 2.0)],
                 lambda c, sp: fuse([(qs(sp), 1.0), (mp, c["w"])]), "frozen", es=es)
        del enc
        if S.device == "cuda":
            torch.cuda.empty_cache()


STAGES = {"controls": stage_controls, "pooling": X.stage_pooling, "graph": stage_graph, "head": stage_head,
          "histquery": stage_histquery, "sid": X.stage_sid, "encoder": stage_encoder, "llm_user": X.stage_llm_user,
          "usermodel": X.stage_usermodel, "sweep": X.stage_sweep, "graphrag_llm": X.stage_graphrag_llm,
          "augment": X.stage_augment, "rerank": X.stage_rerank}


# --------------------------------------------------------------------- output
def fmt_ci(t, digits=4):
    return f"{t[0]:+.{digits}f} [{t[1]:+.{digits}f}, {t[2]:+.{digits}f}]" if t else "-"


def write_table(S, work):
    cols = ["variant", "family", "n test", "val nDCG@10", "test AUC", "MRR", "nDCG@5", "nDCG@10", "ref", "ref nDCG@10",
            "Δ nDCG@10 [95% CI]", "cold AUC", "Δ cold AUC [95% CI]", "warm AUC", "verdict"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in S.records:
        ref_nd = r.get("ref_ndcg@10_same", r["ndcg@10"] if r["name"] == r["ref"] else float("nan"))
        lines.append("| " + " | ".join([
            r["name"], r["family"], f"{r['n_test']:,}", f"{r['val_ndcg@10']:.4f}", f"{r['auc']:.4f}", f"{r['mrr']:.4f}",
            f"{r['ndcg@5']:.4f}", f"{r['ndcg@10']:.4f}", r["ref"], f"{ref_nd:.4f}", fmt_ci(r.get("d_ndcg@10")),
            f"{r.get('cold_auc', float('nan')):.4f}", fmt_ci(r.get("d_cold_auc")),
            f"{r.get('warm_auc', float('nan')):.4f}", r.get("verdict", "control")]) + " |")
    md = "\n".join(lines)
    with open(os.path.join(work, "results.md"), "w", encoding="utf-8") as f:
        f.write("# V11 decision suite (MINDsmall_dev test; Δ and 'ref nDCG@10' are computed on the same impressions as the row)\n\n" + md + "\n")
    with open(os.path.join(work, "results.json"), "w", encoding="utf-8") as f:
        json.dump(S.records, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return md


def _val_key(r):
    return r.get("val_delta", r["val_ndcg@10"])


def summarize(S, pop=False):
    """One representative per family, chosen on VALIDATION (never on test), then compared with its control.  The verdict is also
    Bonferroni-adjusted for the number of families, because picking 'the best of m directions' is a multiple-comparison decision.
    ``pop=False``: content-only variants vs ``frozen``;  ``pop=True``: the same variants fused with popularity vs ``frozen+pop`` (= V10 ``bge_zs_pop``)."""
    from statistics import NormalDist
    log("\n================ DECISION SUMMARY WITH POPULARITY (control = frozen+pop = V10 bge_zs_pop) ================" if pop
        else "\n================ DECISION SUMMARY (content only; control = frozen) ================")
    by_h = {}
    for r in S.records:
        if r["family"] != "control" and "verdict" in r and r["family"].endswith("+pop") == pop:
            by_h.setdefault(r["family"], []).append(r)
    if not by_h:
        return []
    z_adj = NormalDist().inv_cdf(1 - 0.05 / (2 * max(len(by_h), 1)))
    reps = []
    for fam, rs in by_h.items():
        best = max(rs, key=_val_key)                                # selection on validation only
        d, c = best["d_ndcg@10"], best.get("d_cold_auc")
        se = (d[2] - d[1]) / (2 * 1.96)
        adj = R.verdict(d[0] - z_adj * se, d[0] + z_adj * se)
        reps.append(best)
        kept = " [adapter = frozen: no epoch beat epoch 0 on validation]" if best.get("best_epochs") and not any(best["best_epochs"]) else ""
        log(f"{fam:18s} rep={best['name']:24s} (best of {len(rs):2d} by val, n={best['n_test']:,}, vs {best['ref']}) Δ nDCG@10 {fmt_ci(d)} -> {best['verdict']}"
            f" | Bonferroni(m={len(by_h)}): {adj}" + (f" | Δ cold AUC {fmt_ci(c)}" if c and not np.isnan(c[0]) else "") + kept)
    log("Rule: pursue a direction only if its Bonferroni-adjusted verdict is BETTER; prefer the ones that also win on the "
        "cold slice (where LLM embeddings should matter) and whose gain is practically meaningful (>~0.005 nDCG@10).\n"
        "Unadjusted 'BETTER' on one of many rows can be chance: with all effects null, about 1 in 20 rows would still pass.")
    return sorted(reps, key=_val_key, reverse=True)


def head_to_head(S, reps, top=5):
    """Direction-vs-direction comparison on the test split (paired over common impressions), Bonferroni over the pairs.
    Restricted to the ``top`` technique families ranked on VALIDATION (user-model / sweep rows have other controls)."""
    from itertools import combinations
    from statistics import NormalDist
    reps = [r for r in reps if r["family"] not in TECHNIQUE_FAMILIES_EXCLUDED][:top]
    pairs = list(combinations(reps, 2))
    out = []
    if not pairs:
        return out
    z_adj = NormalDist().inv_cdf(1 - 0.05 / (2 * len(pairs)))
    log(f"\n-- HEAD-TO-HEAD between the top-{len(reps)} directions by validation (test, paired on common impressions). "
        f"CI shown is unadjusted 95%; the verdict is Bonferroni-adjusted over {len(pairs)} pairs --")
    for a, b in pairs:
        ra, rb = S.cache[a["name"]]["test"], S.cache[b["name"]]["test"]
        d, c = R.compare(ra, rb, "ndcg@10"), R.compare_slice(ra, rb, "cold")
        se = (d[2] - d[1]) / (2 * 1.96)
        lo, hi = d[0] - z_adj * se, d[0] + z_adj * se
        word = f"{a['family']} > {b['family']}" if lo > 0 else (f"{b['family']} > {a['family']}" if hi < 0 else "not distinguishable")
        out.append({"a": a["name"], "b": b["name"], "d_ndcg@10": d, "d_cold_auc": c, "verdict": word})
        log(f"  {a['family']}:{a['name']} vs {b['family']}:{b['name']}  Δ nDCG@10 {fmt_ci(d)} | Δ cold AUC {fmt_ci(c)} -> {word}")
    return out


def save_arrays(S):
    """Per-impression metrics of every variant (test split) so any further paired analysis needs no re-run."""
    out = {"impr_ids": np.array([r[0] for r in S.data.test])}
    for name, c in S.cache.items():
        t = c["test"]
        for k in R.METRICS:
            out[f"{name}__{k}"] = t["arr"][k].astype(np.float32)
        out[f"{name}__pos_auc"] = t["pos_auc"].astype(np.float32)
        out[f"{name}__ids"] = t["ids"].astype(np.int32)                  # positions of the evaluated impressions (subsets differ)
    full = next((c["test"] for c in S.cache.values() if len(c["test"]["ids"]) == len(S.data.test)), None)
    if full is None:
        return None
    out["pos_news"], out["pos_row"], out["cold"] = full["pos_news"], full["pos_row"], full["cold"]
    path = os.path.join(S.work, "test_arrays.npz")
    np.savez_compressed(path, **out)
    return path


def cross_compare(suites, tags, pairs, path=None):
    """Direct paired comparison between named rows of (possibly different) runs on their common impressions:
    ``pairs = [((tag_a, name_a), (tag_b, name_b)), ...]`` -> Δ nDCG@10 and Δ cold AUC of A - B with 95% CIs."""
    by_tag = dict(zip(tags, suites))
    lines = ["| A | B | common impressions | Δ nDCG@10 (A-B) [95% CI] | Δ cold AUC (A-B) [95% CI] | verdict |", "|---|---|---|---|---|---|"]
    out = []
    for (ta, na), (tb, nb) in pairs:
        try:
            a, b = by_tag[ta].cache[na]["test"], by_tag[tb].cache[nb]["test"]
        except KeyError as e:
            log(f"  cross_compare: {ta}:{na} vs {tb}:{nb} skipped (missing {e})")
            continue
        d, c = R.compare(a, b, "ndcg@10"), R.compare_slice(a, b, "cold")
        n = int(len(np.intersect1d(a["ids"], b["ids"])))
        out.append({"a": f"{ta}:{na}", "b": f"{tb}:{nb}", "n": n, "d_ndcg@10": d, "d_cold_auc": c})
        lines.append(f"| {ta}:{na} | {tb}:{nb} | {n:,} | {fmt_ci(d)} | {fmt_ci(c)} | {R.verdict(d[1], d[2])} |")
    md = "\n".join(lines)
    log("\n================ DIRECT COMPARISONS ACROSS RUNS (paired, common impressions) ================\n" + md)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Direct paired comparisons between rows (A - B)\n\n" + md + "\n")
    return out


def plot_forest(rows, path, top=32):
    """Forest plot of the paired Δ nDCG@10 (95% CI) of the best rows vs the reference; returns the path (None if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rows = rows[:top]
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(rows) + 1.6))
    for i, x in enumerate(rows):
        d = x["d"]
        col = "#1f77b4" if d[1] > 0 else ("#c0392b" if d[2] < 0 else "#7f8c8d")
        ax.plot([d[1], d[2]], [-i, -i], lw=2.2, color=col)
        ax.plot([d[0]], [-i], "o", ms=4, color=col)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([f"{x['run']}:{x['name']}  [{x['family']}]" + (f" n={x['n']:,}" if x["n"] < 70000 else "") for x in rows], fontsize=7)
    ax.set_xlabel("Δ nDCG@10 vs reference (paired 95% CI)")
    ax.set_title("blue: CI above 0   red: CI below 0   grey: not distinguishable", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def leaderboard(suites, tags, ref, path=None, top=None, plot=None, pop=False):
    """One table over several runs (e.g. BGE-small and Qwen3-Embedding), each row compared on the common impressions with a single
    global reference ``ref = (tag, row name)`` -- by default the BGE-small frozen embedding (the V10-comparable control)."""
    by_tag = dict(zip(tags, suites))
    ref_res = by_tag[ref[0]].cache[ref[1]]["test"]
    rows = []
    for tag, S in by_tag.items():
        for r in S.records:
            if r["family"] == "H10-user-model":
                continue                                              # learned user models have their own control (um_frozen)
            keep_controls = ("popularity", "frozen+pop") if pop else ("frozen",)
            if r["family"] == "control":
                if r["name"] not in keep_controls:
                    continue
            elif r["family"].endswith("+pop") != pop:
                continue                                              # content-only and popularity-fused variants are ranked separately
            t = S.cache[r["name"]]["test"]
            d = R.compare(t, ref_res, "ndcg@10")
            if np.isnan(d[0]):
                continue
            rows.append({"run": tag, "name": r["name"], "family": r["family"], "n": r["n_test"], "d": d,
                         "dc": R.compare_slice(t, ref_res, "cold"), "val": _val_key(r), "auc": r["auc"], "ndcg": r["ndcg@10"]})
    rows.sort(key=lambda x: -x["d"][0])
    rows = rows[:top] if top else rows
    lines = [f"| rank | run | variant | family | n | Δ nDCG@10 vs {ref[0]}:{ref[1]} [95% CI] | Δ cold AUC [95% CI] | verdict |", "|---|---|---|---|---|---|---|---|"]
    for i, x in enumerate(rows, 1):
        lines.append(f"| {i} | {x['run']} | {x['name']} | {x['family']} | {x['n']:,} | {fmt_ci(x['d'])} | {fmt_ci(x['dc'])} | {R.verdict(x['d'][1], x['d'][2])} |")
    md = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Leaderboard over runs {tags}: zero-shot representation / re-ranking variants vs one global reference, paired on the common impressions "
                    f"(rows evaluated on subsets use the same impressions in the reference). Learned user-model rows (H10) are excluded: see each run's results.md (control um_frozen).\n\n{md}\n")
    log("\n================ LEADERBOARD (all runs, one global reference) ================\n" + md)
    if plot:
        log(f"forest plot -> {plot_forest(rows, plot)}")
    return rows


# ------------------------------------------------------------------------ main
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind-train", required=True)
    ap.add_argument("--mind-dev", required=True)
    ap.add_argument("--work", default="v11_out")
    ap.add_argument("--cache-dir", default=None, help="where frozen article embeddings are cached (share it between runs)")
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--news-emb", default=None, help="npz from V10 llm_embed.py (exact V10 control)")
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--query-format", default="auto")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--q-max-len", type=int, default=384)
    ap.add_argument("--stages", default=DEFAULT_STAGES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repro", default=None, help='published V10 control "auc,ndcg10" (bge_zeroshot: 0.6241,0.3909)')
    ap.add_argument("--budget-hours", type=float, default=0.0, help="shared wall-clock budget; long loops stop gracefully (0 = unlimited)")
    ap.add_argument("--min-stage-s", type=float, default=300.0, help="do not start a stage with less than this left")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--with-pop", type=int, default=1, help="also report every zero-shot variant fused with V10's online popularity (rows '+pop')")
    ap.add_argument("--repro-pop", default=None, help='V10 "popularity_auc,popularity_ndcg10,bge_zs_pop_auc,bge_zs_pop_ndcg10" (0.6533,0.4026,0.6769,0.4330)')
    ap.add_argument("--pairs", type=int, default=200_000)
    ap.add_argument("--pairs-enc", type=int, default=100_000)
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-enc", type=int, default=1)
    ap.add_argument("--bs-enc", type=int, default=32)
    ap.add_argument("--enc-steps", type=int, default=1500)
    ap.add_argument("--enc-bs", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--hist-k", type=int, default=10)
    ap.add_argument("--hq-plain", type=int, default=1, help="also run the history query without the instruction")
    ap.add_argument("--hq-subset", type=int, default=0, help="evaluate H1 on the same nested subsets as H5")
    ap.add_argument("--louvain-res", type=float, default=1.0)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--knn-k", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--pool-lams", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--pool-taus", type=float, nargs="+", default=[0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5])
    ap.add_argument("--pool-ks", type=int, nargs="+", default=[2, 3, 4, 5, 7, 10])
    # semantic ids
    ap.add_argument("--sid-k", type=int, default=256)
    ap.add_argument("--sid-levels", type=int, default=3)
    ap.add_argument("--sid-epochs", type=int, default=3)
    ap.add_argument("--sid-bs", type=int, default=128)
    ap.add_argument("--sid-lr", type=float, default=1e-3)
    ap.add_argument("--sid-nmax", type=int, default=30)
    ap.add_argument("--sid-d", type=int, default=256)
    ap.add_argument("--sid-layers", type=int, default=4)
    ap.add_argument("--sid-heads", type=int, default=4)
    ap.add_argument("--sid-steps", type=int, default=0)
    ap.add_argument("--sid-val-n", type=int, default=3000)
    ap.add_argument("--sid-ws", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    # encoder sweep
    ap.add_argument("--sweep-encoders", default="BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5,BAAI/bge-m3,thenlper/gte-base,"
                                                 "sentence-transformers/all-mpnet-base-v2,sentence-transformers/all-MiniLM-L6-v2,Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--lsa-dim", type=int, default=256)
    # LLM user tower
    ap.add_argument("--llm-user-modes", default="causal,bidir,soft,causal@1")
    ap.add_argument("--llm-user-steps", type=int, default=800)
    ap.add_argument("--llm-user-bs", type=int, default=16)
    ap.add_argument("--llm-user-lr", type=float, default=1e-4)
    ap.add_argument("--llm-user-r", type=int, default=16)
    ap.add_argument("--llm-user-shared", type=int, default=1, help="share all positives and impression negatives of the batch as negatives")
    ap.add_argument("--llm-val-n", type=int, default=5000)
    ap.add_argument("--llm-test-n", type=int, default=30000)
    ap.add_argument("--llm-enc-bs", type=int, default=64)
    ap.add_argument("--ut-ws", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    # learned user model
    ap.add_argument("--um-views", default="frozen,head,encoder,entity_rag,knn_rag")
    ap.add_argument("--um-seeds", type=int, default=2)
    ap.add_argument("--um-epochs", type=int, default=12)
    ap.add_argument("--um-repro", default=None, help='V10 llmenc_ca "auc,ndcg10" (0.6494,0.3997)')
    # generation-based stages
    ap.add_argument("--gen-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--gen-fp32", type=int, default=0, help="load the generator in fp32 (use if fp16 produces garbage)")
    ap.add_argument("--gen-bs", type=int, default=48)
    ap.add_argument("--gen-new-tokens", type=int, default=56)
    ap.add_argument("--gen-hist-k", type=int, default=5)
    ap.add_argument("--gen-val-n", type=int, default=800)
    ap.add_argument("--gen-test-n", type=int, default=2500)
    ap.add_argument("--augment-minutes", type=float, default=35.0)
    ap.add_argument("--kar-alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--kar-ws", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--gr-max-comm", type=int, default=300)
    ap.add_argument("--gr-titles", type=int, default=8)
    ap.add_argument("--rerank-k", type=int, default=10)
    ap.add_argument("--rerank-hist", type=int, default=8)
    ap.add_argument("--rerank-val-n", type=int, default=600)
    ap.add_argument("--rerank-test-n", type=int, default=2000)
    ap.add_argument("--rerank-minutes", type=float, default=35.0)
    ap.add_argument("--rerank-ws", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    return ap


def guard_embeddings(vecs, meta, reference, tol=0.99):
    """Catch a wrong pooling / EOS / prefix convention up front by comparing 64 vectors with a reference implementation."""
    ref = reference(meta["text"][:64])
    if ref is None:
        log("E0 guard: reference encoder unavailable - pooling/EOS/prefix NOT cross-checked")
        return
    ref = np.asarray(ref, np.float32)
    cos = (vecs[:64] * ref).sum(1) / (np.linalg.norm(vecs[:64], axis=1) * np.linalg.norm(ref, axis=1) + 1e-8)
    log(f"E0 guard: HFEncoder vs reference embeddings, min cos={cos.min():.4f} mean={cos.mean():.4f}")
    if cos.min() < tol:
        raise RuntimeError("HFEncoder disagrees with the reference encoder (pooling/EOS/prefix) - fix before running")


def main(argv=None):
    args = build_parser().parse_args(argv)
    common.CLOCK.start(args.budget_hours)
    os.makedirs(args.work, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    log(f"device={device} encoder={args.encoder} stages={args.stages} budget={'unlimited' if not args.budget_hours else f'{args.budget_hours:.1f}h'}")
    meta = G.load_news_meta(args.mind_train, args.mind_dev)
    cache_dir = args.cache_dir or os.path.join(args.work, "cache")
    if args.news_emb:
        z = np.load(args.news_emb, allow_pickle=True)
        emb = {str(i): v for i, v in zip(z["ids"], z["vecs"])}
        log(f"E0 <- {args.news_emb} ({len(emb):,} vectors)")
    else:
        t0 = time.time()
        vecs, secs = X.embed_news(meta, args.encoder, args.max_len, args.doc_prefix, device, args.enc_bs, cache_dir)
        if EQUIV_REFERENCE is not None:
            guard_embeddings(vecs, meta, EQUIV_REFERENCE)
        log(f"E0 {'loaded from cache' if not secs else f'encoded with {args.encoder} in {secs:.0f}s'} ({vecs.shape[0]:,} x {vecs.shape[1]})")
        emb = dict(zip(meta["ids"], vecs))
    data = MindData.from_mind(args.mind_train, args.mind_dev, llm_embeddings=emb)
    assert data.n_news - 1 == len(meta["ids"]), "article order mismatch between loader and metadata"
    seen_tr, seen_va = G.observed_masks(data)
    S = Suite(data, meta, data.llm_emb, device, args.work, seen_tr, seen_va, args)
    done = set() if args.no_resume else S.load_state()
    if done:
        log(f"resuming: stages already finished -> {sorted(done)} ({len(S.records)} rows restored)")
    wanted = [s for s in args.stages.split(",") if s]
    if "controls" not in wanted:
        wanted.insert(0, "controls")
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {sorted(STAGES)}")
    for s in wanted:
        if s in done:
            log(f"-- stage '{s}' already finished (resume), skipped")
            continue
        if s != "controls" and common.out_of_time(args.min_stage_s):               # controls are the reference: always run
            log(f"-- stage '{s}' SKIPPED: wall-clock budget exhausted ({common.CLOCK.elapsed() / 3600:.1f}h used)")
            continue
        S.records = [r for r in S.records if r.get("stage") != s]                  # a retried stage must not duplicate its rows
        t_stage = time.time()
        S.stage = s
        try:
            STAGES[s](S, args)
            done.add(s)
            S.kv.setdefault("timings", {})[s] = round(time.time() - t_stage)
            log(f"-- stage '{s}' finished in {time.time() - t_stage:.0f}s (total {common.CLOCK.elapsed() / 3600:.2f}h)")
        except Exception:  # one failing stage must not discard the others
            log(f"\n!! stage '{s}' failed:\n{traceback.format_exc()}")
        if device == "cuda":
            torch.cuda.empty_cache()
        write_table(S, args.work)
        S.save_state(done)
    S.close()
    tm = S.kv.get("timings", {})
    if tm:
        log("STAGE TIMINGS (s): " + ", ".join(f"{k}={v}" for k, v in tm.items()) + f" | sum={sum(tm.values())}s")
        with open(os.path.join(args.work, "timings.json"), "w", encoding="utf-8") as f:
            json.dump(tm, f, indent=1)
    md = write_table(S, args.work)
    log("\n" + md)
    S.reps = summarize(S)
    S.h2h = head_to_head(S, S.reps)
    S.reps_pop = summarize(S, pop=True)
    S.h2h_pop = head_to_head(S, S.reps_pop)
    for fn, obj in (("head_to_head.json", S.h2h), ("head_to_head_pop.json", S.h2h_pop)):
        with open(os.path.join(args.work, fn), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"per-impression arrays -> {save_arrays(S)}")
    return S


if __name__ == "__main__":
    main()
