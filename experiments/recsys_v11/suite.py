"""V11 decision suite: which LLM-embedding direction is worth a thesis on news recommendation?

Every variant is scored zero-shot (user = mean of clicked-history vectors) so the table
measures *representation quality*, on the exact V10 protocol (chronological validation,
MINDsmall_dev as the one-shot test).  Hyper-parameters are picked on validation, the test
set is touched once per selected configuration, and every comparison to the control comes
with a paired 95% CI plus a cold-article slice (clicked articles the training rows never saw).

Hypotheses
  H1 history-as-text, instruction-aware query beats the mean of item vectors      (LLM-as-user-encoder)
  H2 co-click contrastive adaptation of the embedding (head / LoRA / DoRA,
     +Matryoshka, +impression hard negatives) beats the frozen embedding          (LLM2Rec-style)
  H3 entity / kNN retrieval-augmented article representations help fresh articles  (RAG)
  H4 GraphRAG-style community context and similar-user retrieval help             (GraphRAG-lite)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import torch

import adapt
import rep_eval as R
import semgraph as G
from data import MindData

EQUIV_REFERENCE = None  # callable(texts)->np.ndarray of reference embeddings (e.g. Sentence-Transformers), set by the notebook

TASK = "Given the headlines a reader recently clicked, retrieve the news article they would most likely read next"


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ helpers
def query_prefix(name, fmt="auto", instruct=True):
    low = name.lower()
    if fmt == "auto":
        fmt = "qwen3" if "qwen3" in low else "bge" if "bge" in low else "e5" if "e5" in low else "plain"
    if fmt == "qwen3":
        return f"Instruct: {TASK}\nQuery:" if instruct else ""
    if fmt == "bge":
        return "Represent this sentence for searching relevant passages: " if instruct else ""
    if fmt == "e5":
        return "query: "
    return ""


def avg_results(rs):
    """Average several seeds of the same variant (per-impression arrays and slice values)."""
    if len(rs) == 1:
        return rs[0]
    out = dict(rs[0])
    out["arr"] = {k: np.nanmean([r["arr"][k] for r in rs], axis=0) for k in R.METRICS}
    out["pos_auc"] = np.mean([r["pos_auc"] for r in rs], axis=0)
    m = {"auc": float(np.nanmean(out["arr"]["auc"])), **{k: float(np.mean(out["arr"][k])) for k in R.METRICS[1:]}}
    if "cold" in out:
        c = out["cold"]
        m["cold_auc"] = float(out["pos_auc"][c].mean()) if c.any() else float("nan")
        m["warm_auc"] = float(out["pos_auc"][~c].mean()) if (~c).any() else float("nan")
    out["mean"] = m
    return out


class Suite:
    def __init__(self, data, meta, E0, device, work, seen_tr, seen_va):
        self.data, self.meta, self.E0, self.device, self.work = data, meta, E0, device, work
        self.seen_tr, self.seen_va = seen_tr, seen_va
        self.es = {"val": R.EvalSet(data.validation, device=device), "test": R.EvalSet(data.test, device=device)}
        self.records, self.cache = [], {}

    def tensor(self, E):
        return torch.from_numpy(np.ascontiguousarray(E, dtype=np.float32)).to(self.device)

    def run(self, scorer, split):
        return R.evaluate(scorer, self.es[split], self.seen_tr)

    def val_ndcg(self, E):
        return self.run(R.mean_pool_scorer(self.tensor(E)), "val")["mean"]["ndcg@10"]

    def record(self, name, family, cfg, val, test, ref_name, extra=None):
        """Store a row; Δ vs the reference is computed on the test split with paired CIs."""
        ref = self.cache[ref_name]["test"] if ref_name in self.cache else None
        row = {"name": name, "family": family, "cfg": cfg, "ref": ref_name,
               "val_ndcg@10": val["mean"]["ndcg@10"], "val_auc": val["mean"]["auc"], **test["mean"],
               "cold_frac": test.get("cold_frac"), **(extra or {})}
        if ref is not None and name != ref_name:
            row["d_ndcg@10"] = R.compare(test, ref, "ndcg@10")
            row["d_auc"] = R.compare(test, ref, "auc")
            row["d_cold_auc"] = R.compare_slice(test, ref, "cold")
            row["d_warm_auc"] = R.compare_slice(test, ref, "warm")
            row["verdict"] = R.verdict(row["d_ndcg@10"][1], row["d_ndcg@10"][2])
        self.records.append(row)
        self.cache[name] = {"val": val, "test": test}
        d = row.get("d_ndcg@10")
        log(f"  {name:34s} val nDCG@10={row['val_ndcg@10']:.4f} | test AUC={row['auc']:.4f} nDCG@10={row['ndcg@10']:.4f}"
            + (f" | Δ={d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] {row['verdict']}" if d else "")
            + (f" | cold AUC={row.get('cold_auc', float('nan')):.4f}" if "cold_auc" in row else ""))
        return row

    # ---- generic grid: choose the config on validation, report it on test once
    def select(self, name, family, configs, make, ref_name, extra=None):
        best = None
        for cfg in configs:
            v = self.run(make(cfg, "val"), "val")
            log(f"    {name} {cfg}: val nDCG@10={v['mean']['ndcg@10']:.4f}")
            if best is None or v["mean"]["ndcg@10"] > best[0]["mean"]["ndcg@10"]:
                best = (v, cfg)
        v, cfg = best
        t = self.run(make(cfg, "test"), "test")
        return self.record(name, family, cfg, v, t, ref_name, extra)

    def export(self, name, E, model_name):
        path = os.path.join(self.work, f"emb_{name}.npz")
        np.savez(path, ids=np.array(self.meta["ids"]), vecs=np.asarray(E[1:], np.float32), model_name=np.array(model_name))
        return path


# ------------------------------------------------------------------- stages
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


def stage_graph(S, args):
    log("\n== H3/H4 retrieval-augmented representations (content-only, causal memory) ==")
    A = G.entity_matrix(S.meta["ents"], S.data.n_news)
    n_with = int((np.asarray(A.sum(1)).ravel() > 0).sum())
    log(f"  entity coverage: {n_with:,}/{S.data.n_news - 1:,} articles carry Wikidata entities")
    mem = {"val": S.seen_tr, "test": S.seen_tr | S.seen_va}
    memo = {sp: G.Memory(S.E0, A, mem[sp], S.device) for sp in ("val", "test")}
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

    comm = {}
    for sp in ("val", "test"):
        t0 = time.time()
        comm[sp] = memo[sp].communities(k=10, resolution=args.louvain_res, seed=0)
        log(f"  communities[{sp}]: {comm[sp][1].shape[0]} via {comm[sp][2]} ({time.time() - t0:.0f}s)")
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
                val_fn=S.val_ndcg, log=lambda *a: None)
            E = _seed_runs(S, name, "H2-adapt", {"kind": kind, "mrl": mrl, "hardneg": hn}, fn, "frozen", args.seeds, trunc)
            r = S.records[-1]
            if best is None or r["val_ndcg@10"] > best[0]:
                best = (r["val_ndcg@10"], name, E)
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
    for mode, mrl in (("lora", True), ("dora", True), ("lora", False)):
        name = f"{mode}{'+mrl' if mrl else ''}+hn"
        enc = adapt.HFEncoder(args.encoder, max_len=args.max_len, device=S.device, prefix=args.doc_prefix)
        t0 = time.time()
        E, info = adapt.train_encoder(enc, texts, pairs, Ef, mode=mode, mrl=mrl, hardneg=True,
                                      epochs=args.epochs_enc, bs=args.bs_enc, r=args.lora_r, val_ids=val_ids,
                                      val_fn=S.val_ndcg, max_steps=args.enc_steps, log=log)
        Et = S.tensor(E)
        S.record(name, "H2-adapt", {"mode": mode, "mrl": mrl, "r": args.lora_r}, S.run(R.mean_pool_scorer(Et), "val"),
                 S.run(R.mean_pool_scorer(Et), "test"), "frozen_hf",
                 {"trainable": info["trainable"], "steps": info["steps"], "train_s": time.time() - t0})
        S.export(name, E, args.encoder)
        del enc
        if S.device == "cuda":
            torch.cuda.empty_cache()


def stage_histquery(S, args):
    log("\n== H1 history-as-text, instruction-aware query ==")
    titles = [""] + S.meta["titles"]
    K = args.hist_k
    keys, rowq = {}, {"val": [], "test": []}
    for sp, rows in (("val", S.data.validation), ("test", S.data.test)):
        for r in rows:
            h = tuple(r[2][-K:])
            rowq[sp].append(keys.setdefault(h, len(keys)) if h else -1)
    uniq = sorted(keys, key=keys.get)
    log(f"  {len(uniq):,} unique histories (last {K} headlines) to encode")
    Et = S.tensor(S.E0)
    for instruct in (True, False):
        pre = query_prefix(args.encoder, args.query_format, instruct)
        enc = adapt.HFEncoder(args.encoder, max_len=args.q_max_len, device=S.device, prefix=pre)
        Q = np.zeros((len(uniq) + 1, S.E0.shape[1]), np.float32)
        t0 = time.time()
        Q[:-1] = enc.encode(["; ".join(titles[i] for i in h) for h in uniq], bs=args.enc_bs)
        log(f"  encoded queries ({'instruct' if instruct else 'plain'}) in {time.time() - t0:.0f}s, prefix={pre[:40]!r}")
        Qt = S.tensor(Q)
        rq = {sp: torch.tensor(np.where(np.array(rowq[sp]) < 0, len(uniq), rowq[sp]), device=S.device) for sp in rowq}
        mp = R.mean_pool_scorer(Et)

        def mk(c, sp, rq=rq, Qt=Qt, mp=mp):
            def f(b):
                s = torch.einsum("bd,bcd->bc", Qt[rq[sp][torch.as_tensor(b["rows"], device=S.device)]], Et[b["cand"]])
                return s + c["w"] * mp(b) if c["w"] else s
            return f
        tag = "instr" if instruct else "plain"
        S.select(f"histq_{tag}", "H1-text-user", [{"w": 0.0}], mk, "frozen")
        S.select(f"histq_{tag}+meanpool", "H1-text-user", [{"w": w} for w in (0.5, 1.0, 2.0)], mk, "frozen")
        del enc


# --------------------------------------------------------------------- output
def fmt_ci(t, digits=4):
    return f"{t[0]:+.{digits}f} [{t[1]:+.{digits}f}, {t[2]:+.{digits}f}]" if t else "-"


def write_table(S, work):
    cols = ["variant", "family", "val nDCG@10", "test AUC", "MRR", "nDCG@5", "nDCG@10", "Δ nDCG@10 [95% CI]",
            "cold AUC", "Δ cold AUC [95% CI]", "warm AUC", "verdict"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in S.records:
        lines.append("| " + " | ".join([
            r["name"], r["family"], f"{r['val_ndcg@10']:.4f}", f"{r['auc']:.4f}", f"{r['mrr']:.4f}",
            f"{r['ndcg@5']:.4f}", f"{r['ndcg@10']:.4f}", fmt_ci(r.get("d_ndcg@10")),
            f"{r.get('cold_auc', float('nan')):.4f}", fmt_ci(r.get("d_cold_auc")),
            f"{r.get('warm_auc', float('nan')):.4f}", r.get("verdict", "control")]) + " |")
    md = "\n".join(lines)
    with open(os.path.join(work, "results.md"), "w", encoding="utf-8") as f:
        f.write("# V11 decision suite (zero-shot representation quality, MINDsmall_dev test)\n\n" + md + "\n")
    with open(os.path.join(work, "results.json"), "w", encoding="utf-8") as f:
        json.dump(S.records, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return md


def summarize(S):
    """One representative per direction, chosen on VALIDATION (never on test), then compared with the
    control.  The verdict is also Bonferroni-adjusted for the number of directions being compared,
    because picking 'the best of 4 directions' is a multiple-comparison decision."""
    from statistics import NormalDist
    log("\n================ DECISION SUMMARY ================")
    by_h = {}
    for r in S.records:
        if r["family"] != "control" and "verdict" in r:
            by_h.setdefault(r["family"], []).append(r)
    z_adj = NormalDist().inv_cdf(1 - 0.05 / (2 * max(len(by_h), 1)))
    for fam, rs in by_h.items():
        best = max(rs, key=lambda r: r["val_ndcg@10"])             # selection on validation only
        d, c = best["d_ndcg@10"], best.get("d_cold_auc")
        se = (d[2] - d[1]) / (2 * 1.96)
        adj = R.verdict(d[0] - z_adj * se, d[0] + z_adj * se)
        log(f"{fam:16s} rep={best['name']:26s} (best of {len(rs)} by val) Δ nDCG@10 {fmt_ci(d)} -> {best['verdict']}"
            f" | Bonferroni(m={len(by_h)}): {adj}"
            + (f" | Δ cold AUC {fmt_ci(c)}" if c and not np.isnan(c[0]) else ""))
    log("Rule: pursue a direction only if its Bonferroni-adjusted verdict is BETTER; prefer the ones that also win on the "
        "cold slice (where LLM embeddings should matter) and whose gain is practically meaningful (>~0.005 nDCG@10).\n"
        "Unadjusted 'BETTER' on one of ~20 rows can be chance: with all effects null, about 1 in 20 rows would still pass.")
    return [max(rs, key=lambda r: r["val_ndcg@10"]) for rs in by_h.values()]


def head_to_head(S, reps):
    """Direction-vs-direction comparison on the test split (paired over impressions), Bonferroni over the pairs.
    The control comparison above says whether a direction helps at all; this says which direction is better."""
    from itertools import combinations
    from statistics import NormalDist
    pairs = list(combinations(reps, 2))
    out = []
    if not pairs:
        return out
    z_adj = NormalDist().inv_cdf(1 - 0.05 / (2 * len(pairs)))
    log(f"\n-- HEAD-TO-HEAD between directions (test, paired). CI shown is unadjusted 95%; the verdict is Bonferroni-adjusted over {len(pairs)} pairs --")
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
    first = next(iter(S.cache.values()))["test"]
    out["pos_news"], out["pos_row"], out["cold"] = first["pos_news"], first["pos_row"], first["cold"]
    path = os.path.join(S.work, "test_arrays.npz")
    np.savez_compressed(path, **out)
    return path


# ------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind-train", required=True)
    ap.add_argument("--mind-dev", required=True)
    ap.add_argument("--work", default="v11_out")
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--news-emb", default=None, help="npz from V10 llm_embed.py (exact V10 control)")
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--query-format", default="auto")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--q-max-len", type=int, default=384)
    ap.add_argument("--stages", default="controls,graph,head,encoder,histquery")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repro", default=None, help='published V10 control "auc,ndcg10" (bge_zeroshot: 0.6241,0.3909)')
    ap.add_argument("--pairs", type=int, default=200_000)
    ap.add_argument("--pairs-enc", type=int, default=100_000)
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-enc", type=int, default=1)
    ap.add_argument("--bs-enc", type=int, default=32)
    ap.add_argument("--enc-steps", type=int, default=1500)
    ap.add_argument("--enc-bs", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--hist-k", type=int, default=15)
    ap.add_argument("--louvain-res", type=float, default=1.0)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--knn-k", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    args = ap.parse_args(argv)

    os.makedirs(args.work, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    log(f"device={device} encoder={args.encoder} stages={args.stages}")
    meta = G.load_news_meta(args.mind_train, args.mind_dev)
    if args.news_emb:
        z = np.load(args.news_emb, allow_pickle=True)
        emb = {str(i): v for i, v in zip(z["ids"], z["vecs"])}
        log(f"E0 <- {args.news_emb} ({len(emb):,} vectors)")
    else:
        cache = os.path.join(args.work, "news_emb_E0.npz")
        if os.path.exists(cache) and str(np.load(cache)["model_name"]) == f"{args.encoder}|{args.max_len}|{args.doc_prefix}":
            z = np.load(cache, allow_pickle=True)
            vecs = z["vecs"]
        else:
            enc = adapt.HFEncoder(args.encoder, max_len=args.max_len, device=device, prefix=args.doc_prefix)
            t0 = time.time()
            vecs = enc.encode(meta["text"], bs=args.enc_bs)
            if EQUIV_REFERENCE is not None:  # catches a wrong pooling / EOS / prefix convention up front
                ref = np.asarray(EQUIV_REFERENCE(meta["text"][:64]), np.float32)
                cos = (vecs[:64] * ref).sum(1) / (np.linalg.norm(vecs[:64], axis=1) * np.linalg.norm(ref, axis=1) + 1e-8)
                log(f"E0 guard: HFEncoder vs reference embeddings, min cos={cos.min():.4f} mean={cos.mean():.4f}")
                if cos.min() < 0.99:
                    raise RuntimeError("HFEncoder disagrees with the reference encoder (pooling/EOS/prefix) - fix before running")
            np.savez(cache, ids=np.array(meta["ids"]), vecs=vecs, model_name=np.array(f"{args.encoder}|{args.max_len}|{args.doc_prefix}"))
            log(f"E0 encoded with {args.encoder} in {time.time() - t0:.0f}s")
            del enc
        emb = dict(zip(meta["ids"], vecs))
    data = MindData.from_mind(args.mind_train, args.mind_dev, llm_embeddings=emb)
    assert data.n_news - 1 == len(meta["ids"]), "article order mismatch between loader and metadata"
    seen_tr, seen_va = G.observed_masks(data)
    S = Suite(data, meta, data.llm_emb, device, args.work, seen_tr, seen_va)
    stages = {"controls": stage_controls, "graph": stage_graph, "head": stage_head,
              "encoder": stage_encoder, "histquery": stage_histquery}
    wanted = [s for s in args.stages.split(",") if s]
    if "controls" not in wanted:
        wanted.insert(0, "controls")
    for s in wanted:
        t_stage = time.time()
        try:
            stages[s](S, args)
            log(f"-- stage '{s}' finished in {time.time() - t_stage:.0f}s")
        except Exception:  # one failing stage must not discard the others
            log(f"\n!! stage '{s}' failed:\n{traceback.format_exc()}")
        write_table(S, args.work)
    md = write_table(S, args.work)
    log("\n" + md)
    S.reps = summarize(S)
    S.h2h = head_to_head(S, S.reps)
    with open(os.path.join(args.work, "head_to_head.json"), "w", encoding="utf-8") as f:
        json.dump(S.h2h, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"per-impression arrays -> {save_arrays(S)}")
    return S


if __name__ == "__main__":
    main()
