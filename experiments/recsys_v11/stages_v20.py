"""v20 stages: the LLM-text directions run at full scale, so that what reaches the fusion is measured on the whole test day.

  kar_full   H6  KAR-style LLM expansion of EVERY article (Qwen3-1.7B, title + abstract in, one or two sentences out), split over every GPU; the expansions are
                 embedded with the run's encoder and added to the article vectors (``E0 + alpha * E_exp``, renormalised).  Rows on the whole validation /
                 test split: ``kar_item`` (mean pool), ``kar_item_lse`` (late interaction), ``kar_exp`` (the expansion vectors alone).  The augmented vectors
                 are written as ``emb_kar_aug.npz`` so that a later run reads them with ``--news-emb`` and trains the learned readers on them.
  sid2       H7  Semantic IDs v2: codes from a chosen text variant, the generative model scored PER CODE LEVEL (the coarse level is the topic cluster,
                 the only part that can generalise to the cold articles; v7 summed all levels and scored at chance), level set and PMI chosen on validation,
                 plus a shuffled-history control.
  histq_mlp  H1+ the history-as-text query of H1 (the frozen LLM encoder reading the last K headlines with an instruction) followed by residual MLP heads
                 trained like a reader (1 click + 4 skipped articles, cross-entropy, early stopping on validation).  At step 0 the heads are the identity, so
                 the row starts exactly at H1; the difference to H1 is what a learned head on the frozen vector adds, the difference to H5 what fine-tuning
                 the LLM itself adds.

Every function takes the ``Suite`` object ``S`` (see suite.py) and the parsed ``args``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import adapt
import common
import fields as FD
import llm_gen
import pooling as P
import rep_eval as R
import semgraph as G
import sid as SID
from common import HistQueries, avg_results, log, query_prefix
from stages_ext import EXPAND_PROMPT, table_from, text_table


# ============================================================================ generation on every GPU
def load_generators(name, devices, dtype, cache_base):
    """One generator per device; each appends to its own cache file (no two writers on one file), all share one in-memory cache so a resumed run
    regenerates nothing that either device finished."""
    gens = [llm_gen.Generator(name, d, dtype, f"{cache_base}.{i}.jsonl") for i, d in enumerate(devices)]
    shared = {}
    for g in gens:
        shared.update(g.cache)
    for g in gens:
        g.cache = shared
    return gens


def generate_parallel(gens, prompts, max_new_tokens=56, bs=48, deadline=None, desc="generate", log_every=4000):
    """``Generator.generate`` over several devices: the prompts not in the cache are sorted by length and cut into batches of ``bs``; batch ``k`` goes to
    generator ``k % n``.  Each batch is generated exactly as one generator would (greedy decoding), so only the wall-clock time is divided.
    -> list of strings (``None`` where the deadline stopped generation)."""
    g0 = gens[0]
    rendered = [g0.render(p) for p in prompts]
    keys = [g0._key("gen", r, str(max_new_tokens)) for r in rendered]
    out = [g0.cache.get(k) for k in keys]
    todo = sorted((i for i, o in enumerate(out) if o is None), key=lambda i: len(rendered[i]))
    batches = [todo[s:s + bs] for s in range(0, len(todo), bs)]
    lock, state, errors = threading.Lock(), {"done": 0, "next_log": log_every}, []
    t0 = time.time()

    def work(gi):
        g = gens[gi]
        try:
            for bi in range(gi, len(batches), len(gens)):
                ix = batches[bi]
                res = g.generate([prompts[i] for i in ix], max_new_tokens=max_new_tokens, bs=len(ix), deadline=deadline, desc=desc)
                for i, t in zip(ix, res):
                    out[i] = t
                with lock:
                    state["done"] += sum(t is not None for t in res)
                    if state["done"] >= state["next_log"]:
                        state["next_log"] += log_every
                        el = time.time() - t0
                        log(f"      [{desc}] {state['done']:,}/{len(todo):,} generated ({el:.0f}s, {state['done'] / max(el, 1e-6):.1f}/s on {len(gens)} device(s))")
                if any(t is None for t in res):          # the deadline passed inside this batch: stop this device
                    break
        except BaseException as e:                        # re-raised in the calling thread
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(gens))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    left = sum(o is None for o in out)
    log(f"      [{desc}] {len(todo) - left:,} new + {len(prompts) - len(todo):,} cached; {left:,} not generated ({time.time() - t0:.0f}s)")
    return out


def catalogue_vectors(mind_train, mind_dev, encoder, cache_dir, device, max_len=256, base_text="ta", enc_bs=128):
    """The frozen vectors a run of ``encoder`` reads (``suite.embed_catalogue``, served from the on-disk cache when that run already encoded them)
    -> ``(article ids, vectors [n_articles, d])``."""
    import suite                                           # late: suite imports this module
    meta = G.load_news_meta(mind_train, mind_dev)
    a = argparse.Namespace(encoder=encoder, base_text=base_text, enc_dtype="auto", max_len=max_len, doc_prefix="", enc_bs=enc_bs)
    vecs, _ = suite.embed_catalogue(meta, a, device, cache_dir)
    return list(meta["ids"]), np.asarray(vecs, np.float32)


def concat_npz(parts, out_path, model_name="concat"):
    """``parts``: ``[(ids, vectors), ...]`` over the same articles -> an ``--news-emb`` file whose vectors are the parts L2-normalised, scaled by
    1/sqrt(parts) and side by side (the scheme of ``suite.embed_catalogue`` for ``concat:``).  -> ``out_path``."""
    ids0 = [str(i) for i in parts[0][0]]
    vs = []
    for ids, v in parts:
        if [str(i) for i in ids] != ids0:
            raise ValueError("the parts list different articles (or another order)")
        v = np.asarray(v, np.float32)
        vs.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8) / np.sqrt(len(parts)))
    np.savez(out_path, ids=np.array(ids0), vecs=np.concatenate(vs, 1).astype(np.float32), model_name=np.array(model_name))
    return out_path


def load_npz_vectors(path):
    z = np.load(path, allow_pickle=True)
    return [str(i) for i in z["ids"]], np.asarray(z["vecs"], np.float32)


# ============================================================================ H6 at full scale
def stage_kar_full(S, args):
    log("\n== H6 at full scale: an LLM expansion of EVERY article (KAR-style), embedded and added to the article vectors ==")
    fam = "H6-LLM-aug"
    T, A = S.titles, [""] + S.meta["abstracts"]
    n = S.data.n_news
    prompts = [EXPAND_PROMPT.format(title=T[i], abstract=A[i][:300]) for i in range(1, n)]
    devs = FD.gpu_devices(S.device)
    half = str(S.device).startswith("cuda") and not args.gen_fp32
    cache_base = os.path.join(S.work, "kar_gen_cache")
    gens = load_generators(args.gen_model, devs, torch.float16 if half else torch.float32, cache_base)
    if half and not gens[0].self_test():                   # fp16 overflow: reload in fp32 (slower)
        log("  generator: fp16 output is broken (overflow?) -> reloading in fp32")
        for g in gens:
            g.free()
        gens = load_generators(args.gen_model, devs, torch.float32, cache_base + ".fp32")
    t0 = time.time()
    out = generate_parallel(gens, prompts, args.gen_new_tokens, args.gen_bs, S.deadline(args.kar_gen_frac), "article expansions (all)")
    for g in gens:
        g.free()
    del gens
    gen_s = time.time() - t0
    exp = {i: t for i, t in zip(range(1, n), out) if t}
    ok = np.array(sorted(exp), np.int64)
    log(f"  {len(ok):,}/{n - 1:,} articles expanded in {gen_s / 60:.1f} min on {len(devs)} device(s)"
        + (f"; example: {T[ok[0]][:60]!r} -> {exp[ok[0]][:140]!r}" if len(ok) else ""))
    with open(os.path.join(S.work, "kar_expansions.jsonl"), "w", encoding="utf-8") as f:
        for i in ok:
            f.write(json.dumps({"id": S.meta["ids"][i - 1], "title": T[i], "expansion": exp[i]}, ensure_ascii=False) + "\n")
    if len(ok) < 0.5 * (n - 1):
        log("  fewer than half of the articles were expanded (budget): rows not computed")
        S.kv["kar_full"] = {"expanded": int(len(ok)), "articles": int(n - 1), "gen_s": round(gen_s)}
        return False
    t1 = time.time()
    dtype = adapt.resolve_dtype(args.encoder, getattr(args, "enc_dtype", "auto"), S.device)
    Eg = np.zeros_like(S.E0)
    Eg[ok] = adapt.encode_multi(args.encoder, [T[i] + ". " + exp[i] for i in ok], args.max_len, args.doc_prefix, dtype, args.enc_bs, devs, log=log)
    S.kv["kar_full"] = {"expanded": int(len(ok)), "articles": int(n - 1), "gen_s": round(gen_s), "embed_s": round(time.time() - t1), "devices": len(devs),
                        "model": args.gen_model, "max_new_tokens": args.gen_new_tokens}
    held = {}

    def aug(a):
        if a not in held:
            held.clear()
            held[a] = G._norm(S.E0 + a * Eg)
        return held[a]
    S.select("kar_item", fam, [{"alpha": a} for a in args.kar_alphas], lambda c, sp: R.mean_pool_scorer(S.tensor(aug(c["alpha"]))), "frozen",
             export=True, kind="zs")
    alpha = S.row("kar_item")["cfg"]["alpha"]
    Eb = S.tensor(aug(alpha))
    ref_lse = "pool_lse" if "pool_lse" in S.cache else "frozen"
    S.select("kar_item_lse", fam, [{"alpha": alpha, "tau": t} for t in args.pool_taus], lambda c, sp: P.late_scorer(Eb, "lse", c["tau"]), ref_lse,
             export=True, kind="zs")
    Ex = S.tensor(G._norm(Eg))
    S.select("kar_exp", fam, [{"text": "title + LLM expansion only"}], lambda c, sp: R.mean_pool_scorer(Ex), "frozen", export=True, kind="aux")
    path = S.export("kar_aug", aug(alpha), f"{args.encoder} + {args.gen_model} expansions (alpha {alpha})")
    S.kv["kar_full"]["alpha"], S.kv["kar_full"]["aug_npz"] = alpha, path
    log(f"  augmented vectors (alpha {alpha}) -> {path} (read them with --news-emb to train readers on them)")
    del Eb, Ex
    if str(S.device).startswith("cuda"):
        torch.cuda.empty_cache()


# ============================================================================ H7 v2
def stage_sid2(S, args):
    log("\n== H7 Semantic IDs v2: codes of a chosen text variant, the generative ranker scored per code level (chosen on validation), shuffled-history control ==")
    K, L, fam = args.sid_k, args.sid_levels, "H7-SemanticID"
    t0 = time.time()
    E_codes = S.E0 if args.sid_text == "ta" else text_table(S, args, {"mode": args.sid_text})
    codes = SID.rq_kmeans(E_codes, S.seen_tr | S.seen_va, L=L, K=K, seed=0, device=S.device)
    pref = SID.prefix_ids(codes, K)
    log(f"  codes from the '{args.sid_text}' vectors: " + ", ".join(f"L{l + 1}: {len(np.unique(pref[l][1:])):,} prefixes" for l in range(L))
        + f" ({time.time() - t0:.0f}s)")
    S.kv["sid2"] = {"text": args.sid_text, "prefixes": [int(len(np.unique(pref[l][1:]))) for l in range(L)]}
    levels_all = [tuple(range(1, l + 1)) for l in range(1, L + 1)]
    S.select("sid_profile", fam, [{"levels": lv} for lv in levels_all], lambda c, sp: SID.sid_profile_scorer(codes, K, c["levels"], S.device), "frozen",
             export=True, kind="zs")
    sets = [tuple(int(ch) for ch in s) for s in args.sid_levelsets.split(",") if s]
    sets = [s for s in sets if s and all(1 <= l <= L for l in s)] or [tuple(range(1, L + 1))]
    val_ids, val_es = S.subset("val", args.sid_val_n)

    def val_fn(model):                                    # epoch choice: the best level set on a validation subset
        cache = SID.LevelCache(val_es, SID.sid_level_scorer(model, codes, args.sid_nmax, device=S.device))
        return max(S.run(cache.scorer(s, S.device), "val", es={"val": val_es})["mean"]["ndcg@10"] for s in sets)

    model, info = SID.train_sid_gpt(codes, K, S.data.train_core, n_max=args.sid_nmax, epochs=args.sid_epochs, bs=args.sid_bs, lr=args.sid_lr, seed=0,
                                    device=S.device, val_fn=val_fn, d=args.sid_d, layers=args.sid_layers, heads=args.sid_heads,
                                    max_steps=args.sid_steps or None, deadline=S.deadline(0.35), log=log)
    params = sum(p.numel() for p in model.parameters())
    log(f"  SID-GPT: {params:,} parameters, best epoch {info['best_epoch']}, {info['steps']} steps, validation (subset, best level set) {info['val']:.4f}")
    codes_t = torch.from_numpy(codes).to(S.device)
    base = SID.history_free_level_logprob(model, codes_t, len(codes))
    t1 = time.time()
    lvf = SID.sid_level_scorer(model, codes, args.sid_nmax, device=S.device)
    cache = {sp: SID.LevelCache(S.es[sp], lvf) for sp in ("val", "test")}
    log(f"  every validation / test slate scored once per level in {time.time() - t1:.0f}s")
    extra = {"params": params, "best_epoch": info["best_epoch"], "steps": info["steps"]}
    cfgs = [{"levels": list(s), "pmi": pmi} for s in sets for pmi in (False, True)]
    mk = lambda c, sp: cache[sp].scorer(c["levels"], S.device, base if c["pmi"] else None)
    S.select("sid_gpt", fam, cfgs, mk, "frozen", extra, export=True, kind="zs")
    full = {"levels": list(range(1, L + 1)), "pmi": False}
    S.select("sid_gpt_sumlevels", fam, [full], mk, "frozen", extra, export=False)          # v7's score (all levels summed), for comparison
    best = S.row("sid_gpt")["cfg"]
    if args.sid_shuf_n:                                   # shuffled-history control on subsets: the same model and score, another reader's history
        sub = {sp: S.subset(sp, args.sid_shuf_n) for sp in ("val", "test")}
        shuf = SID.sid_level_scorer(model, codes, args.sid_nmax, device=S.device, shuffle=True)
        sc = {sp: SID.LevelCache(sub[sp][1], shuf) for sp in sub}
        S.select("sid_gpt_shuffled", fam, [dict(best)], lambda c, sp: sc[sp].scorer(c["levels"], S.device, base if c["pmi"] else None), "sid_gpt",
                 extra, es={sp: sub[sp][1] for sp in sub}, export=False, pop=False)
    del model, cache
    if str(S.device).startswith("cuda"):
        torch.cuda.empty_cache()


# ============================================================================ H1 + a learned head
class _ResMLP(nn.Module):
    """``x + W2 gelu(W1 LN(x))`` with ``W2 = 0`` at the start: the identity."""

    def __init__(self, d, hidden, drop):
        super().__init__()
        self.ln, self.w1, self.w2, self.drop = nn.LayerNorm(d), nn.Linear(d, hidden), nn.Linear(hidden, d), nn.Dropout(drop)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x):
        return x + self.w2(self.drop(F.gelu(self.w1(self.ln(x)))))


class HistHead(nn.Module):
    """Residual heads on the frozen history-as-text query vector and on the frozen article vector; score = scale * <q', c'>."""

    def __init__(self, d, hidden=1024, drop=0.1):
        super().__init__()
        self.q, self.c = _ResMLP(d, hidden, drop), _ResMLP(d, hidden, drop)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def logits(self, qv, cv):
        return torch.exp(self.log_scale) * (self.q(qv)[:, None, :] * self.c(cv)).sum(-1)

    @torch.no_grad()
    def transform(self, X, side, device, bs=8192):
        """``X`` numpy [n, d] -> the head's output, numpy (chunks on ``device``)."""
        self.eval()
        f = self.q if side == "q" else self.c
        out = np.zeros_like(X, dtype=np.float32)
        for s in range(0, len(X), bs):
            out[s:s + bs] = f(torch.from_numpy(np.ascontiguousarray(X[s:s + bs], np.float32)).to(device)).float().cpu().numpy()
        return out


def _train_hist_head(Qtr, samples, E, val_fn, d, args, seed, device):
    """Reader-style training (1 click + the sampled skipped articles, cross-entropy), early stopping on ``val_fn(model)`` -> ``(model, info)``.  Epoch 0
    (the identity = H1) competes."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HistHead(d, args.hm_hidden, args.hm_dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.hm_lr, weight_decay=1e-5)
    qi, pos, negs = samples
    Qt = torch.from_numpy(Qtr).to(device)
    Et = torch.from_numpy(np.ascontiguousarray(E, np.float32)).to(device)
    cands = torch.from_numpy(np.concatenate([pos[:, None], negs], 1)).to(device)
    qi_t = torch.from_numpy(qi).to(device)
    best, best_state, best_ep, bad, hist = val_fn(model), None, 0, 0, []
    hist.append(best)
    for ep in range(1, args.hm_epochs + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(qi))).to(device)
        tot, nb = 0.0, 0
        for s in range(0, len(qi), args.hm_bs):
            ix = perm[s:s + args.hm_bs]
            lo = model.logits(Qt[qi_t[ix]], Et[cands[ix]])
            loss = F.cross_entropy(lo, torch.zeros(len(ix), dtype=torch.long, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        v = val_fn(model)
        hist.append(v)
        log(f"      [histq_mlp seed {seed}] epoch {ep} loss={tot / max(nb, 1):.4f} val nDCG@10={v:.4f}")
        if v > best + args.um_min_delta:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.hm_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    del Qt, Et, cands, qi_t
    return model, {"best_epoch": best_ep, "best_val": float(best), "val_curve": [float(x) for x in hist]}


def stage_histq_mlp(S, args):
    log("\n== H1 + learned head: the frozen LLM encoder reads the history as text (instruction query), residual MLP heads trained like a reader ==")
    K, fam = args.hist_k, "H1-text-user"
    hq = HistQueries(S.data, S.titles, K)
    pre = query_prefix(args.encoder, args.query_format, True)
    rows = S.data.train_core
    keys, sq = {}, np.full(len(rows), -1, np.int64)
    for j, r in enumerate(rows):
        h = tuple(r[1][-K:])
        if h:
            sq[j] = keys.setdefault(h, len(keys))
    uniq = list(keys)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.permutation(len(uniq))[:args.hm_train_n])
    remap = np.full(len(uniq), -1, np.int64)
    remap[pick] = np.arange(len(pick))
    keep = np.nonzero((sq >= 0) & (remap[np.clip(sq, 0, None)] >= 0))[0]
    samples = (remap[sq[keep]], np.array([rows[j][2] for j in keep], np.int64), np.array([rows[j][3] for j in keep], np.int64))
    log(f"  {len(uniq):,} distinct training histories (last {K} headlines); {len(pick):,} encoded -> {len(keep):,} training samples; "
        f"validation / test: {len(hq.uniq):,} distinct histories")
    devs = FD.gpu_devices(S.device)
    dtype = adapt.resolve_dtype(args.encoder, getattr(args, "enc_dtype", "auto"), S.device)
    qpath = os.path.join(S.work, "histq_vectors.npz")
    if os.path.exists(qpath):
        z = np.load(qpath)
        Qvt, Qtr = z["vt"], z["tr"]
        log(f"  query vectors loaded from {qpath}")
    else:
        t0 = time.time()
        Qvt = np.zeros((max(len(hq.uniq), 1), S.E0.shape[1]), np.float32)
        if hq.uniq:
            Qvt[:len(hq.uniq)] = adapt.encode_multi(args.encoder, hq.texts(), args.q_max_len, pre, dtype, args.llm_enc_bs, devs, log=log)
        log(f"  {len(hq.uniq):,} validation / test histories encoded in {time.time() - t0:.0f}s on {len(devs)} device(s)")
        t0 = time.time()
        tr_texts = ["; ".join(S.titles[i] for i in uniq[k]) for k in pick]
        Qtr = adapt.encode_multi(args.encoder, tr_texts, args.q_max_len, pre, dtype, args.llm_enc_bs, devs, log=log)
        log(f"  {len(tr_texts):,} training histories encoded in {time.time() - t0:.0f}s")
        np.savez(qpath, vt=Qvt, tr=Qtr)
    S.kv["histq_mlp"] = {"train_histories": int(len(pick)), "train_samples": int(len(keep)), "valtest_histories": int(len(hq.uniq))}
    Et = S.tensor(S.E0)
    S.select("histq_instr", fam, [{"instruct": True, "k": K}], lambda c, sp: hq.scorer(Qvt, Et, sp, S.device), "frozen", export=True, kind="aux")
    vids, ves = S.subset("val", args.hm_val_n)
    need_v = hq.needed("val", vids)

    def scorer_of(model, split, ids=None, need=None):
        Qh = np.zeros_like(Qvt)
        need = hq.needed(split, ids) if need is None else need
        Qh[need] = model.transform(Qvt[need], "q", S.device)
        Ch = S.tensor(model.transform(S.E0, "c", S.device))
        Ch[0] = 0
        return hq.scorer(Qh, Ch, split, S.device, ids=ids)

    def val_fn(model):
        return S.run(scorer_of(model, "val", vids, need_v), "val", es={"val": ves})["mean"]["ndcg@10"]

    vals, tests, vrag, trag, infos = [], [], [], [], []
    for sd in range(args.hm_seeds):
        t0 = time.time()
        model, info = _train_hist_head(Qtr, samples, S.E0, val_fn, S.E0.shape[1], args, sd, S.device)
        (v_, vr_), (t_, tr_) = S.run_rec(scorer_of(model, "val"), "val"), S.run_rec(scorer_of(model, "test"), "test")
        vals.append(v_), tests.append(t_), vrag.append(vr_), trag.append(tr_), infos.append(info)
        log(f"    seed {sd}: best epoch {info['best_epoch']} (validation subset {info['best_val']:.4f}), test nDCG@10 {t_['mean']['ndcg@10']:.4f} ({time.time() - t0:.0f}s)")
        del model
    per_seed = [t["mean"]["ndcg@10"] for t in tests]
    S.record("histq_mlp", fam, {"hidden": args.hm_hidden, "seeds": args.hm_seeds, "train_histories": int(len(pick))}, avg_results(vals), avg_results(tests),
             "histq_instr", {"best_epochs": [i["best_epoch"] for i in infos], "val_curves": [i["val_curve"] for i in infos], "ndcg10_per_seed": per_seed,
                             **({"ndcg10_std": float(np.std(per_seed, ddof=1))} if len(per_seed) > 1 else {})})
    S.export_scores("histq_mlp", common.ragged_mean_z(vrag), common.ragged_mean_z(trag), {"family": fam, "seeds": len(per_seed), "kind": "reader"})
    if str(S.device).startswith("cuda"):
        torch.cuda.empty_cache()


STAGES = {"kar_full": stage_kar_full, "sid2": stage_sid2, "histq_mlp": stage_histq_mlp}


def add_args(ap):
    """The v20 arguments (``suite.build_parser`` calls this)."""
    ap.add_argument("--kar-gen-frac", type=float, default=0.55, help="kar_full: share of the remaining budget the generation of the expansions may use")
    ap.add_argument("--sid-text", default="ta", choices=["ta", "t", "cta", "ctae"], help="sid2: the text variant whose vectors the codes are fitted on")
    ap.add_argument("--sid-levelsets", default="1,12,123", help="sid2: code-level sets whose summed log-probabilities are tried on validation (e.g. 1 = topic level only)")
    ap.add_argument("--sid-shuf-n", type=int, default=20000, help="sid2: impressions of the shuffled-history control (0 = skip)")
    ap.add_argument("--hm-train-n", type=int, default=300000, help="histq_mlp: distinct training histories encoded (each gives all its training samples)")
    ap.add_argument("--hm-val-n", type=int, default=20000, help="histq_mlp: validation impressions of the early-stopping checks")
    ap.add_argument("--hm-hidden", type=int, default=1024)
    ap.add_argument("--hm-dropout", type=float, default=0.1)
    ap.add_argument("--hm-lr", type=float, default=1e-3)
    ap.add_argument("--hm-bs", type=int, default=512)
    ap.add_argument("--hm-epochs", type=int, default=6)
    ap.add_argument("--hm-patience", type=int, default=2)
    ap.add_argument("--hm-seeds", type=int, default=2)
    return ap
