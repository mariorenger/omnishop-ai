"""Stage functions of the V11 suite beyond the original four (controls/graph/head/encoder/histquery):

  pooling       H8   recency / late-interaction pooling over frozen vectors
  sid           H7   Semantic IDs: RQ-KMeans profile + TIGER-style generative slate ranker
  sweep         --   encoder-scale sweep (+ lexical TF-IDF/LSA floor)
  llm_user      H5   LoRA-tuned LLM user tower, causal / bidirectional / soft attention mask
  usermodel     H10  learned candidate-aware user model (V10 ``llmenc_ca``) on top of each representation
  graphrag_llm  H4b  GraphRAG with LLM-written community summaries
  augment       H6   KAR-style LLM article expansions and reader profiles
  rerank        H9   zero-shot LLM judge re-ranking the top-k

Every function takes the ``Suite`` object ``S`` (see suite.py) and the parsed ``args``.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import torch

import adapt
import llm_gen
import pooling as P
import rep_eval as R
import semgraph as G
import sid as SID
import usermodel as UM
from common import HistQueries, ScoreCache, avg_results, decoder_only, fuse, log, query_prefix, subset_evalset


# ============================================================================ shared helpers
def embed_news(meta, name, max_len, prefix, device, bs, cache_dir, dtype=torch.float32):
    """Frozen article vectors ``[n_articles, d]``, cached on disk per (model, max_len, prefix) so that the encoder
    sweep and a later run with the same model never encode twice.  Returns ``(vecs, seconds)`` (0 if cached)."""
    key = f"{name}|{max_len}|{prefix}|{len(meta['ids'])}"
    path = os.path.join(cache_dir, "emb_cache_" + hashlib.sha1(key.encode()).hexdigest()[:12] + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        if str(z["key"]) == key:
            return z["vecs"], 0.0
    enc = adapt.HFEncoder(name, max_len=max_len, device=device, prefix=prefix, dtype=dtype)
    t0 = time.time()
    vecs = enc.encode(meta["text"], bs=bs)
    secs = time.time() - t0
    del enc
    if device == "cuda":
        torch.cuda.empty_cache()
    if not np.isfinite(vecs).all():
        raise RuntimeError(f"{name}: non-finite embeddings (fp16 overflow?)")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, key=np.array(key), vecs=vecs)
    return vecs, secs


def short(name):
    return (name.split("/")[-1].replace("-en-v1.5", "").replace("Qwen3-Embedding-", "qwen3emb-").lower()
            .replace("-", "").replace(".", ""))


def table_from(S, vecs):
    E = np.zeros((S.data.n_news, vecs.shape[1]), np.float32)
    E[1:] = vecs
    return E


def load_export(S, name):
    z = np.load(os.path.join(S.work, f"emb_{name}.npz"), allow_pickle=True)
    return table_from(S, z["vecs"])


def sec_split(total, frac):
    return max(int(total * frac), 0)


# ============================================================================ H8 pooling
def stage_pooling(S, args):
    log("\n== H8 pooling rule over the frozen vectors (recency / late interaction) ==")
    E = S.tensor(S.E0)
    fam = "H8-pooling"
    S.select("pool_recency", fam, [{"lam": l} for l in args.pool_lams], lambda c, sp: P.recency_scorer(E, c["lam"]), "frozen")
    S.select("pool_maxsim", fam, [{}], lambda c, sp: P.late_scorer(E, "max"), "frozen")
    S.select("pool_topk", fam, [{"k": k} for k in (3, 5)], lambda c, sp: P.late_scorer(E, "topk", c["k"]), "frozen")
    S.select("pool_lse", fam, [{"tau": t} for t in (0.05, 0.1)], lambda c, sp: P.late_scorer(E, "lse", c["tau"]), "frozen")


# ============================================================================ H7 Semantic IDs
def stage_sid(S, args):
    log("\n== H7 Semantic IDs: RQ-KMeans tokenizer, zero-shot profile, TIGER-style generative slate ranker ==")
    K, L, fam = args.sid_k, args.sid_levels, "H7-SemanticID"
    t0 = time.time()
    codes = SID.rq_kmeans(S.E0, S.seen_tr | S.seen_va, L=L, K=K, seed=0, device=S.device)     # content only; cold articles get codes by nearest centroid
    pref = SID.prefix_ids(codes, K)
    n = S.data.n_news - 1
    log("  codebook " + ", ".join(f"L{l + 1}: {len(np.unique(pref[l][1:])):,} prefixes" for l in range(L))
        + f" for {n:,} articles ({time.time() - t0:.0f}s)")
    Et = S.tensor(S.E0)
    mp = R.mean_pool_scorer(Et)
    levels = [tuple(range(1, l + 1)) for l in range(1, L + 1)]
    S.select("sid_profile", fam, [{"levels": lv} for lv in levels],
             lambda c, sp: SID.sid_profile_scorer(codes, K, c["levels"], S.device), "frozen")
    S.select("sid_profile+meanpool", fam, [{"levels": lv, "w": w} for lv in levels for w in args.sid_ws],
             lambda c, sp: fuse([(mp, 1.0), (SID.sid_profile_scorer(codes, K, c["levels"], S.device), c["w"])]), "frozen")
    if S.time_left() < 1200:
        log("  SID-GPT skipped: wall-clock budget")
        return
    val_ids, val_es = S.subset("val", args.sid_val_n)

    def val_fn(model):
        return S.run(SID.sid_scorer(model, codes, args.sid_nmax, False, device=S.device), "val", es={"val": val_es})["mean"]["ndcg@10"]

    model, info = SID.train_sid_gpt(codes, K, S.data.train_core, n_max=args.sid_nmax, epochs=args.sid_epochs, bs=args.sid_bs,
                                    lr=args.sid_lr, seed=0, device=S.device, val_fn=val_fn, d=args.sid_d, layers=args.sid_layers,
                                    heads=args.sid_heads, max_steps=args.sid_steps or None, deadline=S.deadline(0.25), log=log)
    params = sum(p.numel() for p in model.parameters())
    log(f"  SID-GPT: {params:,} parameters, best epoch {info['best_epoch']}, {info['steps']} steps")
    codes_t = torch.from_numpy(codes).to(S.device)
    base = SID.history_free_logprob(model, codes_t, len(codes)).to(S.device)
    t1 = time.time()
    raw = SID.sid_scorer(model, codes, args.sid_nmax, False, device=S.device)
    cache = {sp: ScoreCache(S.es[sp], raw) for sp in ("val", "test")}
    log(f"  scored every validation/test slate once in {time.time() - t1:.0f}s (cached for the fusion grid)")
    lik = {sp: cache[sp].scorer(S.device) for sp in cache}
    pmi = {sp: cache[sp].scorer(S.device, shift=lambda b: base[b["cand"]]) for sp in cache}
    extra = {"params": params, "best_epoch": info["best_epoch"], "steps": info["steps"]}
    S.select("sid_gpt", fam, [{"score": "log P(codes|history)"}], lambda c, sp: lik[sp], "frozen", extra)
    S.select("sid_gpt_pmi", fam, [{"score": "log P(codes|history) - log P(codes)"}], lambda c, sp: pmi[sp], "frozen", extra)
    src = {"lik": lik, "pmi": pmi}
    S.select("sid_gpt+meanpool", fam, [{"src": s, "w": w} for s in src for w in args.sid_ws],
             lambda c, sp: fuse([(mp, 1.0), (src[c["src"]][sp], c["w"])]), "frozen", extra)
    del model
    if S.device == "cuda":
        torch.cuda.empty_cache()


# ============================================================================ encoder sweep
def stage_sweep(S, args):
    log("\n== encoder sweep: frozen zero-shot mean-pool (does a bigger / LLM-based encoder help at all?) ==")
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    t0 = time.time()
    X = TfidfVectorizer(max_features=60_000, stop_words="english", sublinear_tf=True, min_df=2).fit_transform(S.meta["text"])
    Z = TruncatedSVD(min(args.lsa_dim, X.shape[1] - 1), random_state=0).fit_transform(X).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    Et = S.tensor(table_from(S, Z))
    S.record("enc_tfidf_lsa", "enc-sweep", {"model": f"tfidf+svd{Z.shape[1]}", "dim": int(Z.shape[1])},
             S.run(R.mean_pool_scorer(Et), "val"), S.run(R.mean_pool_scorer(Et), "test"), "frozen", {"encode_s": time.time() - t0})
    names = [n for n in args.sweep_encoders.split(",") if n and n != args.encoder]
    for name in names:
        if S.time_left() < 1200:
            log(f"  {name}: skipped (wall-clock budget)")
            continue
        try:
            vecs, secs = embed_news(S.meta, name, args.max_len, "", S.device, args.enc_bs, S.cache_dir)
        except Exception as e:                                            # a missing / unloadable model must not stop the sweep
            log(f"  {name}: failed to load/encode ({repr(e)[:120]}) - skipped")
            continue
        Et = S.tensor(table_from(S, vecs))
        S.record(f"enc_{short(name)}", "enc-sweep", {"model": name, "dim": int(vecs.shape[1])},
                 S.run(R.mean_pool_scorer(Et), "val"), S.run(R.mean_pool_scorer(Et), "test"), "frozen", {"encode_s": secs})
        del Et


# ============================================================================ H5 LLM user tower
def stage_llm_user(S, args):
    from llm_user import UserTower, probe_masks, train_user_tower
    log("\n== H5 fine-tuned LLM user tower: history-as-text, LoRA, causal / bidirectional / soft attention mask ==")
    dec = decoder_only(args.encoder)
    modes = [m for m in args.llm_user_modes.split(",") if m] if dec else ["native"]
    K, fam = args.hist_k, "H5-LLM-user"
    hq = HistQueries(S.data, S.titles, K)
    sets = {sp: S.subset(sp, n) for sp, n in (("val", args.llm_val_n), ("test", args.llm_test_n))}
    es = {sp: sets[sp][1] for sp in sets}
    log(f"  evaluation on {len(sets['val'][0]):,} validation / {len(sets['test'][0]):,} test impressions; modes={modes}")
    E_doc = S.tensor(S.E0)
    mp = R.mean_pool_scorer(E_doc)
    pre = query_prefix(args.encoder, args.query_format, True)
    qfn = lambda h: "; ".join(S.titles[i] for i in h[-K:])
    probe_texts = hq.texts(range(min(3, len(hq.uniq))))

    def encode_split(tw, sp):
        qi = hq.needed(sp, sets[sp][0])
        Q = hq.alloc(E_doc.shape[1])
        Q[qi] = tw.encode(hq.texts(qi), bs=args.llm_enc_bs, deadline=S.deadline(0.3))
        missing = int((np.abs(Q[qi]).sum(1) == 0).sum())
        if missing:
            log(f"      WARNING: {missing:,}/{len(qi):,} {sp} queries were not encoded (wall-clock budget) - their rows score as ties")
            S.kv.setdefault("queries_missing", {})[f"{tw.mode}/{sp}"] = missing
        return Q

    for mode in modes:
        if S.time_left() < 1800:
            log(f"  {mode}: skipped (wall-clock budget)")
            continue
        tw = UserTower(args.encoder, mode, prefix=pre, max_len=args.q_max_len, device=S.device)
        if dec and mode != "causal":
            ok, rep = probe_masks(tw, probe_texts)
            log(f"  attention-mask probe for '{mode}': {'OK' if ok else 'FAILED -> variant skipped'} {({k: round(v, 5) if isinstance(v, float) else v for k, v in rep.items()})}")
            if not ok:
                del tw
                continue

        def eval_fn(tw):
            Q = encode_split(tw, "val")
            tw.eval_payload = Q
            return S.run(hq.scorer(Q, E_doc, "val", S.device, ids=sets["val"][0]), "val", es=es)["mean"]["ndcg@10"]

        info = train_user_tower(tw, qfn, S.data.train_core, E_doc, kind="lora", steps=args.llm_user_steps, bs=args.llm_user_bs,
                                lr=args.llm_user_lr, r=args.llm_user_r, eval_fn=eval_fn, deadline=S.deadline(0.3), log=log)
        info["truncated"] = bool(info["steps"] < args.llm_user_steps)
        if info["truncated"]:
            log(f"      WARNING: '{mode}' trained {info['steps']}/{args.llm_user_steps} steps (wall-clock budget) - not comparable with the other masks")
        Qv = info.pop("payload") if info.get("payload") is not None else encode_split(tw, "val")
        Qt = encode_split(tw, "test")
        info.pop("losses", None)
        sc = {"val": hq.scorer(Qv, E_doc, "val", S.device, ids=sets["val"][0]),
              "test": hq.scorer(Qt, E_doc, "test", S.device, ids=sets["test"][0])}
        S.select(f"ut_{mode}", fam, [{"mode": mode, "steps": args.llm_user_steps, "r": args.llm_user_r}], lambda c, sp: sc[sp],
                 "frozen", info, es=es)
        S.select(f"ut_{mode}+meanpool", fam, [{"mode": mode, "w": w} for w in args.ut_ws],
                 lambda c, sp: fuse([(mp, 1.0), (sc[sp], c["w"])]), "frozen", info, es=es)
        del tw
        if S.device == "cuda":
            torch.cuda.empty_cache()


# ============================================================================ H10 learned user model
def _views(S, args):
    """Representations to feed the learned user model -> {name: {"val": E, "test": E}} (RAG views depend on the period)."""
    out = {}
    wanted = [v for v in args.um_views.split(",") if v]
    for v in wanted:
        try:
            if v == "frozen":
                out[v] = {"val": S.E0, "test": S.E0}
            elif v == "head":
                E = load_export(S, S.kv["best_head"])
                out[v] = {"val": E, "test": E}
            elif v == "encoder":
                rows = [r for r in S.records if r["name"].startswith(("lora", "dora")) and r.get("val_delta") is not None]
                E = load_export(S, max(rows, key=lambda r: r["val_delta"])["name"])
                out[v] = {"val": E, "test": E}
            elif v == "entity_rag":
                cfg, memo = S.row("entity_rag")["cfg"], S.memo()
                out[v] = {sp: memo[sp].entity_view(cfg["beta"])[0] for sp in ("val", "test")}
            elif v == "knn_rag":
                cfg, memo = S.row("knn_rag")["cfg"], S.memo()
                out[v] = {sp: memo[sp].knn_view(cfg["beta"], cfg["k"]) for sp in ("val", "test")}
            else:
                log(f"  unknown view '{v}' ignored")
        except (KeyError, FileNotFoundError, ValueError) as e:
            log(f"  view '{v}' unavailable ({repr(e)[:80]}) - did its stage run?")
    return out


def stage_usermodel(S, args):
    log("\n== H10 learned candidate-aware user model (V10 llmenc_ca) on each representation ==")
    pack = S.store.get("um_pack") or UM.pack_train(S.data.train_core)
    S.store["um_pack"] = pack
    views = _views(S, args)
    fam, ref = "H10-user-model", "um_frozen"
    for vname, V in views.items():
        if S.time_left() < 900:
            log(f"  {vname}: skipped (wall-clock budget)")
            continue
        vals, tests, infos = [], [], []
        for sd in range(args.um_seeds):
            t0 = time.time()

            def val_fn(model):
                return S.run(UM.scorer(model), "val")["mean"]["ndcg@10"]

            model, info = UM.train_user_model(V["val"], pack, val_fn, seed=sd, max_epochs=args.um_epochs, device=S.device,
                                              deadline=S.deadline(0.2), log=lambda *a: None)
            vals.append(S.run(UM.scorer(model, V["val"]), "val"))
            tests.append(S.run(UM.scorer(model, V["test"]), "test"))
            infos.append(info)
            log(f"    {vname} seed {sd}: {info['epochs']} epochs, val nDCG@10={info['best_val']:.4f} ({time.time() - t0:.0f}s)")
        row = S.record(f"um_{vname}", fam, {"view": vname, "seeds": args.um_seeds}, avg_results(vals), avg_results(tests),
                       ref if ref in S.cache else f"um_{vname}",
                       {"epochs": [i["epochs"] for i in infos], "ndcg10_per_seed": [t["mean"]["ndcg@10"] for t in tests]})
        if vname == "frozen" and args.um_repro:
            auc_ref, nd_ref = (float(x) for x in args.um_repro.split(","))
            ok = abs(row["auc"] - auc_ref) < 0.01 and abs(row["ndcg@10"] - nd_ref) < 0.01
            log(f"  reproduction check vs V10 llmenc_ca (AUC {auc_ref}, nDCG@10 {nd_ref}): "
                f"{'MATCH (within 0.01)' if ok else 'DIFFERS - single V10 seed, our seeds average; inspect if the gap is > 0.01'}")
        if S.device == "cuda":
            torch.cuda.empty_cache()


# ============================================================================ generation-based stages
COMM_PROMPT = ("Below are headlines from one cluster of related news articles.\n{titles}\n\n"
               "In one sentence (at most 40 words), describe the common topic of this cluster and what kind of reader would follow it.")
EXPAND_PROMPT = ("News headline: {title}\nSummary: {abstract}\n\n"
                 "In one or two sentences (at most 50 words), state the main topic of this article and what kind of reader would be interested in it.")
PROFILE_PROMPT = ("A reader recently clicked these news headlines (oldest first):\n{titles}\n\n"
                  "In two short sentences (at most 60 words), describe this reader's interests and what other news they would want to read next.")
JUDGE_PROMPT = ("A news reader recently read these articles (oldest first):\n{hist}\n\n"
                "Candidate article: \"{title}\" (category: {cat})\n\n"
                "Would this reader be interested in clicking the candidate article? Answer with exactly one word: Yes or No.")


def stage_graphrag_llm(S, args):
    log("\n== H4b GraphRAG with LLM-written community summaries ==")
    gen, enc = S.generator(), S.doc_encoder()
    comm, memo = S.communities(), S.memo()
    T, fam = S.titles, "H4b-GraphRAG-LLM"
    summ = {}
    for sp in ("val", "test"):
        labels, cent, method = comm[sp]
        mem_ids = memo[sp].mem_ids
        K = len(cent)
        size = np.bincount(labels[mem_ids], minlength=K)
        order = np.argsort(-size)[:args.gr_max_comm]
        prompts = []
        for k in order:
            members = mem_ids[labels[mem_ids] == k]
            top = members[np.argsort(-(S.E0[members] @ cent[k]))[:args.gr_titles]]
            prompts.append(COMM_PROMPT.format(titles="\n".join(f"- {T[i]}" for i in top)))
        out = gen.generate(prompts, max_new_tokens=64, bs=args.gen_bs, deadline=S.deadline(0.1), desc=f"community summaries [{sp}]")
        cent_llm = cent.copy()
        ok = [(k, t) for k, t in zip(order, out) if t]
        if ok:
            V = enc.encode([t for _, t in ok], bs=64)
            cent_llm[[k for k, _ in ok]] = V
        summ[sp] = cent_llm
        log(f"  [{sp}] {len(ok)}/{K} communities summarised by the LLM ({method}); example: {ok[0][1][:110] if ok else '-'!r}")
    S.select("community_llm", fam, [{"lam": l} for l in args.lams],
             lambda c, sp: G.community_scorer(S.E0, comm[sp][0], summ[sp], c["lam"], S.device), "frozen")

    def ctx_view(c, sp):
        E = S.E0.copy()
        E[1:] = G._norm(S.E0[1:] + c["beta"] * summ[sp][np.clip(comm[sp][0][1:], 0, None)])
        return E
    S.select("community_llm_ctx", fam, [{"beta": b} for b in args.betas],
             lambda c, sp: R.mean_pool_scorer(S.tensor(ctx_view(c, sp))), "frozen")
    del enc
    if S.device == "cuda":
        torch.cuda.empty_cache()


def _gen_subsets(S, args, gen, items_of, per_impr, req, kind, minutes, frac_val=0.3):
    """Choose how many validation/test impressions fit the generation allowance (probe throughput on a few items)."""
    allow = min(minutes * 60.0, 0.45 * S.time_left())
    spi = gen.seconds_per_item(kind, 1.0)
    out = {}
    for sp, share in (("test", 1.0 - frac_val), ("val", frac_val)):
        perm = S.perm(sp)
        lists = [items_of(sp, i) for i in perm[:req[sp]]]
        cap = allow * share / max(spi, 1e-3)
        n, n_items = llm_gen.plan_prefix(lists, cap, per_impr)
        out[sp] = (np.sort(perm[:n]), n_items)
    log(f"  allowance {allow / 60:.0f} min at {spi:.2f}s/item -> validation {len(out['val'][0]):,} / test {len(out['test'][0]):,} impressions "
        f"(requested {req['val']:,}/{req['test']:,})")
    return out


def stage_augment(S, args):
    log("\n== H6 LLM-augmented articles and readers (KAR-style expansions + LLM-written profiles) ==")
    gen, enc_d, enc_q = S.generator(), S.doc_encoder(), S.query_encoder()
    T, A, fam = S.titles, [""] + S.meta["abstracts"], "H6-LLM-aug"
    hk, K = args.gen_hist_k, args.hist_k
    rows = {"val": S.data.validation, "test": S.data.test}
    items_of = lambda sp, i: list(rows[sp][i][3]) + list(rows[sp][i][2][-hk:])
    # probe throughput on a handful of expansions (kept in the cache) before sizing the subsets
    probe = list(dict.fromkeys(items_of("test", S.perm("test")[0])))[:32]
    gen.generate([EXPAND_PROMPT.format(title=T[i], abstract=A[i][:300]) for i in probe], max_new_tokens=args.gen_new_tokens,
                 bs=args.gen_bs, desc="probe")
    plan = _gen_subsets(S, args, gen, items_of, 1.3, {"val": args.gen_val_n, "test": args.gen_test_n}, "gen", args.augment_minutes)
    need = sorted({i for sp in plan for r in plan[sp][0] for i in items_of(sp, r)})
    log(f"  expanding {len(need):,} distinct articles")
    out = gen.generate([EXPAND_PROMPT.format(title=T[i], abstract=A[i][:300]) for i in need], max_new_tokens=args.gen_new_tokens,
                       bs=args.gen_bs, deadline=S.deadline(0.3), desc="article expansions")
    exp = {i: t for i, t in zip(need, out) if t}
    ok = np.array(sorted(exp))
    Eg = np.zeros_like(S.E0)
    Eg[ok] = enc_d.encode([T[i] + ". " + exp[i] for i in ok], bs=64)
    log(f"  {len(ok):,}/{len(need):,} expansions embedded; example: {exp[ok[0]][:120]!r}" if len(ok) else "  no expansion produced")
    # keep only impressions whose every needed article was expanded
    ids, es = {}, {}
    for sp in plan:
        keep = [r for r in plan[sp][0] if all(i in exp for i in items_of(sp, r))]
        ids[sp] = np.array(keep, np.int64)
        es[sp] = subset_evalset(rows[sp], ids[sp], S.device)
    log(f"  fully covered impressions: validation {len(ids['val']):,}, test {len(ids['test']):,}")
    if len(ids["test"]) < 50 or len(ids["val"]) < 20:
        log("  too few covered impressions - stage aborted")
        return
    E0t = S.tensor(S.E0)
    mp0 = R.mean_pool_scorer(E0t)
    aug = lambda a: S.tensor(G._norm(S.E0 + a * Eg))
    S.select("kar_item", fam, [{"alpha": a} for a in args.kar_alphas], lambda c, sp: R.mean_pool_scorer(aug(c["alpha"])), "frozen", es=es)
    best_alpha = S.row("kar_item")["cfg"]["alpha"]
    # reader profiles written by the LLM from the last K headlines
    hq = HistQueries(S.data, T, K)
    need_q = np.unique(np.concatenate([hq.needed(sp, ids[sp]) for sp in ids]))
    prof = gen.generate([PROFILE_PROMPT.format(titles="\n".join(f"- {T[i]}" for i in hq.uniq[q])) for q in need_q],
                        max_new_tokens=args.gen_new_tokens + 16, bs=args.gen_bs, deadline=S.deadline(0.3), desc="reader profiles")
    got = [(q, t) for q, t in zip(need_q, prof) if t]
    Q = hq.alloc(S.E0.shape[1])
    if got:
        Q[[q for q, _ in got]] = enc_q.encode([t for _, t in got], bs=64)
        log(f"  {len(got):,}/{len(need_q):,} reader profiles embedded; example: {got[0][1][:130]!r}")
    Eb = aug(best_alpha)
    mp_b = R.mean_pool_scorer(Eb)
    prof_sc = lambda E, sp: hq.scorer(Q, E, sp, S.device, ids=ids[sp])
    S.select("kar_user", fam, [{}], lambda c, sp: prof_sc(E0t, sp), "frozen", es=es)
    S.select("kar_user+meanpool", fam, [{"w": w} for w in args.kar_ws], lambda c, sp: fuse([(mp0, 1.0), (prof_sc(E0t, sp), c["w"])]),
             "frozen", es=es)
    S.select("kar_item+user", fam, [{"alpha": best_alpha, "w": w} for w in args.kar_ws],
             lambda c, sp: fuse([(mp_b, 1.0), (prof_sc(Eb, sp), c["w"])]), "frozen", es=es)
    del enc_d, enc_q
    if S.device == "cuda":
        torch.cuda.empty_cache()


def stage_rerank(S, args):
    log("\n== H9 zero-shot LLM judge re-ranking the top-k of the frozen embedding (pointwise Yes/No log-odds) ==")
    gen = S.generator()
    T, cats, fam, k, hk = S.titles, [""] + S.meta["cats"], "H9-LLM-rerank", args.rerank_k, args.rerank_hist
    first = R.mean_pool_scorer(S.tensor(S.E0))
    rows = {"val": S.data.validation, "test": S.data.test}

    def build(sp, n):
        ids, es = S.subset(sp, n)
        cache = ScoreCache(es, first)
        top, prompts = {}, []
        for pos, orig in enumerate(ids):
            s = cache.rows[pos]
            tk = np.argsort(-s, kind="stable")[:k]
            row = rows[sp][orig]
            htxt = "\n".join(f"- {T[i]}" for i in list(row[2])[-hk:]) or "- (nothing yet)"
            top[int(orig)] = tk
            prompts += [JUDGE_PROMPT.format(hist=htxt, title=T[row[3][j]], cat=cats[row[3][j]]) for j in tk]
        return ids, cache, top, prompts

    t_ids, t_cache, t_top, t_prompts = build("test", args.rerank_test_n)
    gen.yes_no(t_prompts[:64], bs=args.gen_bs, desc="probe")
    spp = gen.seconds_per_item("yn", 0.05)
    allow = min(args.rerank_minutes * 60.0, 0.5 * S.time_left())
    prompts_ok = int(allow / max(spp, 1e-3))
    n_test = int(min(args.rerank_test_n, 0.7 * prompts_ok / k))
    n_val = int(min(args.rerank_val_n, 0.3 * prompts_ok / k))
    log(f"  {spp * 1000:.0f} ms/prompt -> allowance {allow / 60:.0f} min: re-rank {n_val:,} validation / {n_test:,} test impressions (top-{k})")
    if n_test < 50 or n_val < 20:
        log("  allowance too small - stage skipped")
        return
    sets = {}
    for sp, n in (("val", n_val), ("test", n_test)):
        ids, cache, top, prompts = build(sp, n)
        lo = gen.yes_no(prompts, bs=args.gen_bs, deadline=S.deadline(0.45), desc=f"judge [{sp}]")
        off = np.concatenate([[0], np.cumsum([len(top[int(o)]) for o in ids])])
        keep = [i for i in range(len(ids)) if not np.isnan(lo[off[i]:off[i + 1]]).any()]      # impressions fully judged
        sets[sp] = {"ids": ids[keep], "first": {int(ids[i]): cache.rows[i] for i in keep},
                    "top": {int(ids[i]): top[int(ids[i])] for i in keep}, "lo": {int(ids[i]): lo[off[i]:off[i + 1]] for i in keep}}
        log(f"  [{sp}] {len(keep):,}/{len(ids):,} impressions fully judged")
    es = {sp: subset_evalset(rows[sp], sets[sp]["ids"], S.device) for sp in sets}
    zs = lambda x: (x - x.mean()) / (x.std() + 1e-6)

    def make(c, sp):
        d, w = sets[sp], c["w"]

        def f(b):
            out = np.full((len(b["rows"]), b["cand"].shape[1]), -1e4, np.float32)
            for j, p in enumerate(b["rows"]):
                o = int(d["ids"][p])
                s1, tk, lo = d["first"][o], d["top"][o], d["lo"][o]
                base = s1 - 1e3                                    # everything outside the re-ranked group stays below it
                base[tk] = (zs(lo) + 1e-3 * zs(s1[tk])) if w == "llm" else (zs(s1[tk]) + w * zs(lo))
                out[j, :len(base)] = base
            return torch.from_numpy(out).to(S.device)
        return f
    S.select("rerank_llm", fam, [{"w": "llm", "k": k, "model": args.gen_model}], make, "frozen", es=es)
    S.select("rerank_fused", fam, [{"w": w, "k": k, "model": args.gen_model} for w in args.rerank_ws], make, "frozen", es=es)
