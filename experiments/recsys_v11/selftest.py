"""End-to-end self-test on CPU: MIND-format fixture + tiny random BERT + tiny random Qwen3 (decoder).

Numbers are meaningless (random encoders, toy data) -- the point is that every code path of the suite runs
(LoRA/DoRA, Matryoshka, Louvain communities, user-RAG, history-as-text queries, pooling rules, Semantic IDs,
the LLM user tower with causal/bidirectional/soft masks, learned user model, encoder sweep, LLM generation,
reader profiles, LLM judge, paired CIs, exports, resume, wall-clock budget) and that key invariants hold:

  1. the vectorised evaluator reproduces the V10 ``metrics.py`` on the suite's own scores;
  2. every residual head is the identity before training (step 0 == frozen control);
  3. an adaptation that does not beat the frozen embedding on validation never replaces it;
  4. attention masks: causal ignores the future, bidirectional does not, soft interpolates; batched == single;
  5. Semantic-ID scoring: batched == per-sequence reference (empty history included); the model learns;
  6. generation / judging are batch-invariant and cached; subset comparisons equal manual slicing;
  7. resume skips finished stages, a tiny wall-clock budget skips the rest without crashing.

    python selftest.py            # ~4-7 min on CPU
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

import adapt
import common
import llm_gen
import llm_user
import metrics
import rep_eval as R
import sid as SID
import suite
import usermodel as UM


def build_fixture(root, seed=0, n_users=120, T=6):
    rng = random.Random(seed)
    words = {t: [f"t{t}w{j}" for j in range(25)] for t in range(T)}
    shared = [f"sw{j}" for j in range(30)]
    ent = {t: [f"Q{t}{k}" for k in range(5)] for t in range(T)}

    def article(i, t):
        title = " ".join(rng.choice(words[t]) for _ in range(6)) + " " + " ".join(rng.choice(shared) for _ in range(2))
        abstract = " ".join(rng.choice(words[t]) if rng.random() < 0.7 else rng.choice(shared) for _ in range(20))
        es = rng.sample(ent[t], 2) + ([f"Q9{rng.randrange(20)}"] if rng.random() < 0.3 else [])
        js = json.dumps([{"Label": e, "Type": "O", "WikidataId": e, "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": [e]} for e in es])
        return f"N{i}\tcat{t}\tsub{t}\t{title}\t{abstract}\thttp://x\t{js}\t[]"

    old = [(i, i % T) for i in range(1, 361)]
    new = [(i, i % T) for i in range(361, 451)]                 # appear on the test day only (cold articles)
    prefs = {u: rng.sample(range(T), rng.choice([1, 2])) for u in range(n_users)}
    by_topic = {t: [i for i, tt in old if tt == t] for t in range(T)}
    topic_of = {i: t for i, t in old + new}

    def impressions(day, n, pool, start_id):
        out = []
        for k in range(n):
            u = rng.randrange(n_users)
            hist = [rng.choice(by_topic[rng.choice(prefs[u])]) for _ in range(rng.randint(0, 10))]
            cands = rng.sample(pool, 10)
            labs = [1 if (topic_of[c] in prefs[u] and rng.random() < 0.8) or rng.random() < 0.05 else 0 for c in cands]
            if not any(labs):
                labs[0] = 1
            if all(labs):
                labs[-1] = 0
            ts = f"11/{day:02d}/2019 {rng.randint(1, 11)}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} {rng.choice(['AM', 'PM'])}"
            out.append(f"{start_id + k}\tU{u}\t{ts}\t{' '.join('N%d' % h for h in hist)}\t"
                       f"{' '.join('N%d-%d' % (c, l) for c, l in zip(cands, labs))}")
        return out

    os.makedirs(f"{root}/train", exist_ok=True)
    os.makedirs(f"{root}/dev", exist_ok=True)
    with open(f"{root}/train/news.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(article(i, t) for i, t in old) + "\n")
    with open(f"{root}/dev/news.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(article(i, t) for i, t in old + new) + "\n")
    tr = []
    for day in range(9, 14):
        tr += impressions(day, 260, [i for i, _ in old], len(tr) + 1)
    with open(f"{root}/train/behaviors.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(tr) + "\n")
    pool = [i for i, _ in old] + [i for i, _ in new] * 2          # new articles are over-represented on test day
    with open(f"{root}/dev/behaviors.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(impressions(15, 330, pool, 100_000)) + "\n")
    texts = [a.split("\t")[3] + ". " + a.split("\t")[4] for a in
             (article(i, t) for i, t in old + new)]
    return texts


def build_tiny_encoder(path, texts, seed=0):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=600, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tk.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tk.token_to_id("[CLS]")), ("[SEP]", tk.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(path)
    torch.manual_seed(seed)
    BertModel(BertConfig(vocab_size=len(fast), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=128, max_position_embeddings=128)).save_pretrained(path)


def build_tiny_decoder(path, texts, seed=0):
    """Tiny random Qwen3 (causal LM) + word-level tokenizer with a Qwen-style chat template (supports enable_thinking)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"]
    extra = ["Yes No user assistant system Instruct Query Given the headlines a reader recently clicked retrieve news article they "
             "would most likely read next Answer with Yes or No . ; : - Topic Reader interests Would this click"]
    tk.train_from_iterator(list(texts) + extra, trainers.WordLevelTrainer(vocab_size=900, special_tokens=specials))
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<|endoftext|>", eos_token="<|endoftext|>")
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                          "{% if add_generation_prompt %}<|im_start|>assistant\n"
                          "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}")
    fast.save_pretrained(path)
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=1024, tie_word_embeddings=False,
                      pad_token_id=fast.pad_token_id, eos_token_id=fast.eos_token_id)
    Qwen3ForCausalLM(cfg).save_pretrained(path)




def ok(msg):
    print("OK", msg, flush=True)


# --------------------------------------------------------------------------- unit tests
def test_masks_and_tower(dec_dir, data, E_doc, titles):
    """(4) attention-mask semantics on a tiny random Qwen3, batch invariance, tower training in every mode."""
    tw = llm_user.UserTower(dec_dir, "causal", prefix="Instruct: Given the headlines\nQuery:", max_len=64, device="cpu")
    assert tw.decoder and tw.pooling == "last" and tw.tok.padding_side == "left"
    qs = ["t1w3 t1w4 sw2 ; t2w1 t2w2 t2w9 ; sw3", "t3w1 t3w2", "t0w1 t0w2 t0w3 t0w4 t0w5 t0w6 sw1 sw2 ; t1w1 t1w2 t1w3 t1w4"]
    good, rep = llm_user.probe_masks(tw, qs)
    assert good, rep
    batch = tw._tokenize(qs[:1])
    ids = batch["input_ids"].clone()
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % 100 + 5                               # change the LAST token only

    def hid(mode, lam, x):
        b = {k: v.clone() for k, v in batch.items()}
        b["input_ids"] = x
        with torch.no_grad():
            return tw._hidden(b, mode, lam)
    d = {m: float((hid(m, 1.0, ids)[0, 0] - hid(m, 1.0, ids2)[0, 0]).abs().max()) for m in ("causal", "bidir", "soft")}
    assert d["causal"] < 1e-6 and d["bidir"] > 1e-3 and d["soft"] > 1e-3, d
    d0 = float((hid("soft", 1e-6, ids)[0, 0] - hid("soft", 1e-6, ids2)[0, 0]).abs().max())
    assert d0 < 1e-4, f"soft(lambda->0) should be causal, position-0 moved by {d0}"
    for mode, lam in (("causal", 1.0), ("bidir", 1.0), ("soft", 0.3)):
        single = torch.cat([F.normalize(tw.embed([q], mode, lam), dim=-1) for q in qs])
        both = F.normalize(tw.embed(qs, mode, lam), dim=-1)
        assert float((single - both).abs().max()) < 1e-5, f"{mode}: left-padded batch != single sequence"
    ok(f"(4) masks: position-0 reacts to the last token causal={d['causal']:.1e} / bidir={d['bidir']:.2f} / soft={d['soft']:.2f}; "
       f"soft(lambda->0)={d0:.1e}; batched == single in all modes")

    E = torch.from_numpy(E_doc)
    qfn = lambda h: "; ".join(titles[i] for i in h[-8:])
    for mode in ("causal", "bidir", "soft"):
        t = llm_user.UserTower(dec_dir, mode, prefix="Instruct: Given the headlines\nQuery:", max_len=64, device="cpu")
        info = llm_user.train_user_tower(t, qfn, data.train_core, E, steps=120, bs=16, lr=3e-3, r=4, warmup=5, log=lambda *a: None)
        l = info["losses"]
        assert np.mean(l[-10:]) < np.mean(l[:10]) - 0.3, f"{mode}: tower loss did not fall ({np.mean(l[:10]):.2f} -> {np.mean(l[-10:]):.2f})"
        if mode == "soft":
            assert t.lam == 1.0
    t = llm_user.UserTower(dec_dir, "bidir", prefix="", max_len=64, device="cpu")
    info = llm_user.train_user_tower(t, qfn, data.train_core, E, steps=30, bs=8, r=4, deadline=time.time() - 1, log=lambda *a: None)
    assert info["steps"] == 0
    ok("(4b) tower training: loss falls in causal / bidir / soft; a deadline in the past stops it cleanly")


def test_shared_loss():
    """(4c) shared-negative InfoNCE: B=1 equals plain InfoNCE; a candidate equal to another row's positive is masked."""
    torch.manual_seed(0)
    E = F.normalize(torch.randn(30, 16), dim=-1)
    z = torch.randn(3, 16)
    pos = torch.tensor([4, 7, 4])                                      # rows 0 and 2 share the positive article 4
    negs = torch.tensor([[1, 2], [3, 5], [6, 8]])
    one = llm_user.shared_negative_loss(z[:1], E, pos[:1], negs[:1], 0.05)
    a = F.normalize(z[:1], dim=-1)
    manual = F.cross_entropy((a @ E[torch.tensor([4, 1, 2])].T / 0.05), torch.tensor([0]))
    assert abs(float(one) - float(manual)) < 1e-5
    got = llm_user.shared_negative_loss(z, E, pos, negs, 0.05)
    A = F.normalize(z, dim=-1)
    ids = torch.tensor([4, 7, 4, 1, 2, 3, 5, 6, 8])
    lg = A @ E[ids].T / 0.05
    lg[0, 2] = float("-inf")                                            # row 0 must not treat row 2's identical positive as a negative
    lg[2, 0] = float("-inf")
    assert abs(float(got) - float(F.cross_entropy(lg, torch.arange(3)))) < 1e-5
    ok("(4c) shared-negative InfoNCE: matches plain InfoNCE for one row and masks identical-article false negatives")


def test_v10_equivalence(data, E):
    """(10) the learned user model is a clone of V10's ``LLMEncCA``: identical scores given identical weights (needs the V10 repo)."""
    import importlib.util
    import types
    path = os.environ.get("V10_MODELS", "/home/user/nguyenpnguyen/recsys-project/training/models.py")
    if not os.path.exists(path):
        print(f"SKIP (10) V10 models.py not found at {path}; set V10_MODELS=/path/to/training/models.py to run this check")
        return
    spec = importlib.util.spec_from_file_location("v10_models", path)
    v10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v10)
    ref = v10.LLMEncCA(types.SimpleNamespace(llm_emb=E), types.SimpleNamespace(dim=64, dropout=0.2)).eval()
    mine = UM.LLMEncCA(E, 64, 0.2).eval()
    mine.head.load_state_dict(ref.news.head.state_dict())
    mine.cand_proj.load_state_dict(ref.cand_proj.state_dict())
    rows = data.test[:40]
    with torch.no_grad():
        a = ref.score(data.collate_eval(rows))
        worst = 0.0
        for b in R.EvalSet(rows):
            sc = mine.score(b["hist"], b["hist_mask"], b["cand"])
            for k, r in enumerate(b["rows"]):
                n = int(b["cand_mask"][k].sum())
                worst = max(worst, float((sc[k, :n] - a[r, :n]).abs().max()))
    assert worst < 1e-5, worst
    ok(f"(10) learned user model == V10 LLMEncCA (max abs score diff {worst:.1e} over 40 impressions)")


def test_sid():
    """(5) Semantic-ID components."""
    rng = np.random.default_rng(0)
    X_ = np.concatenate([rng.normal(c, 0.05, (200, 8)) for c in (-1, 0, 1)]).astype(np.float32)
    Xt = torch.from_numpy(X_)
    a = SID._assign(Xt, SID.kmeans(Xt, 3, seed=0)).numpy()
    assert np.mean([np.bincount(a[i * 200:(i + 1) * 200], minlength=3).max() / 200 for i in range(3)]) > 0.99
    K, L, n = 8, 3, 60
    codes = np.zeros((n + 1, L), np.int64)
    codes[1:] = rng.integers(0, K, (n, L))
    torch.manual_seed(0)
    model = SID.SIDGPT(K, L, d=32, layers=2, heads=2, max_items=12).eval()
    B, H, C = 6, 6, 4
    hist = rng.integers(1, n + 1, (B, H))
    hm = np.ones((B, H), bool)
    hm[0] = False                                                          # empty history
    hm[1, 2:] = False                                                      # short history (EvalSet left-aligns the valid items)
    batch = {"hist": torch.from_numpy(hist * hm), "hist_mask": torch.from_numpy(hm), "cand": torch.from_numpy(rng.integers(1, n + 1, (B, C)))}
    got = SID.sid_scorer(model, codes, n_max=5, pmi=False)(batch).numpy()
    off = np.arange(L) * K
    ref = np.zeros_like(got)
    for i in range(B):
        h = [int(x) for x, m in zip(hist[i], hm[i]) if m][-5:]
        for j in range(C):
            toks = [model.BOS] + [int(t) for it in h for t in codes[it] + off] + [int(t) for t in codes[int(batch["cand"][i, j])] + off]
            seq = torch.tensor(toks)[None]
            with torch.no_grad():
                lp = model.code_logprob(model(seq[:, :-1], last_n=L).float(), seq[:, -L:]).sum()
            ref[i, j] = float(lp)
    assert np.abs(got - ref).max() < 1e-4, np.abs(got - ref).max()
    pm = SID.sid_scorer(model, codes, n_max=5, pmi=True)(batch).numpy()
    assert np.abs(pm[0]).max() < 1e-4, "PMI of an empty history must be 0"
    # the model learns a deterministic "next item" regularity
    succ = rng.permutation(n) + 1
    codes2 = codes.copy()
    rows = []
    for _ in range(1500):
        start = int(rng.integers(1, n + 1))
        seq = [start]
        for _ in range(3):
            seq.append(int(succ[seq[-1] - 1]))
        rows.append((0, seq[:-1], seq[-1], []))
    m2, info = SID.train_sid_gpt(codes2, K, rows, n_max=5, epochs=10, bs=64, lr=3e-3, seed=0, d=32, layers=2, heads=2, log=lambda *a: None)
    starts = rng.integers(1, n + 1, 200)
    h2 = np.stack([[s, int(succ[s - 1]), int(succ[int(succ[s - 1]) - 1])] for s in starts])
    cand = np.stack([[int(succ[h[-1] - 1])] + list(rng.choice([i for i in range(1, n + 1) if i != int(succ[h[-1] - 1])], 5, replace=False)) for h in h2])
    sc = SID.sid_scorer(m2, codes2, n_max=5)({"hist": torch.from_numpy(h2), "hist_mask": torch.ones(200, 3, dtype=torch.bool), "cand": torch.from_numpy(cand)}).numpy()
    auc = float(np.mean((sc[:, :1] > sc[:, 1:]).astype(float) + 0.5 * (sc[:, :1] == sc[:, 1:])))
    assert auc > 0.9, f"SID-GPT did not learn a deterministic regularity (AUC {auc:.2f})"
    ok(f"(5) SID: k-means purity, batched == per-sequence reference (max diff {np.abs(got - ref).max():.1e}, empty history incl.), "
       f"PMI(empty)=0, learns the regularity (AUC {auc:.2f})")


def test_generator(dec_dir, texts, tmp):
    """(6) generation / judging are batch-invariant, cached, and honour deadlines."""
    cp = os.path.join(tmp, "gen_test_cache.jsonl")
    g = llm_gen.Generator(dec_dir, "cpu", dtype=torch.float32, cache_path=cp)
    prompts = [f"Write about {texts[i][:60]}" for i in (0, 5, 17, 30, 44)]
    batched = g.generate(prompts, max_new_tokens=10, bs=3)
    fresh = llm_gen.Generator(dec_dir, "cpu", dtype=torch.float32)
    assert batched == [fresh.generate([p], max_new_tokens=10, bs=1)[0] for p in prompts], "greedy generation depends on the batch"
    again = llm_gen.Generator(dec_dir, "cpu", dtype=torch.float32, cache_path=cp)
    assert again.generate(prompts, max_new_tokens=10, bs=3) == batched and again.rate.get("gen") is None, "cache not used"
    yb = fresh.yes_no(prompts, bs=2)
    ys = np.array([llm_gen.Generator(dec_dir, "cpu", dtype=torch.float32).yes_no([p], bs=1)[0] for p in prompts])
    assert np.abs(yb - ys).max() < 1e-4
    late = fresh.generate(["a b c"] * 4, max_new_tokens=2, bs=2, deadline=time.time() - 1)
    assert all(x is None for x in late)
    assert llm_gen.plan_prefix([[1, 2], [2, 3], [4, 5, 6], [7]], 5) == (2, 3) and llm_gen.plan_prefix([[1, 2]], 1) == (0, 0)
    ok("(6) generator: batch-invariant greedy decoding and Yes/No log-odds, JSONL cache, deadline, prefix planning")


def test_cache_fuse_subset(S):
    """(6b) ScoreCache/fuse reproduce direct scoring; subset comparisons equal manual slicing."""
    E = S.tensor(S.E0)
    mp = R.mean_pool_scorer(E)
    for sp in ("val", "test"):
        direct = S.run(mp, sp)
        cached = S.run(common.ScoreCache(S.es[sp], mp).scorer(S.device), sp)
        fused = S.run(common.fuse([(mp, 1.0)]), sp)
        for k in R.METRICS:
            assert np.allclose(direct["arr"][k], cached["arr"][k], atol=1e-6, equal_nan=True), k
        assert np.allclose(direct["arr"]["ndcg@10"], fused["arr"]["ndcg@10"], atol=1e-6), "z-score fusion of one scorer changed the ranking"
    ids, es = S.subset("test", 120)
    sub = S.run(mp, "test", es={"test": es})
    full = S.run(mp, "test")
    assert np.allclose(sub["arr"]["ndcg@10"], full["arr"]["ndcg@10"][ids], atol=1e-6)
    d = R.compare(sub, full, "ndcg@10")
    manual = R.ratio_ci(sub["arr"]["ndcg@10"] - full["arr"]["ndcg@10"][ids])
    assert np.allclose(d, manual, equal_nan=True) and abs(d[0]) < 1e-6
    cs = R.compare_slice(sub, full, "cold")
    assert np.isnan(cs[0]) or abs(cs[0]) < 1e-6
    assert S.subset("test", 60)[0].tolist() == sorted(set(S.subset("test", 60)[0]) & set(ids)), "subsets must be nested"
    ok("(6b) ScoreCache == direct scoring, z-fusion of one scorer is rank-preserving, subset CIs equal manual slicing, subsets are nested")


# --------------------------------------------------------------------------------- main
RUN_A = ["--max-len", "32", "--q-max-len", "64", "--seeds", "2", "--pairs", "3000", "--pairs-enc", "600", "--epochs-head", "2",
         "--epochs-enc", "1", "--enc-steps", "8", "--bs-enc", "8", "--lora-r", "4", "--enc-bs", "64", "--device", "cpu", "--hist-k", "8",
         "--betas", "0.5", "1.0", "--knn-k", "5", "10", "--lams", "0.5", "1.0", "--gammas", "0.5", "--sid-k", "8", "--sid-d", "64",
         "--sid-layers", "2", "--sid-heads", "2", "--sid-epochs", "12", "--sid-lr", "3e-3", "--sid-val-n", "100", "--sid-bs", "64",
         "--um-epochs", "4", "--um-seeds", "1", "--lsa-dim", "32", "--no-resume"]


def main():
    tmp = tempfile.mkdtemp(prefix="v11_selftest_")
    print("fixture:", tmp)
    common.CLOCK.reset()
    texts = build_fixture(tmp)
    enc_dir, dec_dir = os.path.join(tmp, "tinyenc"), os.path.join(tmp, "tiny-qwen3-embedding")
    build_tiny_encoder(enc_dir, texts)
    build_tiny_decoder(dec_dir, texts)
    mind = ["--mind-train", f"{tmp}/train", "--mind-dev", f"{tmp}/dev"]

    test_shared_loss()
    test_sid()
    test_generator(dec_dir, texts, tmp)

    # the guard that compares HFEncoder with a reference implementation must be exercised too
    ref_enc = adapt.HFEncoder(enc_dir, max_len=32, device="cpu")
    suite.EQUIV_REFERENCE = lambda t: ref_enc.encode(t)

    # ---------------- run A (encoder-type model): first half, then the rest in a second call that must RESUME
    out = os.path.join(tmp, "outA")
    part1 = "controls,pooling,graph,head,histquery,sid"
    suite.main([*mind, "--work", out, "--encoder", enc_dir, "--stages", part1, "--sweep-encoders", dec_dir, *RUN_A[:-1]])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        S = suite.main([*mind, "--work", out, "--encoder", enc_dir, "--stages", f"{part1},encoder,usermodel,sweep", "--sweep-encoders", dec_dir,
                        *RUN_A[:-1]])                                                       # no --no-resume: must pick up state.pkl
    text = buf.getvalue()
    print(text[-3500:])
    assert "resuming: stages already finished" in text and "stage 'pooling' already finished" in text, "second call did not resume"
    assert text.count("== H8 pooling") == 0 and text.count("== H7 Semantic IDs") == 0, "finished stages were re-run"
    ok("(7) resume: second call skipped the finished stages and reused kv/records (best_head view feeds the user model)")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):                                                   # a different configuration must NOT resume
        suite.main([*mind, "--work", out + "_cfg", "--encoder", enc_dir, "--stages", "controls", "--sweep-encoders", dec_dir, *RUN_A[:-1]])
        shutil.copy(os.path.join(out + "_cfg", "state.pkl"), os.path.join(out + "_cfg2.pkl"))
        suite.main([*mind, "--work", out + "_cfg", "--encoder", enc_dir, "--stages", "controls,pooling", "--sweep-encoders", dec_dir,
                    *[("1234" if a == "3000" else a) for a in RUN_A[:-1]]])
    assert "belongs to a different configuration" in buf.getvalue(), "state from another configuration was reused"
    ok("(7a) resume refuses a state saved with different result-relevant arguments (QUICK can never leak into a full run)")

    names = {r["name"] for r in S.records}
    expected = {"frozen", "random_vec", "frozen_hf", "pool_recency", "pool_maxsim", "pool_topk", "pool_lse", "entity_rag", "knn_rag",
                "community_graphrag", "user_rag", "head_lin", "head_mlp+hn+mrl", "lora+mrl+hn", "dora+mrl+hn", "histq_instr",
                "histq_plain+meanpool", "sid_profile", "sid_profile+meanpool", "sid_gpt", "sid_gpt_pmi", "sid_gpt+meanpool",
                "um_frozen", "um_head", "um_encoder", "um_entity_rag", "um_knn_rag", "enc_tfidf_lsa", "enc_tinyqwen3embedding"}
    assert not (expected - names), f"missing rows: {sorted(expected - names)}"
    assert len([r for r in S.records if r["name"] == "frozen"]) == 1 and len(names) == len(S.records), "duplicate rows after resume"
    for r in S.records:
        for k in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            assert np.isfinite(r[k]), (r["name"], k, r[k])

    # (1) evaluator == V10 metrics.py on the suite's own frozen scores
    E = S.E0
    imps = []
    for row in S.data.test:
        h = list(row[2])[-50:]
        u = E[h].mean(0) if h else np.zeros(E.shape[1], np.float32)
        imps.append((np.array(row[4]), (E[row[3]] @ u).astype(np.float32)))
    ref = metrics.aggregate(imps)
    got = S.cache["frozen"]["test"]["mean"]
    for k in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        assert abs(ref[k] - got[k]) < 1e-5, f"evaluator != metrics.py for {k}: {got[k]} vs {ref[k]}"
    ok("(1) evaluator reproduces metrics.py on the suite's frozen scores")

    # (2) identity at step 0 for every residual head
    for kind in ("lin", "mlp"):
        head = adapt.HEADS[kind](E.shape[1])
        x = torch.from_numpy(E[1:50])
        assert torch.allclose(head(x), F.normalize(x, dim=-1), atol=1e-6)
    ok("(2) residual heads are the identity before training")

    # (3) never worse than frozen on validation by construction
    for r in S.records:
        if r["family"] == "H2-adapt" and r["name"].startswith("head_"):
            assert r["val_ndcg@10"] >= S.cache["frozen"]["val"]["mean"]["ndcg@10"] - 1e-9, r["name"]
    ok("(3) head adapters never fall below the frozen validation score")

    # (5)(6) hard negatives are not a silent no-op, Matryoshka is active when d is large
    pairs = adapt.build_pairs(S.data, max_pairs=20000)
    ea, _ = adapt.train_head(E, pairs, "lin", epochs=2, bs=256, seed=0, log=lambda *a: None)
    eb, _ = adapt.train_head(E, pairs, "lin", hardneg=True, epochs=2, bs=256, seed=0, log=lambda *a: None)
    assert np.abs(ea - eb).max() > 1e-6, "hardneg=True produced identical embeddings (flag ignored?)"
    a = F.normalize(torch.randn(32, 384), dim=-1)
    p_ = F.normalize(a + 0.5 * torch.randn(32, 384), dim=-1)
    ix = torch.arange(32)
    assert abs(adapt.contrastive_loss(a, p_, ix).item() - adapt.contrastive_loss(a, p_, ix, dims=adapt.mrl_dims(384)).item()) > 1e-3
    ok("(5)(6) hard negatives change training; Matryoshka loss differs from full-dim loss at d=384")

    # (7) the encoder-in-the-loop trainer really learns
    tx = [""] + S.meta["text"]
    val_ids = np.unique(np.concatenate([np.asarray(r[3], np.int64) for r in S.data.validation] + [np.asarray(r[2], np.int64) for r in S.data.validation]))
    es_val = R.EvalSet(S.data.validation)
    val_fn = lambda M: R.evaluate(R.mean_pool_scorer(torch.from_numpy(M)), es_val)["mean"]["ndcg@10"]
    logs = []
    enc = adapt.HFEncoder(enc_dir, max_len=32, device="cpu")
    Ef, info = adapt.train_encoder(enc, tx, pairs, E, mode="lora", hardneg=True, epochs=2, bs=32, lr=3e-3, r=8,
                                   val_ids=val_ids, val_fn=val_fn, max_steps=240, log=logs.append)
    losses = [float(l.split("loss=")[1].split()[0]) for l in logs if "step" in l]
    assert losses[-1] < losses[0] and np.abs(Ef - E).mean() > 1e-3 and val_fn(Ef) >= val_fn(E) - 1e-9
    ok(f"(7b) LoRA trainer learns: loss {losses[0]:.3f} -> {losses[-1]:.3f}, val nDCG@10 {val_fn(E):.4f} -> {val_fn(Ef):.4f}")

    test_cache_fuse_subset(S)
    test_v10_equivalence(S.data, S.E0)

    # (8) direction-vs-direction comparison and per-impression arrays
    n_h = min(5, len([r for r in S.reps if r["family"] not in suite.TECHNIQUE_FAMILIES_EXCLUDED]))
    assert len(S.h2h) == n_h * (n_h - 1) // 2 and n_h >= 4, (len(S.reps), len(S.h2h))
    assert all(np.isfinite(h["d_ndcg@10"][0]) for h in S.h2h)
    z = np.load(os.path.join(out, "test_arrays.npz"))
    assert z["frozen__auc"].shape[0] == len(S.data.test) and "histq_instr__ndcg@10" in z.files and "sid_gpt__ids" in z.files
    ok(f"(8) head-to-head: {len(S.h2h)} pairs among the top-{n_h}; test_arrays.npz holds {len(z.files)} arrays")
    for f in ("results.md", "results.json", "state.pkl"):
        assert os.path.getsize(os.path.join(out, f)) > 0
    assert any(n.startswith("emb_") for n in os.listdir(out)), "no embedding export"
    cold = S.cache["frozen"]["test"].get("cold_frac")
    assert cold is not None and 0.0 < cold < 1.0, f"fixture should contain cold clicks, got {cold}"
    ok(f"(4) outputs written; cold-click share in the fixture = {cold:.1%}")

    # ---------------- run B (decoder-only LLM encoder + generative LLM): H5 / H6 / H9 / H4b / learned user model
    outB = os.path.join(tmp, "outB")
    ref_dec = adapt.HFEncoder(dec_dir, max_len=48, device="cpu")
    suite.EQUIV_REFERENCE = lambda t: ref_dec.encode(t)
    SB = suite.main([*mind, "--work", outB, "--encoder", dec_dir, "--cache-dir", os.path.join(out, "cache"), "--max-len", "48", "--q-max-len", "96",
                     "--seeds", "1", "--pairs", "3000", "--epochs-head", "2", "--enc-bs", "64", "--device", "cpu", "--hist-k", "8",
                     "--hq-plain", "0", "--hq-subset", "1", "--betas", "0.5", "--knn-k", "5", "--lams", "0.5", "--gammas", "0.5",
                     "--llm-user-steps", "40", "--llm-user-bs", "8", "--llm-user-r", "4", "--llm-user-lr", "3e-3", "--llm-val-n", "100",
                     "--llm-test-n", "150", "--um-views", "frozen,head,entity_rag", "--um-epochs", "3", "--um-seeds", "1",
                     "--gen-model", dec_dir, "--gen-bs", "8", "--gen-new-tokens", "8", "--gen-val-n", "40", "--gen-test-n", "80",
                     "--augment-minutes", "5", "--rerank-k", "5", "--rerank-val-n", "40", "--rerank-test-n", "80", "--rerank-minutes", "5",
                     "--gr-max-comm", "10", "--gr-titles", "4", "--no-resume",
                     "--stages", "controls,histquery,llm_user,rerank,augment,graphrag_llm,pooling,graph,head,usermodel"])
    nb = {r["name"] for r in SB.records}
    exp_b = {"frozen", "histq_instr", "histq_instr+meanpool", "ut_causal", "ut_bidir", "ut_soft", "ut_causal+meanpool", "ut_soft+meanpool",
             "rerank_llm", "rerank_fused", "kar_item", "kar_user", "kar_user+meanpool", "kar_item+user", "community_llm", "community_llm_ctx",
             "um_frozen", "um_head", "um_entity_rag", "entity_rag", "head_lin"}
    assert not (exp_b - nb), f"run B missing rows: {sorted(exp_b - nb)}"
    sub_rows = [r for r in SB.records if r["name"].startswith(("ut_", "histq", "kar_", "rerank_"))]
    assert all(r["n_test"] < len(SB.data.test) for r in sub_rows) and all("ref_ndcg@10_same" in r for r in sub_rows)
    ok(f"(9) run B produced the LLM rows ({len(sub_rows)} on subsets, each with the reference evaluated on the same impressions)")
    # token-level sanity of what the generation stages fed to the encoders
    gen_cache = os.path.join(outB, "gen_cache.jsonl")
    assert os.path.getsize(gen_cache) > 0
    rows = suite.leaderboard([S, SB], ["A", "B"], ("A", "frozen"), os.path.join(tmp, "leaderboard.md"))
    assert len(rows) > 20 and os.path.getsize(os.path.join(tmp, "leaderboard.md")) > 0
    test_masks_and_tower(dec_dir, SB.data, SB.E0, SB.titles)

    # ---------------- wall-clock budget: a (practically) zero budget skips every optional stage but still writes outputs
    common.CLOCK.reset()
    suite.EQUIV_REFERENCE = lambda t: ref_enc.encode(t)
    outC = os.path.join(tmp, "outC")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SC = suite.main([*mind, "--work", outC, "--encoder", enc_dir, "--stages", "controls,pooling,graph,sid", "--budget-hours", "1e-9", *RUN_A])
    text = buf.getvalue()
    assert text.count("SKIPPED: wall-clock budget exhausted") == 3 and {r["name"] for r in SC.records} >= {"frozen", "random_vec"}
    assert os.path.getsize(os.path.join(outC, "results.md")) > 0
    common.CLOCK.reset()
    ok("(7b) wall-clock budget: optional stages skipped, controls and outputs still written")
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
