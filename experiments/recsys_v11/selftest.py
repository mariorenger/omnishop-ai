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
import copy
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
import fields as FD
import llm_gen
import llm_user
import metrics
import refdata as RD
import refmodels as RM
import pooling as P
import rep_eval as R
import sid as SID
import stages_ext as X
import stages_ref as SR
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


def test_torchao_shim(tmp):
    """(11) Kaggle ships torchao 0.10; recent peft then raises ImportError when a LoRA adapter is injected.  Reproduce it with a fake old
    torchao in a subprocess: plain peft must fail, our adapters (encoder LoRA/DoRA and the LLM tower) must work."""
    import subprocess
    fake = os.path.join(tmp, "fake_torchao")
    os.makedirs(os.path.join(fake, "torchao"), exist_ok=True)
    os.makedirs(os.path.join(fake, "torchao-0.10.0.dist-info"), exist_ok=True)
    open(os.path.join(fake, "torchao", "__init__.py"), "w").close()
    with open(os.path.join(fake, "torchao-0.10.0.dist-info", "METADATA"), "w") as f:
        f.write("Metadata-Version: 2.1\nName: torchao\nVersion: 0.10.0\n")
    code = f"""
import os, sys
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
import peft.import_utils as iu
try:
    iu.is_torchao_available(); raised = False
except ImportError:
    raised = True
import selftest, adapt, llm_user
texts = selftest.build_fixture({os.path.join(tmp, "ta")!r})
selftest.build_tiny_encoder({os.path.join(tmp, "ta_enc")!r}, texts)
selftest.build_tiny_decoder({os.path.join(tmp, "ta_dec-qwen3-embedding")!r}, texts)
a = adapt.HFEncoder({os.path.join(tmp, "ta_enc")!r}, max_len=32, device="cpu").attach_peft("lora", 4)
b = adapt.HFEncoder({os.path.join(tmp, "ta_enc")!r}, max_len=32, device="cpu").attach_peft("dora", 4)
c = llm_user.UserTower({os.path.join(tmp, "ta_dec-qwen3-embedding")!r}, "bidir", device="cpu").attach_peft("lora", 4)
print("RESULT", raised, a > 0, b > 0, c > 0)
"""
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=fake), capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert line and line[0] == "RESULT True True True True", (r.stdout[-800:], r.stderr[-1500:])
    ok("(11) torchao 0.10 + recent peft: plain peft raises ImportError, our LoRA/DoRA encoder and LLM tower still attach")


def test_popularity(S):
    """(13) `frozen+pop` is V10's bge_zs_pop (z-score sum of BGE mean-pool and online popularity); every zero-shot variant has a `+pop` twin."""
    E = S.E0
    imps = []
    for row in S.data.test:
        h = list(row[2])[-50:]
        u = E[h].mean(0) if h else np.zeros(E.shape[1], np.float32)
        s1, s2 = (E[row[3]] @ u).astype(np.float64), np.asarray(row[5], np.float64)
        z = lambda x: (x - x.mean()) / (np.sqrt(((x - x.mean()) ** 2).mean()) + 1e-6)
        imps.append((np.array(row[4]), (z(s1) + z(s2)).astype(np.float32)))
    ref = metrics.aggregate(imps)
    got = S.cache["frozen+pop"]["test"]["mean"]
    for k in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        assert abs(ref[k] - got[k]) < 1e-4, f"frozen+pop != V10-style bge_zs_pop for {k}: {got[k]} vs {ref[k]}"
    imps = [(np.array(r[4]), np.asarray(r[5], np.float32)) for r in S.data.test]
    ref = metrics.aggregate(imps)
    got = S.cache["popularity"]["test"]["mean"]
    assert all(abs(ref[k] - got[k]) < 1e-4 for k in ("auc", "mrr", "ndcg@5", "ndcg@10"))
    twins = {r["name"] for r in S.records if r["family"].endswith("+pop")}
    need = {"pool_recency+pop", "pool_maxsim+pop", "pool_topk+pop", "pool_lse+pop", "entity_rag+pop", "knn_rag+pop", "head_lin+pop",
            "histq_instr+pop", "sid_profile+pop", "sid_profile+meanpool+pop", "sid_gpt+pop", "lora+mrl+hn+pop"}
    assert not (need - twins), f"missing popularity twins: {sorted(need - twins)}"
    for n in twins:                                              # a twin is compared with the popularity twin of whatever its base row is compared with
        base_ref = S.row(S.row(n)["cfg"]["base"])["ref"]
        assert S.row(n)["ref"] == (f"{base_ref}+pop" if f"{base_ref}+pop" in S.cache else "frozen+pop"), (n, S.row(n)["ref"], base_ref)
    assert S.row("txt_t_lse+pop")["ref"] == "pool_lse+pop" and S.row("txt_t+pop")["ref"] == "frozen+pop"
    assert len(S.reps_pop) >= 3 and all(r["family"].endswith("+pop") for r in S.reps_pop)
    ok(f"(13) popularity: frozen+pop == V10-style bge_zs_pop and popularity == raw pop (metrics.py); {len(twins)} '+pop' twins, "
       f"{len(S.reps_pop)} families in the with-popularity summary")


def test_identical_verdict(S):
    """(12) a variant that reproduces its reference exactly must read 'identical', not 'WORSE' because of float rounding."""
    mp = R.mean_pool_scorer(S.tensor(S.E0))
    a, b = suite._slim(S.run(mp, "test")), suite._slim(S.run(mp, "test"))
    d = R.compare(a, b, "ndcg@10")
    assert d == (0.0, 0.0, 0.0) and R.verdict(d[1], d[2]) == "identical"
    assert R.verdict(1e-3, 2e-3) == "BETTER" and R.verdict(-2e-3, -1e-3) == "WORSE" and R.verdict(-1e-3, 1e-3) == "no sig. difference"
    ok("(12) identical results give an exact zero difference and the verdict 'identical'")


def test_probe_criteria():
    """(14) the first Kaggle run measured soft(1e-6) vs causal = 0.099 on the real Qwen3 (huge logits on special tokens) while everything else was
    exact; that report must pass, a model that ignores the 4D mask must not, and soft(0) must be exactly causal."""
    real = {"causal_4d_vs_native": 0.0, "bidir_vs_causal": 0.44879, "soft1_vs_bidir": 0.0, "soft0_vs_causal": 0.09895}
    assert llm_user.probe_ok(real)
    assert not llm_user.probe_ok({**real, "bidir_vs_causal": 0.0})               # mask silently ignored
    assert not llm_user.probe_ok({**real, "causal_4d_vs_native": 0.2})           # hand-built causal mask disagrees with the native path
    attn = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    assert bool((llm_user.build_mask(attn, "soft", 0.0, torch.float32) == llm_user.build_mask(attn, "causal", 1.0, torch.float32)).all())
    assert bool((llm_user.build_mask(attn, "soft", 1.0, torch.float32) == llm_user.build_mask(attn, "bidir", 1.0, torch.float32)).all())
    ok("(14) mask probe: the real-Qwen3 report passes, ignored/wrong masks fail; soft(0) == causal and soft(1) == bidir exactly")


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



# --------------------------------------------------------------------------- reference baselines
def build_fake_glove(path, words, dim, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + " " + " ".join(f"{x:.5f}" for x in rng.normal(0, 0.4, dim)) + "\n")
        f.write("new york " + " ".join(["0.1"] * (dim + 1)) + "\n")            # multi-token entry of the 840B file: must be skipped


def test_ref_data(tmp, mind_train, data, meta):
    """(16) token tables, GloVe loading (txt / zip / sentence-transformers folder / failure chain) and the V10-identical training samples."""
    assert RD.tokenize("Trump's U.S. plan, 2020!") == ["trump", "s", "u", ".", "s", ".", "plan", ",", "2020", "!"]   # the apostrophe is dropped, .,!?;| are tokens
    v = RD.build_vocab([["a", "b", "b"], ["b", "c"]], min_freq=2, keep={"c"})
    assert v == {"<pad>": 0, "<unk>": 1, "b": 2, "c": 3}, v
    enc = RD.encode_tokens([["b", "zzz", "c"], []], v, 4)
    assert enc.shape == (3, 4) and enc[1].tolist() == [2, 1, 3, 0] and not enc[0].any() and not enc[2].any()
    # GloVe: txt, zip, sentence-transformers folder; the corpus words only
    words = ["t1w1", "t2w2", "sw3", "unseen"]
    gp = os.path.join(tmp, "glove.6B.8d.txt")
    build_fake_glove(gp, words, 8)
    want = {"t1w1", "sw3", "t3w3"}
    w, M = RD._glove_local(gp, 8, want)
    assert sorted(w) == ["sw3", "t1w1"] and M.shape == (2, 8)
    import zipfile
    zp = os.path.join(tmp, "glove.6B.zip")
    with zipfile.ZipFile(zp, "w") as z:
        z.write(gp, "glove.6B.8d.txt")
    w2, M2 = RD._glove_local(zp, 8, want)
    assert sorted(w2) == sorted(w) and np.allclose(M2[np.argsort(w2)], M[np.argsort(w)])
    st = os.path.join(tmp, "st_glove", "0_WordEmbeddings")
    os.makedirs(st, exist_ok=True)
    mat = torch.randn(len(words), 8)
    with open(os.path.join(st, "whitespacetokenizer_config.json"), "w") as f:
        json.dump({"vocab": words, "stop_words": [], "do_lower_case": False}, f)
    torch.save({"emb_layer.weight": mat}, os.path.join(st, "pytorch_model.bin"))
    w3, M3 = RD._from_st_dir(os.path.dirname(st), 8, want)
    assert sorted(w3) == ["sw3", "t1w1"] and np.allclose(M3[w3.index("t1w1")], mat[0].numpy())
    off = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
    saved = RD._glove_hf, RD._glove_gensim
    RD._glove_hf, RD._glove_gensim = off, off
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            none = RD.load_glove(want, 8, source=os.path.join(tmp, "missing.txt"))
            ok_ = RD.load_glove(want, 8, source=gp)
    finally:
        RD._glove_hf, RD._glove_gensim = saved
    assert none == (None, None, "") and ok_[2].startswith("file:") and "no GloVe source worked" in buf.getvalue()
    # RefText: the embedding matrix carries the GloVe rows, zero PAD, matched random scale for the rest
    allw = sorted({w_ for t in meta["titles"] + meta["abstracts"] for w_ in RD.tokenize(t)})
    gp2 = os.path.join(tmp, "glove_all.txt")
    build_fake_glove(gp2, allw[:60], 8, seed=1)
    text = RD.RefText(meta, title_len=12, body_len=20, glove_dim=8, glove_source=gp2, log=lambda *a: None)
    assert text.title.shape == (data.n_news, 12) and text.body.shape == (data.n_news, 20) and text.cat.shape == (data.n_news,)
    assert not text.title[0].any() and text.cat[0] == 0 and text.n_cat == 7
    E = text.embedding(8, seed=0)
    j = text.vocab[text.glove_words[0]]
    assert np.allclose(E[j], text.glove_mat[0]) and not E[0].any() and E.shape == (text.vocab_size, 8)
    assert text.stats["glove_vocab_hits"] >= 50 and text.stats["unk_title_share"] < 0.2
    assert not np.allclose(text.embedding(8, 0), text.embedding(8, 1))                  # the seed changes the random rows
    Er = text.embedding(8, 0, init="random")
    assert abs(Er[text.vocab[allw[-1]]].std() - 0.1) < 0.2 and not np.allclose(Er[j], text.glove_mat[0])
    # V10-identical training samples with the full negative pool
    core = RD.read_train_core(mind_train, meta["ids"], data)
    ts = RD.TrainSet(core)
    RD.check_train_alignment(ts, data)
    assert ts.n_samples == len(data.train_core) and ts.n_imp <= data.train_core_impressions
    bad = list(reversed(meta["ids"]))                                                     # a different article order must be caught
    try:
        RD.check_train_alignment(RD.TrainSet(RD.read_train_core(mind_train, bad, data)), data)
        raise AssertionError("a permuted article-id mapping went undetected")
    except ValueError:
        pass
    rng = np.random.default_rng(0)
    d1, d2 = ts.draw(rng), ts.draw(rng)
    assert d1.shape == (ts.n_samples, 4) and not np.array_equal(d1, d2), "negatives are not re-drawn"
    small = 0
    for s_ in range(ts.n_samples):
        i = ts.samp_imp[s_]
        pool = set(ts.neg_flat[ts.neg_off[i]:ts.neg_off[i] + ts.neg_cnt[i]].tolist())
        assert set(d1[s_].tolist()) <= pool
        if ts.neg_cnt[i] >= 4:
            assert len(set(d1[s_].tolist())) == 4, "sampling without replacement expected when the pool holds >= 4"
        else:
            small += 1
    for ep in range(1, 4):                                                                # batches: shapes and the history mask
        H, M, C = next(iter(ts.batches(256, ep, 0, "cpu")))
        assert H.shape[1] == 50 and C.shape[1] == 5 and bool(((H > 0) == M).all())
    union = [set() for _ in range(ts.n_samples)]                                          # over many epochs every negative of a pool gets drawn
    for ep in range(30):
        d = ts.draw(np.random.default_rng(100 + ep))
        for s_ in range(0, ts.n_samples, 7):
            union[s_].update(d[s_].tolist())
    for s_ in range(0, ts.n_samples, 7):
        i = ts.samp_imp[s_]
        pool = set(ts.neg_flat[ts.neg_off[i]:ts.neg_off[i] + ts.neg_cnt[i]].tolist())
        assert union[s_] == pool or len(pool) > 30, (s_, len(union[s_]), len(pool))
    st_ = RD.StaticTrainSet(data.train_core)
    b0 = next(iter(st_.batches(8, 1, 0, "cpu")))
    assert b0[2].shape == (8, 5) and st_.n_samples == len(data.train_core)
    sub = RD.subsample(core, 0.5)
    assert abs(len(sub) - len(core) // 2) <= 1 and RD.subsample(core, 1.0) is core
    ok(f"(16) reference data: regex tokens, vocab/UNK, GloVe (txt, zip, sentence-transformers folder, failure chain), embedding init; "
       f"{ts.n_samples:,} samples == V10 train_core with the full negative pool (re-drawn each epoch, {small} small pools drawn with replacement); "
       f"a permuted article mapping is rejected")
    return text, ts


def naive_fast_attention(m, x, mask):
    """Loop version of the official Fastformer additive attention (one head at a time, softmax over real tokens only)."""
    B, L, d = x.shape
    out = torch.zeros_like(x)
    for b in range(B):
        idx = mask[b].nonzero().flatten()
        q, k = m.q(x[b]), m.k(x[b])
        p = torch.zeros(L, d)
        for h in range(m.h):
            sl = slice(h * m.dh, (h + 1) * m.dh)
            a = torch.softmax(m.q_att(q)[idx, h] / m.dh ** 0.5, 0)
            gq = (a[:, None] * q[idx][:, sl]).sum(0)
            p[:, sl] = k[:, sl] * gq
        u = torch.zeros(L, d)
        for h in range(m.h):
            sl = slice(h * m.dh, (h + 1) * m.dh)
            bta = torch.softmax(m.k_att(p)[idx, h] / m.dh ** 0.5, 0)
            gk = (bta[:, None] * p[idx][:, sl]).sum(0)
            u[:, sl] = gk * q[:, sl]
        out[b] = m.transform(u) + q
    return out


def test_ref_models(data, meta, text, ts):
    """(17) architectures: Fastformer == loop reference, padding invariance, scorer == training forward, V10-NRMS equality, all three learn."""
    import types
    torch.manual_seed(0)
    # Fastformer additive attention against the loop version
    fa = RM.FastSelfAttn(32, 4).eval()
    x = torch.randn(3, 9, 32)
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[1, 5:] = False
    mask[2, 2:] = False
    with torch.no_grad():
        got, want = fa(x, mask), naive_fast_attention(fa, x, mask)
    assert torch.allclose(got[mask], want[mask], atol=1e-5), float((got[mask] - want[mask]).abs().max())
    # additive attention ignores masked positions
    ad = RM.AddAttn(16, 8).eval()
    xx = torch.randn(2, 6, 16)
    mm = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    x2 = xx.clone()
    x2[0, 3:] = 99.0
    assert torch.allclose(ad(xx, mm)[0], ad(x2, mm)[0], atol=1e-5)
    # padding invariance of the masked news encoders (a longer PAD tail must not move the vector); the V10 layout is NOT invariant
    from dataclasses import replace
    base = {"nrms": replace(RM.preset("nrms"), emb_dim=32, heads=4, head_dim=8, init="random"),
            "fastformer": replace(RM.preset("fastformer"), emb_dim=32, heads=4, head_dim=8, ff_layers=1, init="random"),
            "naml": replace(RM.preset("naml"), emb_dim=32, naml_filters=32, init="random")}
    for kind, cfg in base.items():
        net = RM.build_net(cfg, "cpu", 0, text, data).eval()
        wide = RD.RefText.__new__(RD.RefText)
        wide.__dict__.update(text.__dict__)
        wide.title = np.concatenate([text.title, np.zeros((text.title.shape[0], 9), np.int64)], 1)
        net2 = RM.build_net(cfg, "cpu", 0, wide, data).eval()
        net2.load_state_dict({k: v for k, v in net.state_dict().items() if k != "news_enc.pos.weight"}, strict=False)
        if kind == "fastformer":                                                 # positions beyond the short length only exist in the wide model
            net2.news_enc.pos.weight.data[:text.title.shape[1]] = net.news_enc.pos.weight.data
        ids = torch.arange(1, 60)
        with torch.no_grad():
            a, b = net.encode_news(ids), net2.encode_news(ids)
        assert torch.allclose(a, b, atol=1e-4), (kind, float((a - b).abs().max()))
        # batch invariance of the article encoder
        with torch.no_grad():
            one = torch.cat([net.encode_news(ids[i:i + 1]) for i in range(0, 12)])
        assert torch.allclose(one, a[:12], atol=1e-4)
    cfg = replace(base["nrms"], title_mask=False)
    net = RM.build_net(cfg, "cpu", 0, text, data).eval()
    wide = RD.RefText.__new__(RD.RefText)
    wide.__dict__.update(text.__dict__)
    wide.title = np.concatenate([text.title, np.zeros((text.title.shape[0], 9), np.int64)], 1)
    net2 = RM.build_net(cfg, "cpu", 0, wide, data).eval()
    net2.load_state_dict(net.state_dict(), strict=False)
    with torch.no_grad():
        assert not torch.allclose(net.encode_news(torch.arange(1, 30)), net2.encode_news(torch.arange(1, 30)), atol=1e-3), \
            "without a title mask the PAD tail should change the vector (this is the V10 behaviour the ladder removes)"
    # scorer (catalogue encoded once) == the training forward (articles encoded per batch), empty histories tie
    es = R.EvalSet(data.test[:60])
    for kind, cfg in base.items():
        net = RM.build_net(cfg, "cpu", 0, text, data).eval()
        sc = RM.make_scorer(net)
        for b in es:
            with torch.no_grad():
                a, c = sc(b), net(b["hist"], b["hist_mask"], b["cand"])
            m = b["cand_mask"]
            assert torch.allclose(a[m], c[m], atol=1e-4), (kind, float((a[m] - c[m]).abs().max()))
            empty = ~b["hist_mask"].any(1)
            if empty.any():
                assert float(a[empty].abs().max()) == 0.0, "a reader without history must score every candidate equally"
    # V10's NRMS: identical scores once the weights are copied (the ladder's first rung is a re-implementation of it)
    import importlib.util
    path = os.environ.get("V10_MODELS", "/home/user/nguyenpnguyen/recsys-project/training/models.py")
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("v10_models_ref", path)
        v10 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v10)
        ref = v10.NRMS(types.SimpleNamespace(vocab_size=data.vocab_size), types.SimpleNamespace(dim=64, heads=2, dropout=0.2)).eval()
        mine = RM.build_net(RM.preset("v10"), "cpu", 0, None, data).eval()
        with torch.no_grad():
            mine.emb.weight.copy_(ref.news.word_emb.weight)

            def split_mha(mha, att):
                W, bias, D = mha.in_proj_weight, mha.in_proj_bias, mha.in_proj_weight.shape[1]
                for i, lin in enumerate((att.q, att.k, att.v)):
                    lin.weight.copy_(W[i * D:(i + 1) * D])
                    lin.bias.copy_(bias[i * D:(i + 1) * D])
                att.o.weight.copy_(mha.out_proj.weight)
                att.o.bias.copy_(mha.out_proj.bias)
            split_mha(ref.news.mha, mine.news_att)
            split_mha(ref.user_mha, mine.user_att)
            for src, dst in ((ref.news.pool, mine.news_pool), (ref.user_pool, mine.user_pool)):
                dst.proj.load_state_dict(src.proj.state_dict())
                dst.q.load_state_dict(src.query.state_dict())
        rows = [r for r in data.test if len(r[2]) > 0][:50]
        sc = RM.make_scorer(mine)
        with torch.no_grad():
            want = ref.score(data.collate_eval(rows))
            worst = 0.0
            for b in R.EvalSet(rows):
                got = sc(b)
                for k, r in enumerate(b["rows"]):
                    n = int(b["cand_mask"][k].sum())
                    worst = max(worst, float((got[k, :n] - want[r, :n]).abs().max()))
        assert worst < 1e-4, worst
        ok(f"(17b) ladder rung 0 == V10's NRMS: identical scores with copied weights (max abs diff {worst:.1e} over {len(rows)} impressions)")
    else:
        print(f"SKIP (17b) V10 models.py not found at {path}")
    # every family learns the planted topic structure of the fixture
    seen = np.ones(data.n_news, bool)
    es_val = R.EvalSet(data.validation)
    out = {}
    for kind, cfg in base.items():
        cfg = replace(cfg, lr=3e-3, max_epochs=6, min_epochs=6, patience=99, bs=64)
        net = RM.build_net(cfg, "cpu", 0, text, data)
        val_fn = lambda n: R.evaluate(RM.make_scorer(n), es_val, seen)["mean"]["ndcg@10"]
        start = val_fn(net.eval())
        net, info = RM.train_ref(net, ts, val_fn, cfg, seed=0, log=lambda *a: None)
        h = info["history"]
        assert h[-1]["loss"] < h[0]["loss"] - 0.05 and info["best_val"] > start + 0.03, (kind, h[0]["loss"], h[-1]["loss"], start, info["best_val"])
        out[kind] = (round(start, 3), round(info["best_val"], 3))
    cfgf = replace(base["nrms"], lr=3e-3, max_epochs=6, bs=64)
    netf = RM.build_net(cfgf, "cpu", 0, text, data)
    calls = []
    netf, inf = RM.train_ref(netf, ts, lambda n: calls.append(1) or 0.0, cfgf, seed=0, log=lambda *a: None, fixed_epochs=2)
    assert inf["epochs"] == 2 and not calls and np.isnan(inf["history"][0]["val"]) and inf["history"][1]["loss"] < inf["history"][0]["loss"], "fixed_epochs must not validate"
    ok(f"(17) reference models: Fastformer attention == loop reference, masked encoders are padding-invariant (V10 layout is not), scorer == training forward, "
       f"empty history ties; NRMS/Fastformer/NAML learn the planted signal (val nDCG@10 untrained -> trained: {out})")



def test_ref_oom(data, text, ts):
    """(17c) a CUDA out-of-memory error during baseline training is retried with half / quarter the batch size and reported in the row."""
    from dataclasses import replace
    from types import SimpleNamespace
    cfg = replace(RM.preset("nrms"), emb_dim=16, heads=2, head_dim=8, init="random", max_epochs=1, min_epochs=1, bs=64)
    es = R.EvalSet(data.validation[:100])
    tried, orig = [], RM.train_ref

    def fake(net, trainset, val_fn, c, **kw):
        tried.append(c.bs)
        if c.bs > 16:
            raise torch.cuda.OutOfMemoryError("simulated")
        return orig(net, trainset, val_fn, c, **kw)
    RM.train_ref = fake
    try:
        net, info = SR._fit_with_oom_fallback(SimpleNamespace(device="cpu", data=data), cfg, ts, lambda n: R.evaluate(RM.make_scorer(n), es)["mean"]["ndcg@10"],
                                              0, None, text)
        assert tried == [64, 32, 16] and info["bs_used"] == 16, (tried, info["bs_used"])
        tried.clear()
        RM.train_ref = lambda *a, **k: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("always"))
        try:
            SR._fit_with_oom_fallback(SimpleNamespace(device="cpu", data=data), cfg, ts, None, 0, None, text)
            raise AssertionError("an unrecoverable OOM must surface")
        except torch.cuda.OutOfMemoryError:
            pass
    finally:
        RM.train_ref = orig
    ok("(17c) baseline OOM fallback: 64 -> 32 -> 16 recorded in the row; an unrecoverable OOM still raises")


def test_um_readers(data, E):
    """(18) Fastformer / NRMS / additive readers over frozen article vectors: finite, order-aware only where they should be, empty history ties."""
    pack = UM.pack_train(data.train_core[:600])
    rows = data.test[:80]
    for kind in ("add", "nrms", "ff"):
        m = UM.make_model(kind, E, 32, 0.2).eval()
        assert isinstance(m, UM.LLMEncUser)
        for b in R.EvalSet(rows):
            with torch.no_grad():
                s = m.score(b["hist"], b["hist_mask"], b["cand"])
                assert torch.isfinite(s).all()
                empty = ~b["hist_mask"].any(1)
                if empty.any():
                    assert float(s[empty].abs().max()) == 0.0
                perm = b["hist"].clone()
                pm = b["hist_mask"].clone()
                for k in range(len(perm)):                                   # reverse the valid history of every row
                    n = int(pm[k].sum())
                    perm[k, :n] = perm[k, :n].flip(0)
                s2 = m.score(perm, pm, b["cand"])
            same = torch.allclose(s[b["cand_mask"]], s2[b["cand_mask"]], atol=1e-4)
            assert same == (kind != "ff"), (kind, same)                      # add / nrms ignore order; Fastformer has positions
        val_fn = lambda mod: R.evaluate(UM.scorer(mod), R.EvalSet(data.validation[:120]))["mean"]["ndcg@10"]
        model, info = UM.train_user_model(E, pack, val_fn, kind=kind, dim=32, max_epochs=2, min_epochs=1, device="cpu", log=lambda *a: None)
        assert np.isfinite(info["best_val"]) and isinstance(model, UM.LLMEncUser)
    assert isinstance(UM.make_model("ca", E, 32, 0.2), UM.LLMEncCA)
    ok("(18) learned readers over frozen vectors: add/nrms are history-order invariant, the Fastformer reader is not; empty history ties; all train")



def test_ref_stages(tmp, mind, enc_dir, cache_dir, S_A):
    """(19) the reference-baseline stages inside the suite: rows, popularity twins, ladder table, harness check, exclusion from the
    direction summaries, per-model resume, the no-GloVe fallback and the subsampled QUICK path."""
    allw = sorted({w for t in S_A.meta["titles"] + S_A.meta["abstracts"] for w in RD.tokenize(t)})
    glove = os.path.join(tmp, "glove_fixture.txt")
    build_fake_glove(glove, allw, 32, seed=3)
    small = ["--ref-dims", "32,4,8,32,1", "--ref-max-epochs", "3", "--ref-repro", "0.6,0.6"]
    stages = "ref_nrms,ref_naml,ref_ff,ref_ff_refit,ref_nrms_refit,ref_nrms_s1,ref_l0,ref_l1,ref_l2"
    outR = os.path.join(tmp, "outR")
    args = [*mind, "--work", outR, "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", stages, *small, "--ref-glove-path", glove, *RUN_A[:-1]]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SR_ = suite.main(args)
    text = buf.getvalue()
    names = {r["name"] for r in SR_.records}
    need = {"ladder0_v10recipe", "ladder1_data", "ladder2_recipe", "nrms_ref", "naml_ref", "fastformer_ref", "nrms_ref+pop", "naml_ref+pop", "fastformer_ref+pop",
            "nrms_ref@1", "nrms_ref_refit", "fastformer_ref_refit"}
    assert not (need - names), sorted(need - names)
    assert not any(n.startswith("ladder") and n.endswith("+pop") for n in names) and "nrms_ref@1+pop" not in names, "only the headline models get popularity twins"
    assert SR_.row("nrms_ref@1")["ndcg@10"] != SR_.row("nrms_ref")["ndcg@10"], "the second seed must not reproduce the first"
    for r in SR_.records:
        for k in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            assert np.isfinite(r[k]), (r["name"], k)
    assert "harness check vs V10's own NRMS" in text and "ABLATION LADDER" in text and "huggingface" not in text.split("GloVe-32d")[0][-300:]
    assert os.path.getsize(os.path.join(outR, "ladder.md")) > 0 and "ladder0_v10recipe" in open(os.path.join(outR, "ladder.md")).read()
    assert all(r["family"].replace("+pop", "") not in suite.REFERENCE_FAMILIES for r in SR_.reps + SR_.reps_pop), "baselines must not compete as 'directions'"
    row = SR_.row("nrms_ref")
    assert row["bs_used"] == 64 and SR_.row("nrms_ref_refit")["bs_used"] == 64
    assert row["family"] == "ref-baseline" and row["ref"] == "frozen" and "d_ndcg@10" in row and row["init"].startswith("file:") and row["params"] > 0
    rf = SR_.row("nrms_ref_refit")
    assert rf["cfg"]["refit_epochs"] == SR_.row("nrms_ref")["best_epochs"][0] and rf["refit_of"] == "nrms_ref" and rf["ndcg@10"] != SR_.row("nrms_ref")["ndcg@10"]
    assert abs(rf["val_ndcg@10"] - SR_.row("nrms_ref")["val_ndcg@10"]) < 1e-9, "the refit row must repeat the epoch-selection validation columns"
    assert SR_.row("nrms_ref+pop")["ref"] == "frozen+pop" and SR_.row("ladder1_data")["cfg"]["dyn_neg"] and not SR_.row("ladder0_v10recipe")["cfg"]["dyn_neg"]
    rows = suite.leaderboard([S_A, SR_], ["A", "R"], ("A", "frozen"), os.path.join(tmp, "lb_ref.md"))
    assert {"nrms_ref", "fastformer_ref", "ladder0_v10recipe"} <= {x["name"] for x in rows if x["run"] == "R"}
    rows_pop = suite.leaderboard([S_A, SR_], ["A", "R"], ("A", "frozen+pop"), os.path.join(tmp, "lb_ref_pop.md"), pop=True)
    assert {"nrms_ref+pop", "naml_ref+pop"} <= {x["name"] for x in rows_pop if x["run"] == "R"}
    assert "streaming feedback" in open(os.path.join(tmp, "lb_ref_pop.md")).read()
    cc = suite.cross_compare([S_A, SR_], ["A", "R"], [(("A", "pool_lse"), ("R", "nrms_ref")), (("R", "nrms_ref"), ("R", "ladder0_v10recipe"))])
    assert len(cc) == 2 and cc[0]["n"] == len(S_A.data.test)
    # per-model resume: a second call with the same arguments finds every stage finished
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        suite.main([a for a in args if a != "--no-resume"])
    assert buf.getvalue().count("already finished (resume), skipped") >= 9 and "== reference baseline" not in buf.getvalue()
    # changing a reference-model setting re-runs only the reference stages; every other saved row stays
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SR2 = suite.main([a for a in args if a != "--no-resume"] + ["--ref-seeds", "2"])
    t2 = buf.getvalue()
    assert "reference-baseline settings changed" in t2 and t2.count("== reference baseline") == 9 and "stage 'controls' already finished" in t2
    assert SR2.row("nrms_ref")["seeds"] == 2 and len(SR2.row("nrms_ref")["ndcg10_per_seed"]) == 2 and SR2.row("nrms_ref")["ndcg10_std"] >= 0
    ff2, ffr = SR2.row("fastformer_ref"), SR2.row("fastformer_ref_refit")
    assert len(ff2["best_epochs"]) == 2 and ffr["cfg"]["refit_epochs"] == max(int(round(float(np.mean(ff2["best_epochs"])))), 1) and ffr["refit_of"] == "fastformer_ref"
    assert ffr["ref"] == "frozen" and ffr["family"] == "ref-baseline" and "fastformer_ref_refit+pop" not in {r["name"] for r in SR2.records}
    assert [r["name"] for r in SR2.records].count("frozen") == 1 and [r["name"] for r in SR2.records].count("nrms_ref") == 1
    # no GloVe anywhere: same models with random word vectors, labelled; the rung that would duplicate nrms_ref is skipped
    off = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
    saved = RD._glove_hf, RD._glove_gensim
    RD._glove_hf, RD._glove_gensim = off, off
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            SN = suite.main([*mind, "--work", os.path.join(tmp, "outN"), "--encoder", enc_dir, "--cache-dir", cache_dir, "--stages", "ref_nrms_refit,ref_l0,ref_l1,ref_l2,ref_nrms,ref_nrms_refit",
                             *small, "--ref-glove-path", os.path.join(tmp, "missing_glove.txt"), "--ref-train-frac", "0.5", *RUN_A[:-1]])
    finally:
        RD._glove_hf, RD._glove_gensim = saved
    nn_ = {r["name"] for r in SN.records}
    assert "nrms_ref_noglove" in nn_ and "nrms_ref" not in nn_ and "ladder2_recipe" not in nn_ and "no GloVe source worked" in buf.getvalue()
    assert "nrms_ref_refit_noglove" in nn_ and "has not run: its validated epoch count is needed" in buf.getvalue(), "refit before its base run must be skipped, then run after it"
    assert SN.row("nrms_ref_noglove")["init"] == "random" and "ladder0_v10recipe" in nn_ and "nrms_ref_noglove" in open(os.path.join(tmp, "outN", "ladder.md")).read()
    ok(f"(19) reference stages: {len(need)} rows + ladder table + harness check; baselines stay out of the direction summaries; per-model resume; "
       f"no-GloVe fallback is labelled '_noglove' and drops the duplicate rung; QUICK subsampling runs")
    return SR_

# --------------------------------------------------------------------------- more data and multi-interest
def test_fields(tmp, enc_dir, S):
    """(20) text variants, view fusion identity, content-keyed embedding cache."""
    meta = S.meta
    assert FD.phrase("football_nfl") == "football nfl" and FD.phrase("finance-companies") == "finance companies"
    assert FD.phrase("newsworld") == "newsworld" and FD.phrase(None) == ""
    assert FD.category_phrase("sports", "football_nfl") == "Category: sports. Subcategory: football nfl."
    assert FD.category_phrase("news", "news") == "Category: news." and FD.category_phrase("", "x") == ""
    t, ta, cta, ctae = (FD.build_texts(meta, m) for m in ("t", "ta", "cta", "ctae"))
    assert len(t) == len(ta) == len(cta) == len(ctae) == len(meta["ids"]) and ta == meta["text"]
    i = 5
    assert t[i] == meta["titles"][i].strip() and cta[i].startswith("Category: cat") and cta[i].endswith(ta[i])
    assert ctae[i].startswith(cta[i]) and "Entities: " in ctae[i] and meta["ent_labels"][i]
    assert all(lab in ctae[i] for lab in meta["ent_labels"][i][:FD.MAX_ENTITIES])
    assert len({tuple(x) for x in meta["ent_labels"]}) > 1 and all(len(set(x)) == len(x) for x in meta["ent_labels"]), "labels: distinct per article"
    try:
        FD.build_texts(meta, "nope")
        raise AssertionError("an unknown text mode must raise")
    except ValueError:
        pass
    fake = {"titles": ["A b", "C d"], "abstracts": ["", "x y"], "cats": ["sports", ""], "subcats": ["football_nfl", ""]}
    v = FD.view_texts(fake)
    assert v["a"] == ["A b", "x y"] and v["c"][0].startswith("Category: sports") and v["c"][1] == "Category: unknown."
    rng = np.random.default_rng(0)
    parts = [rng.standard_normal((6, 8)).astype(np.float32) for _ in range(3)]
    parts = [p_ / np.linalg.norm(p_, axis=1, keepdims=True) for p_ in parts]
    for w in FD.VIEW_WEIGHTS:
        assert abs(sum(w) - 1) < 1e-9
        Fm, Ft = FD.combine(parts, w), FD.combine([torch.from_numpy(p_) for p_ in parts], w).numpy()
        want = sum(wi * (p_ @ p_.T) for wi, p_ in zip(w, parts))
        assert np.allclose(Fm, Ft, atol=1e-6) and np.allclose(Fm @ Fm.T, want, atol=1e-5) and np.allclose(np.linalg.norm(Fm, axis=1), 1.0, atol=1e-5)
    calls = []

    def emb(texts):
        calls.append(list(texts))
        return np.array([[len(x), 1.0] for x in texts], np.float32)
    out = FD.embed_unique(emb, ["bb", "a", "bb", "a", "ccc"])
    assert calls == [["a", "bb", "ccc"]] and out[:, 0].tolist() == [2, 1, 2, 1, 3], "each distinct string is embedded once"
    cache = os.path.join(tmp, "fields_cache")
    A = ta[:40]
    v1, s1 = FD.embed_texts(A, enc_dir, 32, "", "cpu", 64, cache)
    v2, s2 = FD.embed_texts(A, enc_dir, 32, "", "cpu", 64, cache)
    v3, s3 = FD.embed_texts(A[::-1], enc_dir, 32, "", "cpu", 64, cache)                       # same length, other content: must NOT hit the cache
    assert s1 > 0 and s2 == 0.0 and np.array_equal(v1, v2) and s3 > 0 and not np.allclose(v1, v3) and np.allclose(v1[::-1], v3, atol=1e-5)
    assert len(os.listdir(cache)) == 2 and FD.digest(A) != FD.digest(A[::-1]) and FD.digest(A) == FD.digest(list(A))
    v4, s4 = FD.embed_texts(A, enc_dir, 16, "", "cpu", 64, cache)                             # another max_len is another key
    assert s4 > 0 and len(os.listdir(cache)) == 3
    aud = FD.audit(meta, S.data)
    assert aud["news_columns"] == 8 and 0 <= aud["abstract_empty"] <= 1 and aud["articles"] == len(meta["ids"]) and aud["test_history_max"] <= 10 and aud["test_history_over_50"] == 0.0
    ok("(20) text variants (t/ta/cta/ctae/views) are built as documented, view fusion keeps <u,v> = sum w_v <u_v,v_v>, "
       "the embedding cache is keyed by the texts themselves (same length + different content = miss)")


def test_multi_interest(data, E):
    """(21) zero-shot k-means interests (K=1 is the mean-pool control, K>=clicks is lse) and the learned multi-interest reader."""
    Et = torch.from_numpy(E)
    rows = data.test[:150]
    for tau in (0.05, 0.3):
        for b in R.EvalSet(rows):
            mp, lse = R.mean_pool_scorer(Et)(b), P.late_scorer(Et, "lse", tau)(b)
            assert torch.allclose(P.cluster_scorer(Et, 1, tau)(b), mp, atol=1e-5), "K=1 must be the mean-pool score"
            kmax = b["hist"].shape[1]
            assert torch.allclose(P.cluster_scorer(Et, kmax, tau)(b), lse, atol=1e-4), "K >= clicks must be late interaction (lse)"
            for K in (2, 3, 5):
                s = P.cluster_scorer(Et, K, tau)(b)
                assert torch.isfinite(s).all()
                empty = ~b["hist_mask"].any(1)
                if empty.any():
                    assert float(s[empty].abs().max()) == 0.0, "readers without history score 0"
                mu, share = P.interest_means(Et, b, K)
                n = b["hist_mask"].any(1)
                assert torch.allclose(share.sum(1)[n], torch.ones(int(n.sum())), atol=1e-5) and (share >= 0).all()
                assert ((share > 0).sum(1) <= torch.clamp(b["hist_mask"].sum(1), max=K)).all(), "no more interests than clicks"
    d = 6
    g1, g2 = F.normalize(torch.randn(1, d) + 0.02 * torch.randn(5, d), dim=-1), F.normalize(-torch.randn(1, d) + 0.02 * torch.randn(5, d), dim=-1)
    T2 = torch.zeros(11, d)
    T2[1:6], T2[6:11] = g1, g2
    hb = {"hist": torch.tensor([[1, 6, 2, 7, 3, 8, 4, 9, 5, 10]]), "hist_mask": torch.ones(1, 10, dtype=torch.bool)}
    mu, share = P.interest_means(T2, hb, 2)
    assert sorted(share[0].tolist()) == [0.5, 0.5], "two clean groups must be recovered"
    # the farthest-point start must pick *different* clicks (a min/max mix-up once made it pick the same click K times)
    hb3 = {"hist": torch.tensor([[1, 2, 3, 6, 7]]), "hist_mask": torch.ones(1, 5, dtype=torch.bool)}
    assert int((P.interest_means(T2, hb3, 5)[1] > 0).sum()) == 5, "5 distinct clicks, 5 interests -> 5 non-empty groups"
    pack = UM.pack_train(data.train_core[:600])
    val_fn = lambda mod: R.evaluate(UM.scorer(mod), R.EvalSet(data.validation[:120]))["mean"]["ndcg@10"]
    assert [UM.is_reader(k) for k in ("mi1", "mi4", "mi4d", "mi12", "ca", "ff")] == [True] * 6
    assert not any(UM.is_reader(k) for k in ("mi", "mi0", "mix", "mi4dd", "mi-1", ""))
    assert UM._mi_spec("mi4d") == (4, 0.1) and UM._mi_spec("mi4") == (4, 0.0)
    scores = {}
    for kind in ("mi1", "mi4", "mi4d"):
        m = UM.make_model(kind, E, 32, 0.2).eval()
        assert isinstance(m, UM.LLMEncMI) and m.K == int(kind[2]) and (m.div > 0) == kind.endswith("d")
        for b in R.EvalSet(rows[:60]):
            with torch.no_grad():
                s = m.score(b["hist"], b["hist_mask"], b["cand"])
                assert torch.isfinite(s).all()
                empty = ~b["hist_mask"].any(1)
                if empty.any():
                    assert float(s[empty].abs().max()) == 0.0
                perm, pm = b["hist"].clone(), b["hist_mask"]
                for k in range(len(perm)):
                    nk = int(pm[k].sum())
                    perm[k, :nk] = perm[k, :nk].flip(0)
                assert torch.allclose(s, m.score(perm, pm, b["cand"]), atol=1e-4), "the multi-interest reader has no positions"
            dv = m.diversity(b["hist"], b["hist_mask"])
            if m.K > 1:
                assert -1.0 <= dv["interest_cos"] <= 1.0 and 0.0 <= dv["attn_tv"] <= 1.0
                assert dv["attn_tv"] > 0.01, "interests must attend differently from the start (N(0,1) queries)"
            else:
                assert np.isnan(dv["interest_cos"])
            scores[kind] = s
        mm = UM.make_model(kind, E, 32, 0.2).train()
        mm.score(b["hist"], b["hist_mask"], b["cand"])
        aux = mm.aux_loss()
        assert (float(aux) > 0) == (kind.endswith("d")), "only the d-variant carries the disagreement regulariser"
        assert mm.aux_loss() == 0.0, "the regulariser is consumed once"
        model, info = UM.train_user_model(E, pack, val_fn, kind=kind, dim=32, max_epochs=2, min_epochs=1, device="cpu", log=lambda *a: None)
        assert np.isfinite(info["best_val"]) and isinstance(model, UM.LLMEncMI)
    ok("(21) multi-interest: cluster scorer == mean-pool at K=1 and == lse at K>=clicks, groups recovered, no more interests than clicks; "
       "learned reader finite, order-invariant, empty history ties, interests differ at init, regulariser only in the d-variant, trains")


def test_moredata_rows(S, text1, text2):
    """(22) rows of the moredata stage and of the multi-interest readers inside the full suite run."""
    names = {r["name"] for r in S.records}
    need = {"txt_t", "txt_t_lse", "txt_cta", "txt_cta_lse", "txt_ctae", "txt_ctae_lse", "txt_views", "txt_views_lse", "hist100_mean", "hist100_lse",
            "pool_km2", "pool_km3", "um_mi1_frozen", "um_mi4_frozen", "um_mi4d_frozen", "txt_t+pop", "txt_t_lse+pop", "hist100_lse+pop"}
    assert not (need - names), sorted(need - names)
    assert "txt_ta" not in names, "the suite's own text is the control (frozen), not a row of its own"
    r = S.row("txt_t_lse")
    assert r["ref"] == "pool_lse" and r["family"] == "F-text-lse" and r["cfg"]["tau"] in S.args.pool_taus and r["cfg"]["text"] == "t"
    assert S.row("txt_t")["ref"] == "frozen" and S.row("txt_t+pop")["ref"] == "frozen+pop" and S.row("txt_t")["family"] == "F-text"
    assert S.row("txt_t_lse+pop")["ref"] == "pool_lse+pop" and S.row("txt_t_lse+pop")["family"] == "F-text-lse+pop"
    assert S.row("txt_views")["cfg"]["w"] in [list(w) for w in FD.VIEW_WEIGHTS] and S.row("txt_views_lse")["cfg"]["tau"] in S.args.pool_taus
    for n in ("hist100_mean", "hist100_lse"):                                          # fixture histories have <= 10 clicks: reading 100 changes nothing
        assert abs(S.row(n)["d_ndcg@10"][0]) < 1e-6, n
    assert S.row("hist100_lse")["ref"] == "pool_lse" and S.row("hist100_mean")["ref"] == "frozen"
    bt = S.kv["best_text"]
    cands = {n: S.row(n)["val_ndcg@10"] for n in ("pool_lse", "txt_t_lse", "txt_cta_lse", "txt_ctae_lse", "txt_views_lse")}
    assert bt["name"] == max(cands, key=cands.get) and np.isfinite(bt["val_ndcg@10"]), "best text is chosen on validation among the lse rows and the control"
    assert (bt["spec"]["mode"] == "ta") == (bt["name"] == "pool_lse") and S.kv["data_audit"]["articles"] == len(S.meta["ids"])
    assert "news.tsv has 8 columns" in text1 and "MIND-small carries no article body" in text1 and ("as view 'text'" in text1 or "stays the text for the learned readers" in text1)
    assert S.row("pool_km2")["family"] == "H11-multi-interest" and S.row("pool_km2")["cfg"]["tau"] in S.args.pool_taus
    r1, r4 = S.row("um_mi1_frozen"), S.row("um_mi4_frozen")
    assert r1["cfg"]["reader"] == "mi1" and r4["cfg"]["reader"] == "mi4" and r4["cfg"]["seeds"] == 2 and S.row("um_frozen")["cfg"]["seeds"] == 2
    assert "interest_cos" not in r1 and len(r4["interest_cos"]) == 2 and len(r4["attn_tv"]) == 2 and len(r4["ndcg10_per_seed"]) == 2 and r4["ndcg10_std"] >= 0
    assert r4["ref"] == "um_frozen" and S.row("um_head")["cfg"]["seeds"] == 1 and "um_ff_knn_rag" in names
    assert "attention TV" in text2
    ok(f"(22) moredata + multi-interest rows: {len(need)} checked (references, popularity twins, validation-chosen text, history 100 == 50 on short histories, seeds)")


def test_text_view(S):
    """(22b) the learned readers on the best non-default text variant, forced (the fixture's random encoder decides the real choice)."""
    args = copy.copy(S.args)
    args.um_views, args.um_extra, args.um_seeds, args.um_seeds_focus = "text", "mi4:text,ff:text", 1, 1
    S.kv["best_text"] = {"name": "txt_cta_lse", "spec": {"mode": "cta"}, "val_ndcg@10": 0.0}
    S.stage = "usermodel_text"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        X.stage_usermodel(S, args)
    names = {r["name"] for r in S.records}
    assert {"um_text", "um_mi4_text", "um_ff_text"} <= names, buf.getvalue()[-600:]
    T = X.text_table(S, args, S.kv["best_text"]["spec"])
    assert T.shape[0] == S.data.n_news and np.allclose(np.linalg.norm(T[1:], axis=1), 1.0, atol=1e-4) and not np.allclose(T, S.E0)
    S.kv["best_text"] = {"name": "pool_lse", "spec": {"mode": "ta"}, "val_ndcg@10": 0.0}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        X.stage_usermodel(S, args)
    assert "view 'text': the suite's own title+abstract won" in buf.getvalue() and "mi4:text: skipped" in buf.getvalue()
    ok("(22b) view 'text' rebuilds the best variant from the cache and feeds um_text / um_mi4_text / um_ff_text; when the control wins the view is skipped, not duplicated")


# --------------------------------------------------------------------------------- main
RUN_A = ["--max-len", "32", "--q-max-len", "64", "--seeds", "2", "--pairs", "3000", "--pairs-enc", "600", "--epochs-head", "2",
         "--epochs-enc", "1", "--enc-steps", "8", "--bs-enc", "8", "--lora-r", "4", "--enc-bs", "64", "--device", "cpu", "--hist-k", "8",
         "--betas", "0.5", "1.0", "--knn-k", "5", "10", "--lams", "0.5", "1.0", "--gammas", "0.5", "--sid-k", "8", "--sid-d", "64",
         "--sid-layers", "2", "--sid-heads", "2", "--sid-epochs", "12", "--sid-lr", "3e-3", "--sid-val-n", "100", "--sid-bs", "64",
         "--um-epochs", "4", "--um-seeds", "1", "--lsa-dim", "32", "--llm-user-steps", "40", "--llm-user-bs", "8", "--llm-user-r", "4",
         "--llm-user-lr", "3e-3", "--llm-val-n", "100", "--llm-test-n", "150", "--text-modes", "t,cta,ctae,views", "--hist-lens", "100",
         "--km-ks", "2", "3", "--um-seeds-focus", "2", "--no-resume"]


def main():
    tmp = tempfile.mkdtemp(prefix="v11_selftest_")
    print("fixture:", tmp)
    common.CLOCK.reset()
    texts = build_fixture(tmp)
    enc_dir, dec_dir = os.path.join(tmp, "tinyenc"), os.path.join(tmp, "tiny-qwen3-embedding")
    build_tiny_encoder(enc_dir, texts)
    build_tiny_decoder(dec_dir, texts)
    mind = ["--mind-train", f"{tmp}/train", "--mind-dev", f"{tmp}/dev"]

    test_probe_criteria()
    test_shared_loss()
    test_torchao_shim(tmp)
    test_sid()
    test_generator(dec_dir, texts, tmp)

    # the guard that compares HFEncoder with a reference implementation must be exercised too
    ref_enc = adapt.HFEncoder(enc_dir, max_len=32, device="cpu")
    suite.EQUIV_REFERENCE = lambda t: ref_enc.encode(t)

    # ---------------- run A (encoder-type model): first half, then the rest in a second call that must RESUME
    out = os.path.join(tmp, "outA")
    part1 = "controls,pooling,moredata,graph,head,histquery,sid"
    buf1 = io.StringIO()
    with contextlib.redirect_stdout(buf1):
        suite.main([*mind, "--work", out, "--encoder", enc_dir, "--stages", part1, "--sweep-encoders", dec_dir, *RUN_A[:-1]])
    text1 = buf1.getvalue()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        S = suite.main([*mind, "--work", out, "--encoder", enc_dir, "--stages", f"{part1},encoder,llm_user,usermodel,sweep", "--sweep-encoders", dec_dir,
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
                "um_frozen", "um_head", "um_encoder", "um_entity_rag", "um_knn_rag", "enc_tfidf_lsa", "enc_tinyqwen3embedding",
                "ut_native", "ut_native+meanpool", "um_ff_frozen", "um_ff_knn_rag"}
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
    test_identical_verdict(S)
    test_popularity(S)
    test_v10_equivalence(S.data, S.E0)
    text_ref, ts_ref = test_ref_data(tmp, f"{tmp}/train", S.data, S.meta)
    test_ref_models(S.data, S.meta, text_ref, ts_ref)
    test_ref_oom(S.data, text_ref, ts_ref)
    test_um_readers(S.data, S.E0)
    test_fields(tmp, enc_dir, S)
    test_multi_interest(S.data, S.E0)
    test_moredata_rows(S, text1, text)
    test_ref_stages(tmp, mind, enc_dir, os.path.join(out, "cache"), S)

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
    test_text_view(S)

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
    exp_b = {"frozen", "histq_instr", "histq_instr+meanpool", "ut_causal", "ut_bidir", "ut_soft", "ut_causal+meanpool", "ut_soft+meanpool", "ut_causal@1",
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
    rows_pop = suite.leaderboard([S, SB], ["A", "B"], ("A", "frozen+pop"), os.path.join(tmp, "leaderboard_pop.md"), pop=True)
    assert len(rows_pop) > 15 and all(x["family"].endswith("+pop") or x["family"] == "control" for x in rows_pop)
    assert all(x["family"] != "control" or x["name"] in ("popularity", "frozen+pop") for x in rows_pop)
    cc = suite.cross_compare([S, SB], ["A", "B"], [(("B", "ut_causal"), ("B", "ut_bidir")), (("B", "ut_causal@1"), ("B", "ut_causal")),
                                                    (("B", "ut_causal+meanpool"), ("A", "pool_lse")), (("A", "ut_native"), ("B", "ut_causal")),
                                                    (("B", "nope"), ("A", "pool_lse"))], os.path.join(tmp, "direct.md"))
    assert len(cc) == 4 and all(np.isfinite(x["d_ndcg@10"][0]) for x in cc) and cc[0]["n"] > 50      # the unknown row is skipped, not fatal
    ok("(15) direct paired comparison across runs/subsets works (unknown rows are skipped)")
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
