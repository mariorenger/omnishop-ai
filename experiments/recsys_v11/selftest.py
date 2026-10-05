"""End-to-end self-test on CPU: MIND-format fixture + tiny random BERT.

Numbers are meaningless (random encoder, toy data) -- the point is that every code path of the
suite runs (LoRA, DoRA, Matryoshka, Louvain communities, user-RAG, history-as-text queries,
paired CIs, exports) and that key invariants hold:

  1. the vectorised evaluator reproduces the V10 ``metrics.py`` on the suite's own scores;
  2. every residual head is the identity before training (step 0 == frozen control);
  3. an adaptation that does not beat the frozen embedding on validation never replaces it;
  4. all reported numbers are finite and every stage produced rows.

    python selftest.py            # ~1-3 min on CPU
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile

import numpy as np
import torch

import adapt
import metrics
import rep_eval as R
import suite


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


def main():
    tmp = tempfile.mkdtemp(prefix="v11_selftest_")
    print("fixture:", tmp)
    texts = build_fixture(tmp)
    enc_dir = os.path.join(tmp, "tinyenc")
    build_tiny_encoder(enc_dir, texts)

    # the guard that compares HFEncoder with a reference implementation must be exercised too
    ref_enc = adapt.HFEncoder(enc_dir, max_len=32, device="cpu")
    suite.EQUIV_REFERENCE = lambda t: ref_enc.encode(t)

    out = os.path.join(tmp, "out")
    S = suite.main([
        "--mind-train", f"{tmp}/train", "--mind-dev", f"{tmp}/dev", "--work", out, "--encoder", enc_dir,
        "--max-len", "32", "--q-max-len", "64", "--seeds", "2", "--pairs", "3000", "--pairs-enc", "600",
        "--epochs-head", "2", "--epochs-enc", "1", "--enc-steps", "8", "--bs-enc", "8", "--lora-r", "4",
        "--enc-bs", "64", "--device", "cpu", "--hist-k", "8", "--betas", "0.5", "1.0", "--knn-k", "5", "10",
        "--lams", "0.5", "1.0", "--gammas", "0.5"])

    names = {r["name"] for r in S.records}
    expected = {"frozen", "frozen_hf", "entity_rag", "knn_rag", "community_graphrag", "user_rag",
                "head_lin", "head_mlp+hn+mrl", "lora+mrl+hn", "dora+mrl+hn", "histq_instr", "histq_plain+meanpool"}
    missing = expected - names
    assert not missing, f"stages produced no rows for: {sorted(missing)}"
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
    print("OK (1) evaluator reproduces metrics.py on the suite's frozen scores")

    # (2) identity at step 0 for every residual head
    for kind in ("lin", "mlp"):
        head = adapt.HEADS[kind](E.shape[1])
        x = torch.from_numpy(E[1:50])
        assert torch.allclose(head(x), torch.nn.functional.normalize(x, dim=-1), atol=1e-6)
    print("OK (2) residual heads are the identity before training")

    # (3) never worse than frozen on validation by construction
    for r in S.records:
        if r["family"] == "H2-adapt":
            assert r["val_ndcg@10"] >= S.cache["frozen"]["val"]["mean"]["ndcg@10"] - 1e-9 or r["name"].startswith(("lora", "dora")), r["name"]
    print("OK (3) head adapters never fall below the frozen validation score")

    # (5) the hard-negative flag is not a silent no-op, (6) Matryoshka is active when d is large
    pairs = adapt.build_pairs(S.data, max_pairs=20000)
    ea, _ = adapt.train_head(E, pairs, "lin", epochs=2, bs=256, seed=0, log=lambda *a: None)
    eb, _ = adapt.train_head(E, pairs, "lin", hardneg=True, epochs=2, bs=256, seed=0, log=lambda *a: None)
    assert np.abs(ea - eb).max() > 1e-6, "hardneg=True produced identical embeddings (flag ignored?)"
    a = torch.nn.functional.normalize(torch.randn(32, 384), dim=-1)
    p_ = torch.nn.functional.normalize(a + 0.5 * torch.randn(32, 384), dim=-1)
    ix = torch.arange(32)
    assert abs(adapt.contrastive_loss(a, p_, ix).item() - adapt.contrastive_loss(a, p_, ix, dims=adapt.mrl_dims(384)).item()) > 1e-3
    print("OK (5)(6) hard negatives change training; Matryoshka loss differs from full-dim loss at d=384")

    # (7) the encoder-in-the-loop trainer really learns (loss falls, embeddings move, never below frozen on val)
    tx = [""] + S.meta["text"]
    val_ids = np.unique(np.concatenate([np.asarray(r[3], np.int64) for r in S.data.validation]
                                        + [np.asarray(r[2], np.int64) for r in S.data.validation]))
    es_val = R.EvalSet(S.data.validation)
    val_fn = lambda M: R.evaluate(R.mean_pool_scorer(torch.from_numpy(M)), es_val)["mean"]["ndcg@10"]
    logs = []
    enc = adapt.HFEncoder(enc_dir, max_len=32, device="cpu")
    Ef, info = adapt.train_encoder(enc, tx, pairs, E, mode="lora", hardneg=True, epochs=2, bs=32, lr=3e-3, r=8,
                                   val_ids=val_ids, val_fn=val_fn, max_steps=240, log=logs.append)
    losses = [float(l.split("loss=")[1].split()[0]) for l in logs if "step" in l]
    assert losses[-1] < losses[0], f"training loss did not fall: {losses}"
    assert np.abs(Ef - E).mean() > 1e-3 and val_fn(Ef) >= val_fn(E) - 1e-9
    print(f"OK (7) LoRA trainer learns: loss {losses[0]:.3f} -> {losses[-1]:.3f}, val nDCG@10 {val_fn(E):.4f} -> {val_fn(Ef):.4f}")

    # (8) direction-vs-direction comparison and per-impression arrays exist
    assert len(S.h2h) == len(S.reps) * (len(S.reps) - 1) // 2 and len(S.reps) == 4, (len(S.reps), len(S.h2h))
    assert all(np.isfinite(h["d_ndcg@10"][0]) for h in S.h2h)
    z = np.load(os.path.join(out, "test_arrays.npz"))
    assert z["frozen__auc"].shape[0] == len(S.data.test) and "histq_instr__ndcg@10" in z.files
    print(f"OK (8) head-to-head: {len(S.h2h)} direction pairs; test_arrays.npz holds {len(z.files)} arrays")

    # (4) outputs
    for f in ("results.md", "results.json"):
        assert os.path.getsize(os.path.join(out, f)) > 0
    assert any(n.startswith("emb_") for n in os.listdir(out)), "no embedding export"
    cold = S.cache["frozen"]["test"].get("cold_frac")
    assert cold is not None and 0.0 < cold < 1.0, f"fixture should contain cold clicks, got {cold}"
    print(f"OK (4) outputs written; cold-click share in the fixture = {cold:.1%}")
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
