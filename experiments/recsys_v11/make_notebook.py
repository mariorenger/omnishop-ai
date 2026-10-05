"""Build the self-contained Kaggle notebook ``v11_research_suite.ipynb`` from the module files."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ["data.py", "metrics.py", "rep_eval.py", "semgraph.py", "adapt.py", "suite.py"]


def lines(s):
    out = s.splitlines(keepends=True)
    return out or [""]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(t)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(t)}


def writefile(fn):
    with open(os.path.join(HERE, fn), encoding="utf-8") as f:
        return code(f"%%writefile {fn}\n" + f.read())


cells = [md("""# V11 decision suite — which LLM-embedding direction is worth a thesis on news recommendation?

Zero-shot **representation-level** benchmark on the exact **V10 protocol** (chronological validation split, `MINDsmall_dev` used once as test).
Every row is compared with the frozen embedding using a **paired 95% CI**, and split into **cold** (clicked article never seen in training) vs **warm**.

| Hypothesis | Technique (paper line) | Stage |
|---|---|---|
| H1 | history-as-text, instruction-aware query (LLM as user encoder; 2026 language-based user profiles) | `histquery` |
| H2 | co-click contrastive adaptation: residual heads, **LoRA / DoRA**, **Matryoshka (MRL)**, impression hard negatives (LLM2Rec, KDD'25) | `head`, `encoder` |
| H3 | entity / kNN **retrieval-augmented** article representations (RAG) | `graph` |
| H4 | **GraphRAG-lite**: Louvain communities + similar-user retrieval (K-RagRec, ACL'25 line) | `graph` |

**How to run (3 steps):** (1) Settings → Accelerator **GPU**, Internet **On**. (2) *Add Input* → a MIND dataset with **both** `MINDsmall_train` and `MINDsmall_dev` (e.g. `thinhhuynh3108/mindsmall`). (3) **Run All**.
Set `QUICK = True` first for a ~10 min smoke run on the real data.

The run ends with a **DECISION SUMMARY** (one representative per direction, chosen on validation, Bonferroni-adjusted), a **HEAD-TO-HEAD** table between the directions, and `test_arrays.npz` (per-impression metrics of every variant, so further paired analysis needs no re-run).

The first cell of results prints a **reproduction check**: the `frozen` row must match the V10 `bge_zeroshot` row (AUC 0.6241, nDCG@10 0.3909) — if it says MISMATCH, stop and fix before trusting anything else."""),
         md("### 0. Dependencies"),
         code("%pip install -q peft sentence-transformers\n"),
         md("### 1. Modules (written to the working directory)")]
cells += [writefile(m) for m in MODULES]
cells += [
    md("### 2. Configuration"),
    code('''import os
QUICK = bool(int(os.environ.get("V11_QUICK", "0")))        # True = ~10 min smoke run on the real data
ENCODER = os.environ.get("V11_ENCODER", "BAAI/bge-small-en-v1.5")
RUN_QWEN3 = bool(int(os.environ.get("V11_QWEN3", "0")))    # also run the cheap stages with Qwen3-Embedding-0.6B
QWEN3 = "Qwen/Qwen3-Embedding-0.6B"
STAGES = "controls,graph,head,encoder,histquery"
MIND_TRAIN = os.environ.get("V11_MIND_TRAIN")               # leave None -> auto-detect under /kaggle/input
MIND_DEV = os.environ.get("V11_MIND_DEV")
DEVICE = os.environ.get("V11_DEVICE", "auto")
WORK = "/kaggle/working/v11_out" if os.path.isdir("/kaggle/working") else "v11_out"
MAX_LEN = 256
'''),
    md("### 3. Locate MIND (needs a train + dev pair from the SAME dataset)"),
    code('''import glob
if not MIND_TRAIN:
    hits = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/news.tsv", recursive=True)})
    trs = [d for d in hits if "train" in d.lower()]
    dvs = [d for d in hits if "dev" in d.lower() or "valid" in d.lower()]
    for t in trs:
        sib = [d for d in dvs if os.path.dirname(d) == os.path.dirname(t)]
        if sib:
            MIND_TRAIN, MIND_DEV = t, sib[0]
            break
assert MIND_TRAIN and MIND_DEV and MIND_TRAIN != MIND_DEV, (
    "Attach a MIND dataset that has BOTH MINDsmall_train and MINDsmall_dev (e.g. thinhhuynh3108/mindsmall).")
count = lambda p: sum(1 for _ in open(p, encoding="utf-8"))
print("train:", MIND_TRAIN, "| behaviors:", f"{count(MIND_TRAIN + '/behaviors.tsv'):,}")
print("dev  :", MIND_DEV, "| behaviors:", f"{count(MIND_DEV + '/behaviors.tsv'):,}  (MIND-small: ~157k / ~73k)")
'''),
    md("### 4. Run the suite (encoder: BGE-small). Each stage is isolated: a failing stage is reported and the rest still run."),
    code('''import sys
sys.path.insert(0, os.getcwd())
import torch, suite

def reference_encoder(name, max_len):
    """Sentence-Transformers reference used to guard against wrong pooling/EOS/prefix conventions."""
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
        st.max_seq_length = max_len
        return lambda texts: st.encode(list(texts), normalize_embeddings=True)
    except Exception as e:
        print("reference encoder unavailable, skipping the pooling guard:", repr(e)[:120])
        return None

QUICK_ARGS = ["--seeds", "1", "--pairs", "20000", "--pairs-enc", "3000", "--enc-steps", "60", "--epochs-head", "1",
              "--betas", "0.5", "--knn-k", "10", "--lams", "1.0", "--gammas", "0.5", "--bs-enc", "16"]
repro = ["--repro", "0.6241,0.3909"] if "bge-small-en-v1.5" in ENCODER else []
suite.EQUIV_REFERENCE = reference_encoder(ENCODER, MAX_LEN)
S = suite.main(["--mind-train", MIND_TRAIN, "--mind-dev", MIND_DEV, "--work", WORK, "--encoder", ENCODER,
                "--stages", STAGES, "--device", DEVICE, "--max-len", str(MAX_LEN), *repro, *(QUICK_ARGS if QUICK else [])])
'''),
    md("### 5. (Optional) same cheap stages with Qwen3-Embedding-0.6B — a stronger LLM-based encoder (set `RUN_QWEN3 = True`)"),
    code('''if RUN_QWEN3:
    suite.EQUIV_REFERENCE = reference_encoder(QWEN3, MAX_LEN)
    S3 = suite.main(["--mind-train", MIND_TRAIN, "--mind-dev", MIND_DEV, "--work", WORK + "_qwen3", "--encoder", QWEN3,
                     "--stages", "controls,graph,head,histquery", "--device", DEVICE, "--max-len", str(MAX_LEN),
                     "--q-max-len", "512", "--enc-bs", "32", *(QUICK_ARGS if QUICK else [])])
'''),
    md("### 6. Results"),
    code('''print(open(os.path.join(WORK, "results.md"), encoding="utf-8").read())
print("exported embeddings (usable with V10 bench.py --news-emb):", [f for f in os.listdir(WORK) if f.startswith("emb_")])
'''),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "v11_research_suite_v2.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(cells))
