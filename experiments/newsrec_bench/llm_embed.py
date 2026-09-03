"""Optional: precompute real LLM / sentence-transformer embeddings for MIND news.

The ``LLMEnc`` model reads ``MindData.llm_emb``. On synthetic data this is a
simulated embedding; for a faithful LLM run on real MIND, precompute embeddings
of the news titles (or title + abstract) with a sentence encoder and pass the
resulting dict to ``MindData.from_mind(..., llm_embeddings=...)``.

    pip install sentence-transformers
    python llm_embed.py --news /data/MINDsmall_train/news.tsv --out news_emb.npz

Then in your own script:

    import numpy as np
    from data import MindData
    z = np.load("news_emb.npz", allow_pickle=True)
    emb = {nid: v for nid, v in zip(z["ids"], z["vecs"])}
    data = MindData.from_mind(train_dir, dev_dir, llm_embeddings=emb)

Any encoder works — swap the model name for a larger instruction-tuned embedder
(e.g. a Qwen/E5/GTE model) to mirror an LLM-as-encoder pipeline.
"""
from __future__ import annotations

import argparse

import numpy as np


def read_titles(path: str) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            ids.append(p[0])
            title, abstract = p[3], (p[4] if len(p) > 4 else "")
            texts.append((title + ". " + abstract).strip())
    return ids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", required=True, help="path to MIND news.tsv")
    ap.add_argument("--out", default="news_emb.npz")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    ids, texts = read_titles(args.news)
    enc = SentenceTransformer(args.model)
    vecs = enc.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                      normalize_embeddings=True)
    np.savez(args.out, ids=np.array(ids), vecs=np.asarray(vecs, np.float32))
    print(f"Saved {len(ids)} embeddings of dim {vecs.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
