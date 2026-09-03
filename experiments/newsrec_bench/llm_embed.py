"""Precompute strong lightweight news embeddings (BGE-M3 / Jina / ST) for MIND.

The ``llmenc`` base model and the ``supermodel`` (with ``--news pretrained``) read
``MindData.llm_emb``. On synthetic data that is a simulated embedding; for a real
run on MIND, precompute embeddings of the news titles (or title + abstract) with a
modern sentence embedder and pass the dict to ``MindData.from_mind(..., llm_embeddings=)``.

    pip install sentence-transformers
    # lightweight & strong, pick one:
    python llm_embed.py --news /data/MINDsmall_train/news.tsv --out news_emb.npz \
        --model BAAI/bge-small-en-v1.5            # ~33M, very light (default)
    #   --model BAAI/bge-m3                       # multilingual, stronger (~560M)
    #   --model jinaai/jina-embeddings-v3         # multilingual (needs trust-remote-code)
    #   --model jinaai/jina-embeddings-v2-small-en  # ~33M, very light

Then in your own script:

    import numpy as np
    from data import MindData
    z = np.load("news_emb.npz", allow_pickle=True)
    emb = {nid: v for nid, v in zip(z["ids"], z["vecs"])}
    data = MindData.from_mind(train_dir, dev_dir, llm_embeddings=emb)

BGE-M3 and Jina are good news-embedding choices: multilingual, instruction-free,
and small enough to run on CPU. The projection head in the model adapts whatever
dimension the encoder outputs, so any sentence encoder drops in.
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
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="e.g. BAAI/bge-m3, jinaai/jina-embeddings-v3, or any ST model")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="required for jina-embeddings-v3 and some custom encoders")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    ids, texts = read_titles(args.news)
    enc = SentenceTransformer(args.model, trust_remote_code=args.trust_remote_code)
    vecs = enc.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                      normalize_embeddings=True)
    np.savez(args.out, ids=np.array(ids), vecs=np.asarray(vecs, np.float32))
    print(f"Saved {len(ids)} embeddings of dim {vecs.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
