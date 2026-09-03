"""Unified train / eval / timing harness for the news-rec benchmark.

Run examples
------------
# synthetic controlled demo (default) — every model, small + fast:
python bench.py --source synthetic --epochs 3

# a single model:
python bench.py --models nrms hybridopt --epochs 5

# real MIND (download MINDsmall from https://msnews.github.io first):
python bench.py --source mind \
    --mind-train /data/MINDsmall_train --mind-dev /data/MINDsmall_dev --epochs 4

Because the loss (softmax cross-entropy over candidates) and the metrics
(AUC / MRR / nDCG) are shared, the only thing that varies between rows of the
results table is the model architecture and its speed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import metrics
from data import MindData, iter_batches
from models import build_model


@dataclass
class Cfg:
    dim: int = 64
    dropout: float = 0.2
    heads: int = 2
    gcn_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 3
    batch_size: int = 64
    eval_batch_size: int = 128


def train_model(model, data, cfg, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    t0 = time.perf_counter()
    per_epoch = []
    for _ in range(cfg.epochs):
        e0 = time.perf_counter()
        total, nb = 0.0, 0
        for batch in iter_batches(data.train, cfg.batch_size, data.collate_train,
                                  shuffle=True):
            batch = batch.to(device)
            logits = model.score(batch)                       # [B, C], pos at 0
            target = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            nb += 1
        per_epoch.append(time.perf_counter() - e0)
    return time.perf_counter() - t0, float(np.mean(per_epoch)), total / max(nb, 1)


@torch.no_grad()
def evaluate(model, data, cfg, device):
    model.eval()
    impressions = []
    t0 = time.perf_counter()
    for batch in iter_batches(data.dev, cfg.eval_batch_size, data.collate_eval):
        batch = batch.to(device)
        logits = model.score(batch)
        logits = logits.masked_fill(~batch.cand_mask, float("-inf"))
        scores = logits.cpu().numpy()
        labels = batch.labels.cpu().numpy()
        mask = batch.cand_mask.cpu().numpy()
        for i in range(scores.shape[0]):
            m = mask[i]
            impressions.append((labels[i][m], scores[i][m]))
    eval_time = time.perf_counter() - t0
    result = metrics.aggregate(impressions)
    result["dev_impr_per_s"] = len(impressions) / max(eval_time, 1e-9)
    return result


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_table(rows: list[dict]) -> str:
    cols = ["model", "auc", "mrr", "ndcg@5", "ndcg@10",
            "params", "train_s", "s/epoch", "infer_impr/s"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"],
            f"{r['auc']:.4f}", f"{r['mrr']:.4f}",
            f"{r['ndcg@5']:.4f}", f"{r['ndcg@10']:.4f}",
            f"{r['params']:,}", f"{r['train_s']:.1f}",
            f"{r['s_per_epoch']:.2f}", f"{r['infer_impr_per_s']:.0f}",
        ]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="News recommendation benchmark")
    ap.add_argument("--source", choices=["synthetic", "mind"], default="synthetic")
    ap.add_argument("--mind-train", default=None)
    ap.add_argument("--mind-dev", default=None)
    ap.add_argument("--models", nargs="+",
                    default=["nrms", "naml", "fastformer", "lightgcn", "llmenc", "hybridopt"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--gcn-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    ap.add_argument("--out", default="results")
    # synthetic knobs
    ap.add_argument("--syn-users", type=int, default=2000)
    ap.add_argument("--syn-news", type=int, default=3000)
    ap.add_argument("--syn-train", type=int, default=8000)
    ap.add_argument("--syn-dev", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    if args.source == "mind":
        assert args.mind_train and args.mind_dev, "provide --mind-train and --mind-dev"
        print(f"Loading real MIND from {args.mind_train} / {args.mind_dev} ...")
        data = MindData.from_mind(args.mind_train, args.mind_dev, seed=args.seed)
    else:
        print("Generating synthetic MIND-shaped dataset (controlled demo) ...")
        data = MindData.synthetic(
            n_users=args.syn_users, n_news=args.syn_news,
            n_train=args.syn_train, n_dev=args.syn_dev, seed=args.seed,
        )
    print(f"  users={data.n_users}  news={data.n_news}  vocab={data.vocab_size}  "
          f"cats={data.n_cat}  train={len(data.train)}  dev={len(data.dev)}  device={device}")

    cfg = Cfg(dim=args.dim, dropout=args.dropout, heads=args.heads,
              gcn_layers=args.gcn_layers, lr=args.lr, epochs=args.epochs,
              batch_size=args.batch_size)

    rows = []
    for name in args.models:
        torch.manual_seed(args.seed)  # same init budget per model
        model = build_model(name, data, cfg)
        train_s, s_epoch, last_loss = train_model(model, data, cfg, device)
        res = evaluate(model, data, cfg, device)
        row = {
            "model": name, **{k: res[k] for k in ("auc", "mrr", "ndcg@5", "ndcg@10")},
            "params": n_params(model), "train_s": train_s, "s_per_epoch": s_epoch,
            "infer_impr_per_s": res["dev_impr_per_s"], "final_loss": last_loss,
        }
        rows.append(row)
        print(f"[{name:10s}] AUC={row['auc']:.4f} MRR={row['mrr']:.4f} "
              f"nDCG@10={row['ndcg@10']:.4f} | {row['params']:,} params | "
              f"{train_s:.1f}s train | {row['infer_impr_per_s']:.0f} impr/s")

    rows.sort(key=lambda r: r["auc"], reverse=True)
    table = format_table(rows)
    print("\n" + table)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"source": args.source, "config": vars(args), "rows": rows}, f, indent=2)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(f"# Benchmark results ({args.source})\n\n{table}\n")
    print(f"\nSaved -> {args.out}/results.json, {args.out}/results.md")


if __name__ == "__main__":
    main()
