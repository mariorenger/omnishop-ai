"""Data layer for the news-recommendation benchmark.

Two sources, one interface:

* ``MindData.from_mind(train_dir, dev_dir)`` reads the real MIND dataset
  (``news.tsv`` + ``behaviors.tsv`` in each split; download from
  https://msnews.github.io). Nothing about the models below is MIND-specific.
* ``MindData.synthetic(...)`` generates a MIND-shaped dataset with a *known*
  generative process so every model family has something real to learn:
    - a per-news topic distribution drives the words in its title  -> content
      encoders (NRMS/NAML/Fastformer/LLM) can recover topic affinity;
    - a low-rank user/news latent factor adds collaborative signal not visible
      from text  -> the pure-graph model (LightGCN) can exploit it;
    - a frozen "LLM" embedding is a noisy linear image of the topic vector,
      standing in for a precomputed sentence-transformer / LLM encoding.
  This makes the quality comparison *meaningful as a controlled demo*; absolute
  numbers are not comparable to real-MIND leaderboards.

Everything is returned as plain tensors + index tables so the training loop in
``bench.py`` is identical for all models.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

PAD = 0  # reserved word / padding index


@dataclass
class Batch:
    hist_title: torch.Tensor  # [B, H, L] long
    hist_cat: torch.Tensor    # [B, H]    long
    hist_idx: torch.Tensor    # [B, H]    long (global news index)
    hist_mask: torch.Tensor   # [B, H]    bool (True = real click)
    cand_title: torch.Tensor  # [B, C, L] long
    cand_cat: torch.Tensor    # [B, C]    long
    cand_idx: torch.Tensor    # [B, C]    long
    cand_mask: torch.Tensor   # [B, C]    bool (True = real candidate; eval padding)
    user_idx: torch.Tensor    # [B]       long
    labels: torch.Tensor      # [B, C]    float (1 = clicked) — used for eval

    def to(self, device):
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f).to(device))
        return self


@dataclass
class MindData:
    # news feature tables, indexed by global news id (0 = pad news)
    news_title: np.ndarray            # [n_news, L] int
    news_cat: np.ndarray              # [n_news]    int
    n_news: int
    n_users: int
    vocab_size: int
    n_cat: int
    title_len: int
    max_hist: int
    llm_emb: np.ndarray               # [n_news, llm_dim] float32 (frozen)
    # samples
    train: list = field(default_factory=list)  # (user, hist[list], pos, [neg...])
    dev: list = field(default_factory=list)     # (user, hist[list], cands[list], labels[list])
    # graph adjacency (built lazily from train interactions)
    _edges: list = field(default_factory=list)  # (user, news) pairs seen in training

    # ---------------------------------------------------------------- batching
    def _pad_hist(self, hist: list[int]):
        hist = hist[-self.max_hist:]
        mask = [True] * len(hist)
        while len(hist) < self.max_hist:
            hist.append(PAD)
            mask.append(False)
        return hist, mask

    def _gather(self, idxs: list[int]):
        """Title + category rows for a list of news ids."""
        title = self.news_title[idxs]                 # [n, L]
        cat = self.news_cat[idxs]                     # [n]
        return title, cat

    def collate_train(self, rows) -> Batch:
        H, L = self.max_hist, self.title_len
        C = 1 + len(rows[0][3])  # 1 positive + n_neg
        B = len(rows)
        hist_title = np.zeros((B, H, L), np.int64)
        hist_cat = np.zeros((B, H), np.int64)
        hist_idx = np.zeros((B, H), np.int64)
        hist_mask = np.zeros((B, H), bool)
        cand_title = np.zeros((B, C, L), np.int64)
        cand_cat = np.zeros((B, C), np.int64)
        cand_idx = np.zeros((B, C), np.int64)
        user_idx = np.zeros((B,), np.int64)
        for b, (user, hist, pos, negs) in enumerate(rows):
            h, m = self._pad_hist(list(hist))
            hist_idx[b] = h
            hist_mask[b] = m
            hist_title[b], hist_cat[b] = self._gather(h)
            cands = [pos] + list(negs)
            cand_idx[b] = cands
            cand_title[b], cand_cat[b] = self._gather(cands)
            user_idx[b] = user
        cand_mask = np.ones((B, C), bool)
        labels = np.zeros((B, C), np.float32)
        labels[:, 0] = 1.0  # positive is always slot 0 during training
        return _to_batch(hist_title, hist_cat, hist_idx, hist_mask,
                         cand_title, cand_cat, cand_idx, cand_mask, user_idx, labels)

    def collate_eval(self, rows) -> Batch:
        """Eval impressions have variable candidate counts -> pad to max C."""
        H, L = self.max_hist, self.title_len
        C = max(len(r[2]) for r in rows)
        B = len(rows)
        hist_title = np.zeros((B, H, L), np.int64)
        hist_cat = np.zeros((B, H), np.int64)
        hist_idx = np.zeros((B, H), np.int64)
        hist_mask = np.zeros((B, H), bool)
        cand_title = np.zeros((B, C, L), np.int64)
        cand_cat = np.zeros((B, C), np.int64)
        cand_idx = np.zeros((B, C), np.int64)
        cand_mask = np.zeros((B, C), bool)
        labels = np.zeros((B, C), np.float32)
        user_idx = np.zeros((B,), np.int64)
        for b, (user, hist, cands, labs) in enumerate(rows):
            h, m = self._pad_hist(list(hist))
            hist_idx[b] = h
            hist_mask[b] = m
            hist_title[b], hist_cat[b] = self._gather(h)
            n = len(cands)
            cand_idx[b, :n] = cands
            t, c = self._gather(cands)
            cand_title[b, :n] = t
            cand_cat[b, :n] = c
            cand_mask[b, :n] = True
            labels[b, :n] = labs
            user_idx[b] = user
        return _to_batch(hist_title, hist_cat, hist_idx, hist_mask,
                         cand_title, cand_cat, cand_idx, cand_mask, user_idx, labels)

    # --------------------------------------------------------------- graph adj
    def build_adjacency(self) -> torch.Tensor:
        """Symmetric normalized bipartite adjacency for LightGCN.

        Node order: users [0, n_users) then news [n_users, n_users + n_news).
        Returns a sparse FloatTensor D^-1/2 (A) D^-1/2.
        """
        U, N = self.n_users, self.n_news
        rows, cols = [], []
        for u, n in self._edges:
            rows += [u, U + n]
            cols += [U + n, u]
        if not rows:  # degenerate guard
            rows, cols = [0], [0]
        idx = np.array([rows, cols], dtype=np.int64)
        vals = np.ones(idx.shape[1], np.float32)
        size = U + N
        deg = np.zeros(size, np.float64)
        np.add.at(deg, idx[0], vals)
        dinv = np.zeros(size, np.float64)
        nz = deg > 0
        dinv[nz] = deg[nz] ** -0.5
        norm = (dinv[idx[0]] * dinv[idx[1]]).astype(np.float32)
        return torch.sparse_coo_tensor(
            torch.from_numpy(idx), torch.from_numpy(norm), (size, size)
        ).coalesce()

    # -------------------------------------------------------------- factories
    @classmethod
    def synthetic(
        cls,
        n_users: int = 2000,
        n_news: int = 3000,
        n_topics: int = 16,
        words_per_topic: int = 50,
        title_len: int = 12,
        max_hist: int = 30,
        n_train: int = 8000,
        n_dev: int = 2000,
        n_neg: int = 4,
        cands_per_impr: int = 20,
        collab_dim: int = 8,
        llm_dim: int = 64,
        seed: int = 0,
    ) -> "MindData":
        rng = np.random.default_rng(seed)
        vocab_size = 1 + n_topics * words_per_topic  # +1 for PAD

        # --- latent structure -------------------------------------------------
        news_topic = rng.dirichlet(np.ones(n_topics) * 0.3, size=n_news)  # [N,T]
        news_topic[PAD] = 0
        news_cat = news_topic.argmax(1).astype(np.int64)  # category ~ dominant topic
        news_cat[PAD] = 0
        user_pref = rng.dirichlet(np.ones(n_topics) * 0.3, size=n_users)  # [U,T]
        # collaborative low-rank factors (signal NOT explainable from text)
        user_fac = rng.normal(scale=1.0, size=(n_users, collab_dim))
        news_fac = rng.normal(scale=1.0, size=(n_news, collab_dim))

        # --- titles: sample words from the news' topic mixture ----------------
        news_title = np.zeros((n_news, title_len), np.int64)
        for n in range(1, n_news):
            topics = rng.choice(n_topics, size=title_len, p=news_topic[n])
            offs = rng.integers(0, words_per_topic, size=title_len)
            news_title[n] = 1 + topics * words_per_topic + offs

        # --- frozen "LLM" embedding: noisy linear image of the topic vector ---
        proj = rng.normal(scale=1.0, size=(n_topics, llm_dim))
        llm_emb = news_topic @ proj + rng.normal(scale=0.1, size=(n_news, llm_dim))
        llm_emb = (llm_emb / (np.linalg.norm(llm_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

        def affinity(u, n):
            return 1.6 * float(user_pref[u] @ news_topic[n]) + 0.9 * float(
                user_fac[u] @ news_fac[n]
            ) / np.sqrt(collab_dim)

        # --- per-user click history -------------------------------------------
        hist_by_user: list[list[int]] = [[] for _ in range(n_users)]
        edges: list[tuple[int, int]] = []
        for u in range(n_users):
            cand = rng.integers(1, n_news, size=60)
            sc = np.array([affinity(u, int(n)) for n in cand])
            clicked = cand[sc > np.quantile(sc, 0.6)][:max_hist]
            hist_by_user[u] = list(map(int, clicked))
            edges += [(u, int(n)) for n in clicked]

        def make_impression(u, rng):
            hist = hist_by_user[u]
            pool = rng.integers(1, n_news, size=cands_per_impr * 3)
            pool = [int(n) for n in pool if n not in hist]
            sc = np.array([affinity(u, n) for n in pool])
            p = 1.0 / (1.0 + np.exp(-(sc - np.median(sc))))
            clicks = rng.random(len(pool)) < (0.5 * p)
            return hist, pool, clicks

        # --- train samples (1 pos + n_neg, softmax-over-candidates style) ------
        train = []
        tries = 0
        while len(train) < n_train and tries < n_train * 20:
            tries += 1
            u = int(rng.integers(0, n_users))
            if not hist_by_user[u]:
                continue
            hist, pool, clicks = make_impression(u, rng)
            pos_ids = [pool[i] for i in np.where(clicks)[0]]
            neg_ids = [pool[i] for i in np.where(~clicks)[0]]
            if not pos_ids or len(neg_ids) < n_neg:
                continue
            pos = int(rng.choice(pos_ids))
            negs = list(map(int, rng.choice(neg_ids, size=n_neg, replace=False)))
            train.append((u, list(hist), pos, negs))
            edges.append((u, pos))  # clicked item joins the training graph

        # --- dev impressions (full slate with binary labels) ------------------
        dev = []
        tries = 0
        while len(dev) < n_dev and tries < n_dev * 20:
            tries += 1
            u = int(rng.integers(0, n_users))
            if not hist_by_user[u]:
                continue
            hist, pool, clicks = make_impression(u, rng)
            if clicks.sum() == 0 or clicks.all():
                continue  # AUC undefined
            dev.append((u, list(hist), list(map(int, pool)), list(map(int, clicks))))

        obj = cls(
            news_title=news_title, news_cat=news_cat, n_news=n_news, n_users=n_users,
            vocab_size=vocab_size, n_cat=int(news_cat.max()) + 1, title_len=title_len,
            max_hist=max_hist, llm_emb=llm_emb, train=train, dev=dev, _edges=edges,
        )
        return obj

    @classmethod
    def from_mind(
        cls,
        train_dir: str,
        dev_dir: str,
        title_len: int = 20,
        max_hist: int = 50,
        n_neg: int = 4,
        min_word_freq: int = 3,
        llm_dim: int = 64,
        llm_embeddings: dict[str, np.ndarray] | None = None,
        seed: int = 0,
    ) -> "MindData":
        """Parse real MIND ``news.tsv`` / ``behaviors.tsv``.

        ``llm_embeddings`` maps MIND news id -> vector (e.g. from
        ``precompute_llm_embeddings``). If omitted, a random-projection
        placeholder is used so the pipeline still runs end to end.
        """
        import os

        rng = np.random.default_rng(seed)

        def read_news(path):
            rows = {}
            with open(path, encoding="utf-8") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    # id, category, subcategory, title, abstract, url, ...
                    rows[p[0]] = (p[1], p[3])
            return rows

        news_rows = read_news(os.path.join(train_dir, "news.tsv"))
        news_rows.update(read_news(os.path.join(dev_dir, "news.tsv")))

        # vocab + category maps
        from collections import Counter

        wf = Counter()
        for _, title in news_rows.values():
            wf.update(title.lower().split())
        vocab = {"<pad>": PAD}
        for w, c in wf.items():
            if c >= min_word_freq:
                vocab[w] = len(vocab)
        cats = {c for c, _ in news_rows.values()}
        cat_map = {c: i + 1 for i, c in enumerate(sorted(cats))}

        nid_map = {"<pad>": PAD}  # MIND news id -> global index
        for nid in news_rows:
            nid_map[nid] = len(nid_map)
        n_news = len(nid_map)
        news_title = np.zeros((n_news, title_len), np.int64)
        news_cat = np.zeros((n_news,), np.int64)
        for nid, (cat, title) in news_rows.items():
            gi = nid_map[nid]
            toks = [vocab.get(w, PAD) for w in title.lower().split()][:title_len]
            news_title[gi, : len(toks)] = toks
            news_cat[gi] = cat_map[cat]

        # llm embeddings aligned to global index
        if llm_embeddings:
            dim = len(next(iter(llm_embeddings.values())))
            llm_emb = np.zeros((n_news, dim), np.float32)
            for nid, gi in nid_map.items():
                if nid in llm_embeddings:
                    llm_emb[gi] = llm_embeddings[nid]
        else:  # placeholder — replace with real embeddings for a fair LLM run
            proj = rng.normal(size=(title_len, llm_dim))
            llm_emb = (news_title @ proj).astype(np.float32)
        llm_emb = (llm_emb / (np.linalg.norm(llm_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

        uid_map: dict[str, int] = {}

        def uidx(uid):
            if uid not in uid_map:
                uid_map[uid] = len(uid_map)
            return uid_map[uid]

        def read_behaviors(path):
            out = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    # impr_id, user, time, history, impressions
                    user, hist, imprs = p[1], p[3], p[4]
                    hist_ids = [nid_map[h] for h in hist.split() if h in nid_map]
                    cands, labs = [], []
                    for it in imprs.split():
                        nid, lab = it.split("-")
                        if nid in nid_map:
                            cands.append(nid_map[nid])
                            labs.append(int(lab))
                    out.append((uidx(user), hist_ids, cands, labs))
            return out

        train_beh = read_behaviors(os.path.join(train_dir, "behaviors.tsv"))
        dev = read_behaviors(os.path.join(dev_dir, "behaviors.tsv"))

        train, edges = [], []
        for user, hist, cands, labs in train_beh:
            edges += [(user, n) for n in hist]
            pos = [c for c, l in zip(cands, labs) if l == 1]
            neg = [c for c, l in zip(cands, labs) if l == 0]
            if not pos or len(neg) < 1:
                continue
            for p in pos:
                sampled = list(rng.choice(neg, size=min(n_neg, len(neg)),
                                          replace=len(neg) < n_neg))
                train.append((user, hist, p, sampled))
                edges.append((user, p))

        n_users = len(uid_map)
        return cls(
            news_title=news_title, news_cat=news_cat, n_news=n_news, n_users=n_users,
            vocab_size=len(vocab), n_cat=len(cat_map) + 1, title_len=title_len,
            max_hist=max_hist, llm_emb=llm_emb, train=train, dev=dev, _edges=edges,
        )


def _to_batch(*arrays) -> Batch:
    (hist_title, hist_cat, hist_idx, hist_mask, cand_title, cand_cat,
     cand_idx, cand_mask, user_idx, labels) = arrays
    t = torch.from_numpy
    return Batch(
        hist_title=t(hist_title), hist_cat=t(hist_cat), hist_idx=t(hist_idx),
        hist_mask=t(hist_mask), cand_title=t(cand_title), cand_cat=t(cand_cat),
        cand_idx=t(cand_idx), cand_mask=t(cand_mask), user_idx=t(user_idx),
        labels=t(labels),
    )


def iter_batches(rows, batch_size, collate, shuffle=False, seed=0):
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [rows[j] for j in order[i : i + batch_size]]
        yield collate(chunk)
