"""Semantic-graph / retrieval-augmented representations ("GraphRAG-lite").

Why this exists: V10 showed the *click* graph (LightGCN) is noise on MIND-small -- a
fresh article has no click edge.  Here every variant is **content-only** (article
embeddings + the Wikidata entities MIND annotates on titles/abstracts, never click
labels), so it can still say something about *cold* articles:

* ``entity_view``   entity-grounded retrieval: an article is enriched with the context
                    vector of its entities (centroid of older articles mentioning them).
* ``knn_view``      dense retrieval: enrich with its k nearest older articles.
* ``communities``   GraphRAG-style global view: Louvain communities on the article kNN
                    graph; a user's history is summarised at community level.
* ``user_rag``      collaborative retrieval: add the clicks of similar *other* users.

Causality: the memory (what may be retrieved) is restricted to articles observed before
the evaluation period -- validation uses train_core only, test uses train_core+validation.
This is retrieval-augmented *representation* (no text generation); an LLM that reads the
retrieved context is the natural next step and is left out on purpose.
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------- loading
def load_news_meta(train_dir, dev_dir):
    """Same article order as ``MindData.from_mind`` (global id = position + 1)."""
    by_id, n_cols = {}, 0
    for d in (train_dir, dev_dir):
        path = os.path.join(d, "news.tsv") if d else None
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 5:
                    n_cols = max(n_cols, len(p))
                    by_id[p[0]] = (p[1], p[3], p[4], p[6] if len(p) > 6 else "", p[7] if len(p) > 7 else "", p[2])
    ids = list(by_id)
    titles = [by_id[i][1] for i in ids]
    abstracts = [by_id[i][2] for i in ids]

    def wikidata(*cols):
        out = set()
        for c in cols:
            try:
                for e in json.loads(c) if c else []:
                    if e.get("WikidataId"):
                        out.add(e["WikidataId"])
            except (ValueError, TypeError, AttributeError):
                pass
        return sorted(out)

    def labels(*cols):
        """Entity names in order of first appearance (title entities first), duplicates removed."""
        out = []
        for c in cols:
            try:
                for e in json.loads(c) if c else []:
                    lab = (e.get("Label") or "").strip()
                    if lab and lab not in out:
                        out.append(lab)
            except (ValueError, TypeError, AttributeError):
                pass
        return out

    ents = [wikidata(by_id[i][3], by_id[i][4]) for i in ids]
    return {"ids": ids, "titles": titles, "abstracts": abstracts, "cats": [by_id[i][0] for i in ids],
            "subcats": [by_id[i][5] for i in ids],
            "text": [(t + ". " + a).strip() for t, a in zip(titles, abstracts)], "ents": ents,
            "ent_labels": [labels(by_id[i][3], by_id[i][4]) for i in ids], "n_cols": n_cols}


def observed_masks(data):
    """Articles the *training* rows observed (history / clicked / sampled negative) and the
    articles seen in validation slates.  Index 0 (PAD) is always False."""
    n = data.n_news
    tr = np.zeros(n, bool)
    for _user, hist, pos, negs in data.train_core:
        tr[np.asarray(hist, dtype=np.int64)] = True
        tr[pos] = True
        tr[np.asarray(negs, dtype=np.int64)] = True
    va = np.zeros(n, bool)
    for _i, _u, hist, cands, _labs, _pop in data.validation:
        va[np.asarray(hist, dtype=np.int64)] = True
        va[np.asarray(cands, dtype=np.int64)] = True
    tr[0] = va[0] = False
    return tr, va


def entity_matrix(ents, n_news):
    """Binary article x entity incidence (row 0 = PAD stays empty)."""
    vocab, rows, cols = {}, [], []
    for i, es in enumerate(ents):
        for e in es:
            rows.append(i + 1)
            cols.append(vocab.setdefault(e, len(vocab)))
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_news, max(len(vocab), 1)))
    return A


def _norm(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# ---------------------------------------------------------------------- memory
class Memory:
    """Retrieval memory over a boolean mask of admissible (older) articles."""

    def __init__(self, E, A, mem, device="cpu"):
        self.E = np.ascontiguousarray(E, dtype=np.float32)
        self.A, self.mem, self.device = A, np.asarray(mem, bool), device
        self.mem_ids = np.nonzero(self.mem)[0]
        self.n = len(E)

    # --- entity-grounded retrieval
    def entity_view(self, beta):
        Am = sp.diags(self.mem.astype(np.float32)) @ self.A
        df = np.asarray(Am.sum(0)).ravel()
        ctx = _norm(np.asarray(Am.T @ self.E, np.float32))               # entity context vectors
        idf = np.log((len(self.mem_ids) + 1.0) / (df + 1.0)) * (df > 0)  # no memory -> no context
        R = np.asarray((self.A @ sp.diags(idf.astype(np.float32))) @ ctx, np.float32)
        has = np.linalg.norm(R, axis=1) > 0
        out = self.E.copy()
        out[has] = _norm(self.E[has] + beta * _norm(R[has]))
        return out, float(has[1:].mean())

    # --- dense retrieval
    def neighbours(self, rows, k, chunk=2048):
        """Top-k admissible neighbours (self excluded) for ``rows`` -> (idx [r,k], sim [r,k])."""
        dev = self.device
        Em = torch.from_numpy(self.E[self.mem_ids]).to(dev)
        pos_in_mem = np.full(self.n, -1, np.int64)
        pos_in_mem[self.mem_ids] = np.arange(len(self.mem_ids))
        rows = np.asarray(rows)
        nbr, sim = [], []
        for s in range(0, len(rows), chunk):
            r = rows[s:s + chunk]
            sc = torch.from_numpy(self.E[r]).to(dev) @ Em.T
            pm = pos_in_mem[r]
            hit = np.nonzero(pm >= 0)[0]
            if len(hit):
                sc[torch.from_numpy(hit).to(dev), torch.from_numpy(pm[hit]).to(dev)] = float("-inf")
            v, i = sc.topk(min(k, sc.shape[1]), dim=1)
            nbr.append(self.mem_ids[i.cpu().numpy()])
            sim.append(v.cpu().numpy())
        return np.concatenate(nbr), np.concatenate(sim)

    def knn_view(self, beta, k, rows=None):
        rows = np.arange(1, self.n) if rows is None else np.asarray(rows)
        nbr, sim = self.neighbours(rows, k)
        w = np.clip(sim, 0, None)
        w = w / (w.sum(1, keepdims=True) + 1e-8)
        agg = (self.E[nbr] * w[:, :, None]).sum(1)
        out = self.E.copy()
        out[rows] = _norm(self.E[rows] + beta * _norm(agg))
        return out

    # --- GraphRAG-style communities
    def communities(self, k=10, resolution=1.0, seed=0, max_nodes=80_000):
        """Louvain communities on the memory kNN graph; every other article joins its nearest
        community centroid.  Returns (labels [n], centroids [K,d])."""
        ids = self.mem_ids
        try:
            if len(ids) > max_nodes:
                raise RuntimeError("memory too large for networkx Louvain")
            import networkx as nx
            nbr, sim = self.neighbours(ids, k)
            G = nx.Graph()
            G.add_nodes_from(ids.tolist())
            for a, row, sw in zip(ids, nbr, sim):
                for b, w in zip(row, sw):
                    if w > 0 and a != b:
                        G.add_edge(int(a), int(b), weight=float(w))
            comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
            lab_mem = np.zeros(len(ids), np.int64)
            where = {int(n): i for i, n in enumerate(ids)}
            for ci, c in enumerate(comms):
                for n in c:
                    lab_mem[where[n]] = ci
            method = "louvain"
        except Exception as exc:  # fall back to k-means centroids on the same embeddings
            from sklearn.cluster import MiniBatchKMeans
            K = int(max(8, min(500, len(ids) // 100)))
            lab_mem = MiniBatchKMeans(K, random_state=seed, n_init=3).fit_predict(self.E[ids])
            method = f"kmeans({exc.__class__.__name__})"
        K = int(lab_mem.max()) + 1
        cent = np.zeros((K, self.E.shape[1]), np.float32)
        np.add.at(cent, lab_mem, self.E[ids])
        cent = _norm(cent)
        labels = np.full(self.n, -1, np.int64)
        labels[ids] = lab_mem
        rest = np.nonzero(labels < 0)[0]
        rest = rest[rest > 0]
        if len(rest):
            labels[rest] = (self.E[rest] @ cent.T).argmax(1)
        return labels, cent, method


# --------------------------------------------------------------------- scorers
def community_scorer(E, labels, cent, lam, device="cpu"):
    """score = cos(user, cand) + lam * community-level affinity of the user's history."""
    Et = torch.from_numpy(np.ascontiguousarray(E, np.float32)).to(device)
    lab = torch.from_numpy(labels).to(device)
    C = torch.from_numpy(np.ascontiguousarray(cent, np.float32)).to(device)
    Cs, K = C @ C.T, len(cent)                                             # [K,K] community affinity

    def f(b):
        m = b["hist_mask"].unsqueeze(-1).float()
        u = (Et[b["hist"]] * m).sum(1) / m.sum(1).clamp(min=1)
        base = torch.einsum("bd,bcd->bc", u, Et[b["cand"]])
        hl = lab[b["hist"]].clamp(min=0)
        P = torch.zeros(hl.shape[0], K, device=Et.device).scatter_add_(1, hl, b["hist_mask"].float())
        P = P / P.sum(1, keepdim=True).clamp(min=1)                        # user's community histogram
        aff = P @ Cs                                                       # [B,K] affinity to each community
        return base + lam * aff.gather(1, lab[b["cand"]].clamp(min=0))
    return f


class UserMemory:
    """Profiles of users observed in training rows (history + clicks) and their click centroids."""

    def __init__(self, data, E, device="cpu"):
        U, N = data.n_users, data.n_news
        pu, pn, cu, cn = [], [], [], []
        for user, hist, pos, _negs in data.train_core:
            pu.extend([user] * len(hist)); pn.extend(hist)
            pu.append(user); pn.append(pos)
            cu.append(user); cn.append(pos)
        prof = sp.csr_matrix((np.ones(len(pu), np.float32), (pu, pn)), shape=(U, N)).sign()   # unique pairs
        clk = sp.csr_matrix((np.ones(len(cu), np.float32), (cu, cn)), shape=(U, N)).sign()
        self.profile = torch.from_numpy(_norm(np.asarray(prof @ E, np.float32))).to(device)
        self.clicks = torch.from_numpy(_norm(np.asarray(clk @ E, np.float32))).to(device)
        self.valid = (self.profile.norm(dim=1) > 0)
        self.device = device


def user_rag_scorer(E, um, gamma, device="cpu", k=20, chunk=256):
    """u' = u + gamma * mean(click centroids of the k most similar OTHER users)."""
    Et = torch.from_numpy(np.ascontiguousarray(E, np.float32)).to(device)
    prof, clicks, valid = um.profile.to(device), um.clicks.to(device), um.valid.to(device)

    def f(b):
        m = b["hist_mask"].unsqueeze(-1).float()
        u = (Et[b["hist"]] * m).sum(1) / m.sum(1).clamp(min=1)
        nb = torch.zeros_like(u)
        for s in range(0, u.shape[0], chunk):                              # bound the [chunk, n_users] similarity
            sim = F.normalize(u[s:s + chunk], dim=-1) @ prof.T
            sim = sim.masked_fill(~valid[None, :], float("-inf"))
            uid = b["user"][s:s + chunk].clamp(max=sim.shape[1] - 1).unsqueeze(1)
            sim.scatter_(1, uid, float("-inf"))                            # never retrieve the user itself
            top = sim.topk(min(k, sim.shape[1]), dim=1).indices
            nb[s:s + chunk] = clicks[top].mean(1)
        u2 = u + gamma * nb * (m.sum(1) > 0).float()                       # cold users keep their (empty) profile
        return torch.einsum("bd,bcd->bc", u2, Et[b["cand"]])
    return f
