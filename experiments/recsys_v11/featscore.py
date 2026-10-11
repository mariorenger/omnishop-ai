"""History / slate features of every candidate, computed from the frozen article vectors (no parameters, nothing learned, no label).

The linear supermodel sees one number per candidate and component (similarity to the user's mean vector, to the closest click, ...).  The
stacker (``stacker.py``) can use more of what the frozen LLM vectors say about the *situation* a candidate is in:

  ``last1`` / ``last3``   cosine similarity to the most recent click / mean cosine to the three most recent ones (what the reader is reading *now*);
  ``in_hist``             the article is already in the reader's click history;
  ``cat_share`` / ``sub_share``   share of the reader's clicks in the candidate's category / sub-category (the ``cta`` text variants put the
                          category into the *vector*; this is the same fact as a plain number);
  ``cat_last``            the candidate has the category of the reader's most recent click;
  ``n_hist`` / ``n_cand`` length of the history / size of the slate (how much the other columns can be trusted);
  ``slate_mean`` / ``slate_max``  mean / maximum cosine to the OTHER candidates of the slate (an outlier among the candidates, or one of several
                          near-duplicates about the same story);
  ``slate_cat``           share of the slate in the candidate's category.

All of them are functions of the impression's history, its candidate list and the frozen vectors only.
"""
from __future__ import annotations

import numpy as np
import torch

NAMES = ("last1", "last3", "in_hist", "cat_share", "sub_share", "cat_last", "n_hist", "n_cand", "slate_mean", "slate_max", "slate_cat")
ENT_NAMES = ("ent_ov", "ent_ov3", "ent_frac", "ent_n")      # v16: entity overlap between the candidate and the reader's clicks (needs the article entities of news.tsv)
ENT_MAX = 12                                                # entities kept per article (the rarest ones when it has more)


def code_arrays(meta, n_news):
    """Integer codes of the category / sub-category of every article (row 0 = padding = code 0, codes start at 1)."""
    out = []
    for key in ("cats", "subcats"):
        vals = list(meta[key])
        codes = {v: i + 1 for i, v in enumerate(sorted(set(vals)))}
        arr = np.zeros(n_news, np.int64)
        arr[1:1 + len(vals)] = [codes[v] for v in vals]
        out.append(arr)
    return out


@torch.no_grad()
def batch_features(b, E, cat, sub):
    """``{name: [B, C] float32}`` for one EvalSet batch.  ``E``: [n_news, d] vectors (L2-normalised, row 0 = padding); ``cat`` / ``sub``: long tensors
    [n_news] of category / sub-category codes on the same device.  Entries of padded candidates are 0."""
    cand, hist, hm, cm = b["cand"], b["hist"], b["hist_mask"], b["cand_mask"]
    B, C = cand.shape
    H = hist.shape[1]
    dev = cand.device
    n = hm.sum(1)                                                              # [B] clicks in the history (chronological, left-aligned)
    has = (n > 0)[:, None]
    nf = n.clamp(min=1).to(torch.float32)[:, None]
    sim = torch.einsum("bcd,bhd->bch", E[cand], E[hist])                       # [B, C, H]
    pos = torch.arange(H, device=dev)[None, :]
    last = (n - 1).clamp(min=0)
    out = {}
    out["last1"] = torch.where(has, sim.gather(2, last[:, None, None].expand(B, C, 1)).squeeze(-1), torch.zeros(B, C, device=dev))
    k3 = ((pos >= (n[:, None] - 3)) & hm).to(torch.float32)                    # the (up to) three most recent clicks
    out["last3"] = torch.where(has, (sim * k3[:, None, :]).sum(-1) / k3.sum(1).clamp(min=1)[:, None], torch.zeros(B, C, device=dev))
    same_item = (cand[:, :, None] == hist[:, None, :]) & hm[:, None, :]
    out["in_hist"] = same_item.any(-1).to(torch.float32)
    for key, codes in (("cat_share", cat), ("sub_share", sub)):
        same = (codes[cand][:, :, None] == codes[hist][:, None, :]) & hm[:, None, :]
        out[key] = same.sum(-1).to(torch.float32) / nf
    last_code = cat[hist.gather(1, last[:, None])].squeeze(1)                  # category of the most recent click
    out["cat_last"] = ((cat[cand] == last_code[:, None]) & has).to(torch.float32)
    out["n_hist"] = n.to(torch.float32)[:, None].expand(B, C)
    ncand = cm.sum(1).to(torch.float32)
    out["n_cand"] = ncand[:, None].expand(B, C)
    cc = torch.einsum("bcd,bkd->bck", E[cand], E[cand])                        # [B, C, C] similarity among the candidates of the slate
    other = cm[:, None, :] & ~torch.eye(C, dtype=torch.bool, device=dev)[None]
    cnt = other.sum(-1).clamp(min=1).to(torch.float32)
    out["slate_mean"] = (cc * other).sum(-1) / cnt
    out["slate_max"] = cc.masked_fill(~other, float("-inf")).max(-1).values.clamp(min=-1.0)
    out["slate_max"] = torch.where(other.any(-1), out["slate_max"], torch.zeros_like(out["slate_max"]))
    samec = (cat[cand][:, :, None] == cat[cand][:, None, :]) & cm[:, None, :]
    out["slate_cat"] = samec.sum(-1).to(torch.float32) / ncand.clamp(min=1)[:, None]
    return {k: torch.where(cm, v.to(torch.float32), torch.zeros_like(v, dtype=torch.float32)) for k, v in out.items()}


def reference_features(hist, cands, vec, cat, sub):
    """Plain-python version of ``batch_features`` for ONE impression (``hist``: chronological list of article ids, ``cands``: list of ids,
    ``vec``: numpy [n_news, d], ``cat`` / ``sub``: numpy code arrays).  Used by the tests as the definition the vectorised code must agree with."""
    n, C = len(hist), len(cands)
    out = {k: np.zeros(C, np.float64) for k in NAMES}
    for j, c in enumerate(cands):
        if n:
            sims = [float(vec[c] @ vec[h]) for h in hist]
            out["last1"][j] = sims[-1]
            out["last3"][j] = float(np.mean(sims[-3:]))
            out["cat_share"][j] = sum(cat[h] == cat[c] for h in hist) / n
            out["sub_share"][j] = sum(sub[h] == sub[c] for h in hist) / n
            out["cat_last"][j] = float(cat[hist[-1]] == cat[c])
        out["in_hist"][j] = float(c in hist)
        out["n_hist"][j] = n
        out["n_cand"][j] = C
        others = [k for k in range(C) if k != j]
        if others:
            ss = [float(vec[c] @ vec[cands[k]]) for k in others]
            out["slate_mean"][j] = float(np.mean(ss))
            out["slate_max"][j] = max(ss)
        out["slate_cat"][j] = sum(cat[x] == cat[c] for x in cands) / C
    return out


# ------------------------------------------------------------------------------------------------ entity overlap (v16)
def entity_tables(ents, n_news, emax=ENT_MAX):
    """``(idx, idf)`` for the entity-overlap features.  ``ents``: one list of entity ids per article (article ``i`` has global id ``i + 1``; the Wikidata ids of
    ``news.tsv``); ``idx`` [n_news, emax] int64: the entities of every article as ids 1..n_ent (0 = none; an article with more than ``emax`` keeps its rarest
    ones); ``idf`` [n_ent + 1] float32: ``log((n_articles + 1) / (df + 1)) + 1`` over the catalogue's article texts (``idf[0] = 0``).  Text only: no label,
    no impression."""
    vocab, per = {}, [[] for _ in range(n_news)]
    for i, es in enumerate(ents):
        for e in es:
            per[i + 1].append(vocab.setdefault(e, len(vocab) + 1))
    n_ent = len(vocab)
    df = np.zeros(n_ent + 1, np.float64)
    for lst in per:
        if lst:
            df[np.unique(lst)] += 1.0
    n_art = max(len(ents), 1)
    idf = np.zeros(n_ent + 1, np.float32)
    idf[1:] = (np.log((n_art + 1.0) / (df[1:] + 1.0)) + 1.0).astype(np.float32)
    idx = np.zeros((n_news, emax), np.int64)
    for a, lst in enumerate(per):
        if lst:
            lst = sorted(set(lst), key=lambda j: (-idf[j], j))[:emax]
            idx[a, :len(lst)] = lst
    return idx, idf


@torch.no_grad()
def batch_entity_features(b, ent_idx, idf):
    """``{name: [B, C] float32}`` for one EvalSet batch, ``ENT_NAMES``: ``ent_ov`` = IDF-weighted number of the candidate's entities that occur among the
    entities of the reader's clicks; ``ent_ov3`` the same against the three most recent clicks only; ``ent_frac`` the share of the candidate's entities that
    occur in the history; ``ent_n`` how many entities the candidate has.  ``ent_idx`` / ``idf``: ``entity_tables`` as tensors on the batch's device.
    Entries of padded candidates are 0."""
    cand, hist, hm, cm = b["cand"], b["hist"], b["hist_mask"], b["cand_mask"]
    B, C = cand.shape
    H = hist.shape[1]
    dev = cand.device
    n_ent1 = idf.shape[0]
    n = hm.sum(1)
    k3 = ((torch.arange(H, device=dev)[None, :] >= (n[:, None] - 3)) & hm)
    he = ent_idx[hist]                                                              # [B, H, K]

    def present(mask):
        idx = torch.where(mask[:, :, None].expand_as(he), he, torch.zeros_like(he)).reshape(B, -1)
        out = torch.zeros(B, n_ent1, dtype=torch.bool, device=dev)
        out.scatter_(1, idx, True)
        out[:, 0] = False                                                           # column 0 collects the padding / masked-out positions
        return out
    ce = ent_idx[cand]                                                              # [B, C, K]
    valid = ce > 0
    w = idf[ce] * valid
    flat = ce.reshape(B, -1)
    p_all = present(hm).gather(1, flat).reshape(B, C, -1) & valid
    p_3 = present(k3).gather(1, flat).reshape(B, C, -1) & valid
    n_ent = valid.sum(-1).to(torch.float32)
    out = {"ent_ov": (p_all * w).sum(-1), "ent_ov3": (p_3 * w).sum(-1), "ent_frac": p_all.sum(-1).to(torch.float32) / n_ent.clamp(min=1.0), "ent_n": n_ent}
    return {k: torch.where(cm, v.to(torch.float32), torch.zeros_like(v, dtype=torch.float32)) for k, v in out.items()}


def reference_entity_features(hist, cands, ents, idf_of):
    """Plain-python version for ONE impression: ``hist`` / ``cands`` lists of article ids, ``ents``: {article id: list of entity ids kept for it (as in ``idx``)},
    ``idf_of``: {entity id: idf}."""
    C = len(cands)
    out = {k: np.zeros(C, np.float64) for k in ENT_NAMES}
    in_hist = set(e for h in hist for e in ents.get(h, []))
    in_last3 = set(e for h in hist[-3:] for e in ents.get(h, []))
    for j, c in enumerate(cands):
        es = ents.get(c, [])
        out["ent_n"][j] = len(es)
        out["ent_ov"][j] = sum(idf_of[e] for e in es if e in in_hist)
        out["ent_ov3"][j] = sum(idf_of[e] for e in es if e in in_last3)
        out["ent_frac"][j] = sum(1 for e in es if e in in_hist) / len(es) if es else 0.0
    return out
