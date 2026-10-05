"""Generative-LLM utilities for the retrieval-augmented / LLM-reranking stages (H4b, H6, H9).

* ``Generator.generate``      batched greedy generation (chat template, thinking off), JSONL-cached;
* ``Generator.yes_no``        pointwise judge: ``logit(Yes) - logit(No)`` of the first answer token
                              (the pointwise zero-shot ranking prompt of the LLM-as-reranker line, e.g. LLM4Rerank WWW'25).

Both honour an absolute ``deadline`` and report which items were not processed, so a slow GPU shrinks the
experiment (the stage records ``n``) instead of killing the notebook.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

import numpy as np
import torch

from common import load_hf, log

_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(text):
    text = _THINK.sub("", text)
    if "<think>" in text:                                   # truncated reasoning: nothing usable
        text = text.split("<think>")[0]
    return text.strip()


class Generator:
    def __init__(self, name, device="cpu", dtype=torch.float16, cache_path=None, trust_remote_code=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.device = name, device
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = load_hf(AutoModelForCausalLM, name, dtype, trust_remote_code).to(device).eval()
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[r["k"]] = r["v"]
                    except (ValueError, KeyError):
                        pass
        self.rate = {}                                       # kind -> (items, seconds) for budget planning
        yes, no = self.tok.encode("Yes", add_special_tokens=False), self.tok.encode("No", add_special_tokens=False)
        self.yes_id, self.no_id = yes[0], no[0]

    # -- prompts
    def render(self, text, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        if getattr(self.tok, "chat_template", None):
            try:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + "\n"

    def _key(self, kind, prompt, extra=""):
        return hashlib.sha1(f"{self.name}|{kind}|{extra}|{prompt}".encode("utf-8")).hexdigest()

    def _persist(self, items):
        if not self.cache_path:
            return
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for k, v in items:
                f.write(json.dumps({"k": k, "v": v}) + "\n")

    def _tok(self, rendered):
        return self.tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=1536,
                        add_special_tokens=False).to(self.device)

    def _note(self, kind, n, secs):
        a, b = self.rate.get(kind, (0, 0.0))
        self.rate[kind] = (a + n, b + secs)

    def seconds_per_item(self, kind, default=1.0):
        n, s = self.rate.get(kind, (0, 0.0))
        return s / n if n else default

    # -- generation
    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=64, bs=32, system=None, deadline=None, desc="generate"):
        """``prompts``: user messages.  Returns a list of strings (``None`` where the deadline cut generation short)."""
        rendered = [self.render(p, system) for p in prompts]
        keys = [self._key("gen", r, str(max_new_tokens)) for r in rendered]
        out = [self.cache.get(k) for k in keys]
        todo = [i for i, o in enumerate(out) if o is None]
        todo.sort(key=lambda i: len(rendered[i]))
        t0, done = time.time(), 0
        for s in range(0, len(todo), bs):
            if deadline is not None and time.time() > deadline:
                log(f"      [{desc}] deadline reached with {len(todo) - s} prompts left")
                break
            ix = todo[s:s + bs]
            t1 = time.time()
            batch = self._tok([rendered[i] for i in ix])
            gen = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
            texts = self.tok.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            new = []
            for i, t in zip(ix, texts):
                out[i] = clean(t)
                self.cache[keys[i]] = out[i]
                new.append((keys[i], out[i]))
            self._persist(new)
            self._note("gen", len(ix), time.time() - t1)
            done += len(ix)
            if (s // bs) % 20 == 19:
                log(f"      [{desc}] {done}/{len(todo)} generated ({time.time() - t0:.0f}s)")
        return out

    # -- pointwise judge
    @torch.no_grad()
    def yes_no(self, prompts, bs=32, system=None, deadline=None, desc="judge"):
        """``logit(Yes) - logit(No)`` for the first answer token of each prompt (NaN where not computed)."""
        rendered = [self.render(p, system) for p in prompts]
        keys = [self._key("yn", r) for r in rendered]
        out = np.array([self.cache.get(k, np.nan) for k in keys], np.float64)
        todo = [i for i in range(len(prompts)) if np.isnan(out[i])]
        todo.sort(key=lambda i: len(rendered[i]))
        t0, base = time.time(), getattr(self.model, "model", None)
        for s in range(0, len(todo), bs):
            if deadline is not None and time.time() > deadline:
                log(f"      [{desc}] deadline reached with {len(todo) - s} prompts left")
                break
            ix = todo[s:s + bs]
            t1 = time.time()
            batch = self._tok([rendered[i] for i in ix])
            h = base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state[:, -1]
            logits = self.model.lm_head(h).float()
            lo = (logits[:, self.yes_id] - logits[:, self.no_id]).cpu().numpy()
            new = []
            for i, v in zip(ix, lo):
                out[i] = float(v)
                self.cache[keys[i]] = float(v)
                new.append((keys[i], float(v)))
            self._persist(new)
            self._note("yn", len(ix), time.time() - t1)
            if (s // bs) % 50 == 49:
                log(f"      [{desc}] {s + len(ix)}/{len(todo)} judged ({time.time() - t0:.0f}s)")
        return out

    def free(self):
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ budget planning
def plan_prefix(item_lists, cap, per_impr=0.0):
    """Largest ``n`` such that ``|union(item_lists[:n])| + per_impr * n <= cap`` -> ``(n, n_items)``.
    Impressions are taken in a fixed random order, so a smaller ``n`` is a prefix (subset) of a larger one."""
    seen, best = set(), (0, 0)
    for n, items in enumerate(item_lists, 1):
        seen.update(items)
        if len(seen) + per_impr * n > cap:
            break
        best = (n, len(seen))
    return best
