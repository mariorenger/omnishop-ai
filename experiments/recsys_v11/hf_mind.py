"""Get MIND (small or large) from Hugging Face into the layout the suite reads.

Microsoft's ``recommenders`` library hosts the MIND archives on the Hugging Face Hub dataset ``Recommenders/MIND`` (files
``MINDsmall_train.zip``, ``MINDsmall_dev.zip``, ``MINDlarge_train.zip``, ``MINDlarge_dev.zip``, ``MINDlarge_test.zip`` ...; the original Azure
blob no longer allows public access).  The suite only needs ``news.tsv`` and ``behaviors.tsv`` of the train and the dev archive
(MIND-large's *test* archive carries no labels and is never used).

``ensure_mind`` does not trust a fixed layout (it could not be checked from the development sandbox, whose egress policy blocks
huggingface.co): it uses extracted folders or archives that already sit in the Kaggle input directory, otherwise lists the repository, picks the
train / dev archives by name, falls back to extracted folders in the repository and to plain HTTP.  Every step is logged, so a failed run says exactly
what the repository looked like.
"""
from __future__ import annotations

import os
import re
import shutil
import time
import urllib.request
import zipfile

REPO = "Recommenders/MIND"
ARCHIVES = {"small": ("MINDsmall_train.zip", "MINDsmall_dev.zip"), "large": ("MINDlarge_train.zip", "MINDlarge_dev.zip")}
NEEDED = ("news.tsv", "behaviors.tsv")


# ------------------------------------------------------------------------------------------- local layout
def has_mind(d):
    return bool(d) and all(os.path.isfile(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > 0 for f in NEEDED)


def find_extracted(root, size):
    """(train_dir, dev_dir) among the directories under ``root`` that hold both files and whose path names the size and the split."""
    hits = sorted({dp for dp, _dn, fn in os.walk(root) if "news.tsv" in fn})
    hits = [d for d in hits if has_mind(d) and size in os.path.basename(d).lower()]       # the folder's own name decides (parents may say 'dev' for other reasons)
    tr = [d for d in hits if "train" in os.path.basename(d).lower()]
    dv = [d for d in hits if re.search(r"dev|valid", os.path.basename(d).lower())]
    return (tr[0], dv[0]) if tr and dv else (None, None)


def _count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            n += chunk.count(b"\n")
    return n


def validate(train_dir, dev_dir, log=print):
    """Cheap sanity checks: 5-column behaviors, labelled candidates ('<news id>-<0|1>'), news with >= 5 columns, separate train / dev."""
    if os.path.abspath(train_dir) == os.path.abspath(dev_dir):
        raise ValueError("train and dev point at the same directory")
    for name, d in (("train", train_dir), ("dev", dev_dir)):
        if not has_mind(d):
            raise ValueError(f"{name} directory {d} lacks news.tsv / behaviors.tsv")
        with open(os.path.join(d, "news.tsv"), encoding="utf-8") as f:
            first = [next(f, "") for _ in range(20)]
        if any(line and len(line.rstrip("\n").split("\t")) < 5 for line in first):
            raise ValueError(f"{d}/news.tsv: expected >= 5 tab-separated columns")
        with open(os.path.join(d, "behaviors.tsv"), encoding="utf-8") as f:
            lines = [next(f, "") for _ in range(50)]
        for line in lines:
            if not line:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                raise ValueError(f"{d}/behaviors.tsv: expected 5 tab-separated columns, got {len(p)}")
            if not all(re.fullmatch(r".+-[01]", it) for it in p[4].split()):
                raise ValueError(f"{d}/behaviors.tsv has unlabelled candidates (is this the MIND-large *test* archive, which carries no labels?)")
        log(f"  {name}: {d} | news {_count_lines(os.path.join(d, 'news.tsv')):,} lines, behaviors {_count_lines(os.path.join(d, 'behaviors.tsv')):,} lines "
            f"({os.path.getsize(os.path.join(d, 'behaviors.tsv')) / 1e6:.0f} MB)")


# ------------------------------------------------------------------------------------------- download
def list_repo(repo=REPO, token=None, log=print):
    """File names of the dataset repository ([] when the Hub cannot be listed)."""
    try:
        from huggingface_hub import HfApi
        files = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
        log(f"  Hugging Face dataset {repo}: {len(files)} files" + (": " + ", ".join(sorted(files)[:40]) + (" ..." if len(files) > 40 else "") if files else ""))
        return list(files)
    except Exception as e:
        log(f"  cannot list {repo} on the Hub ({repr(e)[:160]}); will try the standard file names")
        return []


def _hf_download(repo, filename, local_dir, token=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=local_dir, token=token)


def _stream(url, path, token=None, log=print, retries=4, chunk=1 << 22):
    """Plain HTTP download with resume (``Range``) and retries; used when ``huggingface_hub`` fails."""
    last = None
    for attempt in range(retries):
        have = os.path.getsize(path + ".part") if os.path.exists(path + ".part") else 0
        req = urllib.request.Request(url, headers={**({"Authorization": f"Bearer {token}"} if token else {}), **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resumed = have > 0 and getattr(r, "status", 200) == 206
                total = int(r.headers.get("Content-Length") or 0) + (have if resumed else 0)
                done, t0, mark = (have if resumed else 0), time.time(), 0.1
                with open(path + ".part", "ab" if resumed else "wb") as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b)
                        done += len(b)
                        if total and done / total >= mark:
                            log(f"    {os.path.basename(path)}: {done / 1e6:.0f}/{total / 1e6:.0f} MB ({time.time() - t0:.0f}s)")
                            mark += 0.1
                if total and done != total:                      # http.client returns b"" on a connection closed early instead of raising: check the length
                    raise IOError(f"connection closed after {done:,} of {total:,} bytes")
            os.replace(path + ".part", path)
            return path
        except Exception as e:
            if getattr(e, "code", None) == 416 and have > 0:       # the .part file already holds the whole resource (an earlier run died before the rename)
                os.replace(path + ".part", path)
                return path
            last = e
            log(f"    download attempt {attempt + 1}/{retries} failed ({repr(e)[:120]})")
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"could not download {url}: {last!r}")


def fetch_file(repo, filename, dest_dir, token=None, log=print):
    """One repository file into ``dest_dir`` (huggingface_hub first, plain HTTP second)."""
    os.makedirs(dest_dir, exist_ok=True)
    t0 = time.time()
    try:
        p = _hf_download(repo, filename, dest_dir, token)
        log(f"  downloaded {filename} via huggingface_hub ({os.path.getsize(p) / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
        return p
    except Exception as e:
        log(f"  huggingface_hub failed for {filename} ({repr(e)[:160]}); trying plain HTTP")
    p = _stream(f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}", os.path.join(dest_dir, os.path.basename(filename)), token, log)
    log(f"  downloaded {filename} over HTTP ({os.path.getsize(p) / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
    return p


def extract_needed(zip_path, out_dir, log=print):
    """Extract ``news.tsv`` and ``behaviors.tsv`` (wherever they sit inside the archive) flat into ``out_dir``; the entity .vec files are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        got = {}
        for info in z.infolist():
            base = os.path.basename(info.filename)
            if base in NEEDED and not info.is_dir():
                got[base] = info
        missing = [n for n in NEEDED if n not in got]
        if missing:
            raise ValueError(f"{zip_path}: no {missing} inside (members: {[i.filename for i in z.infolist()][:10]})")
        for base, info in got.items():                                    # the target name is fixed: no path from the archive is ever used
            with z.open(info) as src, open(os.path.join(out_dir, base), "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
    log(f"  extracted {sorted(got)} from {os.path.basename(zip_path)} -> {out_dir}")
    return out_dir


def _match(files, wanted):
    """The repository path of ``wanted`` (exact basename first, then a case-insensitive one)."""
    for f in files:
        if os.path.basename(f) == wanted:
            return f
    low = wanted.lower()
    for f in files:
        if os.path.basename(f).lower() == low:
            return f
    return None


def find_zips(roots, size):
    """``{split: path}`` of MIND<size>_train.zip / MIND<size>_dev.zip found under ``roots`` (names compared case-insensitively)."""
    want = {w.lower(): split for split, w in zip(("train", "dev"), ARCHIVES[size])}
    found = {}
    for root in roots:
        if root and os.path.isdir(root):
            for dp, _dn, fn in os.walk(root):
                for f in sorted(fn):
                    split = want.get(f.lower())
                    if split and split not in found:
                        found[split] = os.path.join(dp, f)
    return found


def ensure_mind(size="large", dest="mind_hf", repo=REPO, token=None, keep_zips=False, extra_roots=(), log=print):
    """-> ``(train_dir, dev_dir)`` holding ``news.tsv`` / ``behaviors.tsv`` of the MIND ``size`` train and dev archives.

    Order, per split: (1) already extracted under ``dest`` or ``extra_roots`` (e.g. a Kaggle input); (2) the archive in ``extra_roots`` (extracted into
    ``dest``, never deleted); (3) the Hub archive ``MIND<size>_train.zip`` / ``MIND<size>_dev.zip``; (4) extracted folders in the Hub repository.
    The Hub is not contacted when (1) or (2) covers both splits.  Raises ``RuntimeError`` with the repository listing otherwise."""
    if size not in ARCHIVES:
        raise ValueError(f"size must be one of {sorted(ARCHIVES)}")
    os.makedirs(dest, exist_ok=True)
    for root in (dest, *extra_roots):
        if root and os.path.isdir(root):
            tr, dv = find_extracted(root, size)
            if tr:
                log(f"MIND-{size}: using the files already under {root}")
                validate(tr, dv, log)
                return tr, dv
    local = find_zips(extra_roots, size)
    names = ARCHIVES[size]
    files, listed = [], False
    out = {}
    for split, wanted in zip(("train", "dev"), names):
        target = os.path.join(dest, wanted.replace(".zip", ""))
        if has_mind(target):
            out[split] = target
            continue
        if split in local:
            try:
                log(f"  {wanted}: using {local[split]}")
                extract_needed(local[split], target, log)
                out[split] = target
                continue
            except Exception as e:
                log(f"  {local[split]}: {repr(e)[:200]}")
        if not listed:
            files, listed = list_repo(repo, token, log), True
        path = _match(files, wanted) if files else wanted
        if path is not None:
            try:
                zp = fetch_file(repo, path, dest, token, log)
                extract_needed(zp, target, log)
                if not keep_zips:
                    for p in {zp, os.path.join(dest, os.path.basename(path))}:
                        if os.path.isfile(p):
                            os.remove(p)
                out[split] = target
                continue
            except Exception as e:
                log(f"  {wanted}: {repr(e)[:200]}")
        folder = [f for f in files if os.path.basename(f) in NEEDED and size in f.lower() and (split in f.lower() or (split == "dev" and re.search(r"valid", f.lower())))]
        if len({os.path.basename(f) for f in folder}) == 2:                # extracted layout: <something with size and split>/news.tsv + behaviors.tsv
            os.makedirs(target, exist_ok=True)
            for f in folder:
                shutil.copyfile(fetch_file(repo, f, os.path.join(dest, "_raw"), token, log), os.path.join(target, os.path.basename(f)))
            out[split] = target
    if "train" not in out or "dev" not in out:
        raise RuntimeError(
            f"could not get MIND-{size} {sorted(set(('train', 'dev')) - set(out))} from the Hugging Face dataset {repo}"
            + (f" (files there: {sorted(files)[:60]})" if files else " (the repository could not be listed: is Internet on?)")
            + ". Alternatives: Add Input -> a dataset that holds MIND" + size + "_train and MIND" + size + "_dev (the notebook looks in /kaggle/input), or set "
            "V11_HF_REPO to another dataset repository that holds the zip files.  If the repository is gated or private, add a Kaggle secret named HF_TOKEN.")
    validate(out["train"], out["dev"], log)
    return out["train"], out["dev"]
