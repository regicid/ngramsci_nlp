"""Corpus extraction from the press scrape (one CSV per outlet: id, date, titre, contenu, url, section), into
jsonl.gz shards of N articles: {doc_id=<media>_<id>, id, media, url, date, section, text=titre///contenu}.
Filters (AND): --ids TSV (media<TAB>id, e.g. a rubrique from n-gramsci/rubriques-presse), --start/--end dates
(inclusive), --regex on title+text. Outlets are processed in parallel; each writes its own chunks, then `shard`
renumbers all chunks as shard_00000.jsonl.gz... (hard links, no copy)."""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

csv.field_size_limit(sys.maxsize)


def extract_media(csv_path, out, media=None, ids=None, start="", end="9999", regex=None, chunk=5000):
    media = media or os.path.basename(csv_path)[:-4]
    want = None
    if ids:
        want = {l.rstrip("\n").split("\t")[1] for l in open(ids, encoding="utf-8") if l.startswith(media + "\t")}
        if not want:
            return {"media": media, "wanted": 0, "found": 0, "chunks": 0}
    rx = re.compile(regex) if regex else None
    os.makedirs(out, exist_ok=True)
    buf, j, n, seen = [], 0, 0, set()

    def flush():
        nonlocal buf, j
        if buf:
            tmp = os.path.join(out, f".{media}_{j:04d}.tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                for r in buf:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, os.path.join(out, f"{media}_{j:04d}.jsonl.gz")); j += 1; buf = []
    for row in csv.DictReader(open(csv_path, encoding="utf-8", errors="replace")):
        i, d = row.get("id"), (row.get("date") or "")[:10]
        if i in seen or (want is not None and i not in want) or not (start <= d <= end):
            continue
        text = (row.get("titre") or "").strip() + "///" + (row.get("contenu") or "").strip()
        if rx and not rx.search(text):
            continue
        seen.add(i); n += 1
        buf.append({"doc_id": f"{media}_{i}", "id": i, "media": media, "url": row.get("url"), "date": d,
                    "section": row.get("section"), "text": text})
        if len(buf) == chunk:
            flush()
    flush()
    return {"media": media, "wanted": len(want) if want is not None else None, "found": n, "chunks": j}


def _one(csv_dir, out, kw, f):
    return extract_media(os.path.join(csv_dir, f), out, **kw)


def extract(csv_dir, out, medias=None, exclude=(), procs=12, **kw):
    files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    files = [f for f in files if (not medias or f[:-4] in medias) and f[:-4] not in exclude]
    with ProcessPoolExecutor(procs) as ex:
        for r in ex.map(partial(_one, csv_dir, out, kw), files):
            print(json.dumps(r), flush=True)


def shard(chunks_dir, shards_dir):
    """Number all chunks as shards (hard links) and write shards.tsv (shard -> chunk)."""
    os.makedirs(shards_dir, exist_ok=True)
    chunks = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".jsonl.gz"))
    with open(os.path.join(shards_dir, "shards.tsv"), "w") as m:
        for k, f in enumerate(chunks):
            dst = os.path.join(shards_dir, f"shard_{k:05d}.jsonl.gz")
            if not os.path.exists(dst):
                os.link(os.path.join(chunks_dir, f), dst)
            m.write(f"{k:05d}\t{f}\n")
    print(f"{len(chunks)} shards in {shards_dir}")
    return len(chunks)
