"""Reading text shards: jsonl.gz, one article per line, {doc_id, text, media, date, ...}. Title and body are joined
by '///' in the saclay scrape (see ngramsci.extract)."""
import gzip
import json

import pandas as pd


def read_text(path: str, min_chars: int = 4) -> pd.DataFrame:
    """jsonl.gz (saclay scrape) or parquet (Le Monde shards: row, doc_id, url, date, section, text)."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            df = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    if df.empty:
        return pd.DataFrame(columns=["doc_id", "text"])
    df["doc_id"] = df["doc_id"].astype(str)
    df["text"] = df["text"].fillna("")
    return df[df["text"].str.len() >= min_chars].reset_index(drop=True)


def write_jsonl_gz(rows, path: str):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
