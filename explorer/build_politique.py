"""On winston: portrayal shards (board) + dates (private text) -> explorer cells (cells.parquet, names.parquet).
Then: explorer.site(cells, names, "site") and publish explorer/index.html with site/index.json + site/data/*.json
(the Artifact « Political Portraits »). Run with src/ngramsci on sys.path (2026-11-22: copied next to this file)."""
import gzip, json, os, sys, time
from multiprocessing import Pool
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
try:
    from ngramsci import explorer as E
except ImportError:
    import explorer as E
from huggingface_hub import snapshot_download

W = "/scratch/decourson/explorer_pol"; TXT = "/scratch/decourson/ngramsci_work/private/politique/text"

def dates(k):
    out = {}
    with gzip.open(f"{TXT}/shard_{k:05d}.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line); out[d["doc_id"]] = d.get("date") or None
    return pd.Series(out, dtype=object)

def one(k):
    p = pd.read_parquet(f"{W}/board/portrayal/shard_{k:05d}.parquet")
    ds = dates(k)
    miss = (~p.doc_id.isin(ds.dropna().index)).sum()
    c, n = E.cells(p, ds)
    return c, n, miss, len(p), ds.dropna().str[:4].value_counts().to_dict()

if __name__ == "__main__":
    t = time.time()
    snapshot_download("regicid/qcoref-politique-work", repo_type="dataset", allow_patterns=["portrayal/*"], local_dir=f"{W}/board", max_workers=16)
    print("downloaded", round(time.time() - t), "s", flush=True)
    with Pool(48) as pool:
        res = pool.map(one, range(305))
    print("rows", sum(r[3] for r in res), "rows without date", sum(r[2] for r in res), flush=True)
    years = pd.DataFrame([r[4] for r in res]).sum().sort_index(); print(years.to_dict())
    c, n = E.merge([(r[0], r[1]) for r in res])
    c.to_parquet(f"{W}/cells.parquet"); n.to_parquet(f"{W}/names.parquet")
    print("cells", len(c), "keys", c.key.nunique(), round(time.time() - t), "s", flush=True)
    tot = c.groupby("key")[["n_docs", "n_seg"]].sum()
    for m in [5, 10, 20, 50]:
        k = tot.index[tot.n_docs >= m]; print(f"min_docs {m}: {len(k)} names, {c.key.isin(k).sum()} cells")
