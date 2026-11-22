"""Le Monde (propp-qwen run on cca) as an ngramsci disk corpus for portrayal, and its family-name counts.

Each published group (shard s, reuse|ner) becomes one ngramsci shard k (mapping.tsv): symlinks
  <work>/private/lemonde/text/shard_K.parquet       -> qcoref/lemonde/text/shard_S.parquet
  <work>/board/lemonde/mentions/shard_K.ner.parquet -> qcoref/lemonde/mentions/shard_S.<group>.parquet
  <work>/board/lemonde/entities/shard_K.parquet     -> qcoref/lemonde/out/entities_S_<group>.v3.parquet
usage: python lemonde_setup.py links            (once)
       python lemonde_setup.py counts I N       (task I of N: name counts of its shards -> counts/part_I.parquet)"""
import os, sys
import pandas as pd
sys.path.insert(0, "/sps/humanum/home/bdecourson/ngramsci_nlp/src")
Q = "/sps/humanum/home/bdecourson/qcoref/lemonde"; W = "/sps/humanum/home/bdecourson/ngramsci_work"
PRIV, BOARD = f"{W}/private/lemonde", f"{W}/board/lemonde"

def pairs():
    out = []
    for s in range(332):
        for g in ("reuse", "ner"):
            if os.path.exists(f"{Q}/out/entities_{s:05d}_{g}.v3.parquet") and os.path.exists(f"{Q}/mentions/shard_{s:05d}.{g}.parquet"):
                out.append((s, g))
    return out

def links():
    for d in (f"{PRIV}/text", f"{BOARD}/mentions", f"{BOARD}/entities"):
        os.makedirs(d, exist_ok=True)
    rows = []
    for k, (s, g) in enumerate(pairs()):
        for src, dst in ((f"{Q}/text/shard_{s:05d}.parquet", f"{PRIV}/text/shard_{k:05d}.parquet"),
                         (f"{Q}/mentions/shard_{s:05d}.{g}.parquet", f"{BOARD}/mentions/shard_{k:05d}.ner.parquet"),
                         (f"{Q}/out/entities_{s:05d}_{g}.v3.parquet", f"{BOARD}/entities/shard_{k:05d}.parquet")):
            if not os.path.lexists(dst):
                os.symlink(src, dst)
        rows.append((k, s, g))
    pd.DataFrame(rows, columns=["k", "shard", "group"]).to_csv(f"{BOARD}/mapping.tsv", sep="\t", index=False)
    print(len(rows), "shards")

def counts(i, n):
    from ngramsci.stages.portrayal import load_inputs, people
    m = pd.read_csv(f"{BOARD}/mapping.tsv", sep="\t"); rows = []
    for k in m.k[m.k % n == i]:
        docs, e = load_inputs(f"{PRIV}/text/shard_{k:05d}.parquet", f"{BOARD}/mentions/shard_{k:05d}.ner.parquet",
                              f"{BOARD}/entities/shard_{k:05d}.parquet", exclude_possessive=True)
        for d, g in e.groupby("doc_id", sort=False):
            if d not in docs.index:
                continue
            y = str(docs.at[d, "date"])[:4]
            rows += [(d, nm, int(y)) for nm in {n for _, n, _ in people(docs.at[d, "text"], g)} if y.isdigit()]
        print(k, len(rows), flush=True)
    from ngramsci.stages.portrayal import name_counts
    os.makedirs(f"{BOARD}/counts", exist_ok=True)
    name_counts(pd.DataFrame(rows, columns=["doc_id", "full_name", "year"])).to_parquet(f"{BOARD}/counts/part_{i:03d}.parquet")
    print("DONE", flush=True)

if __name__ == "__main__":
    {"links": lambda: links(), "counts": lambda: counts(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
