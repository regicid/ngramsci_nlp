"""Le Monde release (cca): propp-qwen characters parts + per-character portrayal + name_source + date, under the
propp-press names data/characters_qwen4b/lemonde/part-SSSSS-<group>.parquet. Entities are left as published.
usage (cca htc array): python explorer/release_lemonde.py I N"""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ngramsci import release as Rl
from ngramsci.stages.portrayal import Surnames, load_inputs
Q = "/sps/humanum/home/bdecourson/qcoref/lemonde"; W = "/sps/humanum/home/bdecourson/ngramsci_work"
PRIV, BOARD, OUT = f"{W}/private/lemonde", f"{W}/board/lemonde", f"{W}/release_lemonde"
SN = Surnames.from_json(os.path.join(os.path.dirname(__file__), "..", "data", "surnames.json"))

def one(k, s, g):
    docs, ents = load_inputs(f"{PRIV}/text/shard_{k:05d}.parquet", f"{BOARD}/mentions/shard_{k:05d}.ner.parquet",
                             f"{BOARD}/entities/shard_{k:05d}.parquet", exclude_possessive=True)
    c = pd.read_parquet(f"{Q}/out/characters_{s:05d}_{g}.v3.parquet")
    # the score_all rerun (portrayal_all/, from 2026-11-25) only with PORTRAYAL_ALL=1, once every shard is scored: while it
    # runs, the board holds both and a release must not mix them (score_all passages also need Rl.locate(mode="chain"))
    allp = f"{BOARD}/portrayal_all/shard_{k:05d}.parquet"
    if os.environ.get("PORTRAYAL_ALL") == "1":
        p = Rl.locate(pd.read_parquet(allp), docs, ents, mode="chain")
    else:
        p = pd.read_parquet(f"{BOARD}/portrayal/shard_{k:05d}.parquet")
    co = Rl.characters(c, p, docs["date"], sources=Rl.name_sources(docs, ents, SN), newspaper="lemonde")
    f = f"{OUT}/data/characters_qwen4b/lemonde/part-{s:05d}-{g}.parquet"
    co.to_parquet(f, index=False)
    r = pd.read_parquet(f, columns=["portrayal_n", "date"])
    ok = len(r) == len(c) and int(r.portrayal_n.sum()) == len(p) and r.date.notna().all()
    return [k, bool(ok), len(c), int((r.portrayal_n > 0).sum()), len(p)]

if __name__ == "__main__":
    i, n = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(f"{OUT}/data/characters_qwen4b/lemonde", exist_ok=True)
    m = pd.read_csv(f"{BOARD}/mapping.tsv", sep="\t")
    res = []
    for r in m[m.k % n == i].itertuples():
        res.append(one(r.k, r.shard, r.group)); print(res[-1], flush=True)
    json.dump(res, open(f"{OUT}/slice_{i:03d}.json", "w")); print("DONE", flush=True)
