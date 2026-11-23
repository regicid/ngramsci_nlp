"""Page data from the released characters files (politique + Le Monde), on cca:
  python explorer/build_site.py cells          -> <W>/explorer/cells.parquet, names.parquet (8 processes)
  python explorer/build_site.py sizes          -> page sizes for a few thresholds
  python explorer/build_site.py site MIN MONTHLY -> <W>/explorer/site/ (index.json, data/, ranking.json)"""
import functools, glob, json, os, sys
import multiprocessing as mp
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ngramsci import explorer as E
W = "/sps/humanum/home/bdecourson/ngramsci_work"; OUT = f"{W}/explorer"
COLS = ["doc_id", "date", "newspaper", "full_name", "portrayal_n", "portrayal_pertinent",
        "portrayal_valence", "portrayal_extremisme", "portrayal_dynamique", "portrayal_journaliste"]

RUB = f"{W}/lemonde_rub/rubriques.parquet"   # explorer/lemonde_rubriques.py final

@functools.lru_cache(1)
def politique_ids():
    r = pd.read_parquet(RUB, columns=["doc_id", "rubrique"])
    return set(r.loc[r.rubrique == "politique", "doc_id"])

def one(f):
    ch = pd.read_parquet(f, columns=COLS)
    if "/lemonde/" in f and os.path.exists(RUB):   # Le Monde split: « politique » section vs every other section
        ch["newspaper"] = ch["doc_id"].isin(politique_ids()).map({True: "lemonde_politique", False: "lemonde_autres"})
    return E.cells_from_characters(ch)

def cells():
    fs = sorted(glob.glob(f"{W}/release/data/characters_qwen4b/politique/*.parquet") +
                glob.glob(f"{W}/release_lemonde/data/characters_qwen4b/lemonde/*.parquet"))
    print(len(fs), "files", flush=True)
    with mp.get_context("spawn").Pool(8) as p:
        parts = p.map(one, fs, chunksize=4)
    c, n = E.merge(parts)
    os.makedirs(OUT, exist_ok=True)
    c.to_parquet(f"{OUT}/cells.parquet"); n.to_parquet(f"{OUT}/names.parquet")
    print("cells", len(c), "names", c.key.nunique(), "years", c.month.min(), c.month.max(), flush=True)

def sizes():
    c = pd.read_parquet(f"{OUT}/cells.parquet"); c["year"] = c.month.str[:4]
    tot = c.groupby("key").n_docs.sum()
    for m in (20, 30, 50, 100):
        k = tot.index[tot >= m]; d = c[c.key.isin(k)]
        y = d.groupby(["key", "year", "outlet"]).size().shape[0]
        for mo in (300, 1000, 3000):
            big = tot.index[tot >= mo]; mm = d[d.key.isin(big)].groupby(["key", "month"]).size().shape[0]
            print(f"min {m}: {len(k)} names, {y} year-outlet cells; monthly >= {mo}: {mm} cells; ~{(y + mm) * 70 / 1e6:.0f} MB")

def site(min_docs, monthly):
    c = pd.read_parquet(f"{OUT}/cells.parquet"); n = pd.read_parquet(f"{OUT}/names.parquet")
    os.makedirs(f"{OUT}/site", exist_ok=True)
    print(E.site(c, n, f"{OUT}/site", min_docs=min_docs, monthly_docs=monthly))
    outs = json.load(open(f"{OUT}/site/index.json"))["outlets"]
    print(E.ranking(c, n, f"{OUT}/site", outlets=outs))

if __name__ == "__main__":
    {"cells": cells, "sizes": sizes, "site": lambda: site(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
