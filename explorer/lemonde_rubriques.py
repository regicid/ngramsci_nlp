"""Section (rubrique) of every Le Monde article of the propp-qwen run, for the page's « Le Monde · politique » split.
  python explorer/lemonde_rubriques.py join      n-gramsci/rubriques-presse (media le_monde, key id) + saclay id<->url table
                                                 -> <W>/lemonde_rub/known.parquet, unmatched.parquet (doc_id, text)
  (classify unmatched.parquet with bert-rubriques -> classified.parquet)
  python explorer/lemonde_rubriques.py final     -> <W>/lemonde_rub/rubriques.parquet: doc_id, rubrique, source"""
import glob, os, sys
import pandas as pd
W = "/sps/humanum/home/bdecourson/ngramsci_work"; O = f"{W}/lemonde_rub"; Q = "/sps/humanum/home/bdecourson/qcoref/lemonde"

def join():
    from huggingface_hub import hf_hub_download
    r = pd.read_parquet(hf_hub_download("n-gramsci/rubriques-presse", "rubriques_presse.parquet", repo_type="dataset"),
                        columns=["media", "id", "rubrique"], filters=[("media", "==", "le_monde")])
    u = pd.read_csv(f"{O}/url_id.tsv.gz", sep="\t", names=["id", "url", "date"], dtype={"id": str}, quoting=3)
    r["id"] = r["id"].astype(str)
    m = u.merge(r[["id", "rubrique"]], on="id").drop_duplicates("url")
    t = pd.concat([pd.read_parquet(f, columns=["doc_id", "url", "text"]) for f in sorted(glob.glob(f"{Q}/text/shard_*.parquet"))])
    k = t.merge(m[["url", "rubrique"]], on="url", how="left")
    print("le monde articles", len(t), "with a rubrique", int(k.rubrique.notna().sum()), "rubriques-presse rows", len(r))
    k[k.rubrique.notna()][["doc_id", "rubrique"]].to_parquet(f"{O}/known.parquet")
    k[k.rubrique.isna()][["doc_id", "text"]].to_parquet(f"{O}/unmatched.parquet")

def final():
    k = pd.read_parquet(f"{O}/known.parquet").assign(source="rubriques-presse")
    c = pd.read_parquet(f"{O}/classified.parquet", columns=["doc_id", "rubrique"]).assign(source="bert-rubriques")
    a = pd.concat([k, c]).drop_duplicates("doc_id"); a.to_parquet(f"{O}/rubriques.parquet")
    print(len(a), a.rubrique.value_counts().to_dict(), a.source.value_counts().to_dict())

if __name__ == "__main__":
    {"join": join, "final": final}[sys.argv[1]]()
