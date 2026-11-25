"""Section (rubrique) of every presse3 / presse4 article (presse3: Le Figaro, Libération, JDD; presse4: La Croix, Le Parisien,
Les Échos, Mediapart, Paris Match, Télérama, Valeurs Actuelles; both minus the politique corpus), on cca
([RUB_CORPUS=presse4], default presse3):
  python explorer/presse3_rubriques.py join    n-gramsci/rubriques-presse on (media, id) -> <W>/presse3_rub/known.parquet,
                                               unmatched.parquet (doc_id, text[:3000]) for regicid/bert-rubriques
  (classify unmatched.parquet with bert-rubriques -> classified.parquet; L40S: gpu_l40s_interactive, srun only)
  python explorer/presse3_rubriques.py final   -> <W>/presse3_rub/rubriques.parquet: doc_id, rubrique, source"""
import glob, gzip, json, os, sys
import pandas as pd
CORPUS = os.environ.get("RUB_CORPUS", "presse3")
W = "/sps/humanum/home/bdecourson/ngramsci_work"; O = f"{W}/{CORPUS}_rub"; T = f"{W}/private/{CORPUS}/text"
MEDIA = {"presse3": ["le_figaro", "liberation", "le_journal_du_dimanche"],
         "presse4": ["la_croix", "leparisien", "les_echos", "mediapart", "paris_match", "telerama", "valeurs_actuelles"]}[CORPUS]


def join():
    from huggingface_hub import hf_hub_download
    os.makedirs(O, exist_ok=True)
    r = pd.read_parquet(hf_hub_download("n-gramsci/rubriques-presse", "rubriques_presse.parquet", repo_type="dataset"),
                        columns=["media", "id", "rubrique"],
                        filters=[("media", "in", MEDIA)])
    r["id"] = r["id"].astype(str)
    rows = []
    for f in sorted(glob.glob(f"{T}/shard_*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line); rows.append((d["doc_id"], d["media"], str(d["id"]), d["date"], d["text"][:3000]))
    t = pd.DataFrame(rows, columns=["doc_id", "media", "id", "date", "text"])
    k = t.merge(r.drop_duplicates(["media", "id"]), on=["media", "id"], how="left")
    print(CORPUS, "articles", len(t), "with a rubrique", int(k.rubrique.notna().sum()), flush=True)
    m = k[k.rubrique.isna()]
    print("unmatched by media and year:\n", m.groupby(["media", m.date.str[:4]]).size().unstack(0).fillna(0).astype(int).to_string(), flush=True)
    k[k.rubrique.notna()][["doc_id", "rubrique"]].to_parquet(f"{O}/known.parquet")
    m[["doc_id", "text"]].to_parquet(f"{O}/unmatched.parquet")


def final():
    k = pd.read_parquet(f"{O}/known.parquet").assign(source="rubriques-presse")
    c = pd.read_parquet(f"{O}/classified.parquet", columns=["doc_id", "rubrique"]).assign(source="bert-rubriques")
    a = pd.concat([k, c]).drop_duplicates("doc_id"); a.to_parquet(f"{O}/rubriques.parquet")
    print(len(a), a.rubrique.value_counts().to_dict(), a.source.value_counts().to_dict())


if __name__ == "__main__":
    {"join": join, "final": final}[sys.argv[1]]()
