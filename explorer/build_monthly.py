"""Name x month x newspaper x section table from the released characters files (politique, Le Monde, presse3, presse4), on cca:
  python explorer/build_monthly.py cells    -> <W>/explorer/monthly/cells_all.parquet (+ names, verbs), 8 processes
  python explorer/build_monthly.py top N    -> <W>/explorer/monthly/monthly_topN.parquet (N most mentioned names)
  python explorer/build_monthly.py sqlite [MIN]  -> <W>/explorer/monthly/portraits.sqlite (cells + names, indexed: the API
                                                db), names in at least MIN articles (default 100); then portraits_precompute.py adds
                                                cells_year (E_/EW_ expected-score sums) and totals_month

Sections: Le Monde (newspaper « lemonde ») by its rubrique (lemonde_rub/rubriques.parquet: culture, international,
politique, société, sport, économie); every politique-corpus outlet is section « politique »; presse3 (Le Figaro,
Libération, JDD outside the politique corpus) by presse3_rub/rubriques.parquet (rubriques-presse + bert-rubriques), presse4
(La Croix, Le Parisien, Les Échos, Mediapart, Paris Match, Télérama, Valeurs Actuelles) by presse4_rub/rubriques.parquet.
Display names have their apostrophes unified (« d’Estaing » -> « d'Estaing »; the keys already were).
Momentum-free valence: each passage weighted by P(pertinent) x (1 - P(dynamique != 0)) instead of P(pertinent):
W_nm, S_valence_nm (+ WW_nm, SS_valence_nm, WS_valence_nm): valence_nm = S_valence_nm / W_nm. Momentum passages carry
~22 % of the positive valence mass in the web press (2026-11-23), concentrated in campaign coverage.
Cells keep sums, not means, so any set of cells merges exactly: mean_k = S_k / W; with article sums (W_d, S_d) the
article-clustered variance of the mean is sum_d (S_d - mean W_d)^2 / W^2 = (SS - 2 mean WS + mean^2 WW) / W^2.

Per (key, month, newspaper): n_articles (articles naming the person; several chains of one name in an article count once),
n_mentions (sum of chain occurrences), n_segments (portrayal passages), W = sum P(pertinent) x passages, and the portrayal
means valence / extremisme / dynamique / journaliste = sum W x / W, as on the site (explorer.cells_from_characters).
Agency (propp slots + ext_rules): agent = agent - agent_drop + agent2 (verbs the person is the subject of, passive
agents « par X », control verbs), of which agent_speech (speech verbs, SPEECH); patient = patient + patient2 (objects,
passive subjects); obl = obl2 (indirect objects: « on lui reproche »).
Verbs: per (key, year, slot, lemma) counts for the names of CANDIDATES only (agent / patient), for the agency analysis.
"""
import functools, glob, json, os, sys
import multiprocessing as mp
from collections import Counter
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ngramsci.explorer import name_key, norm_apos

W_ = "/sps/humanum/home/bdecourson/ngramsci_work"; OUT = f"{W_}/explorer/monthly"
RUB = f"{W_}/lemonde_rub/rubriques.parquet"
RUBS = [RUB] + [f"{W_}/{c}_rub/rubriques.parquet" for c in ("presse3", "presse4")]
PER_MENTION = ("presse3", "presse4")       # releases without passage lists: momentum-free sums from the board passages


def board_passages(corpus, k):
    """The board passages of shard k (release cache): portrayal_all/ (score_all) where it exists, else portrayal/."""
    for d in ("portrayal_all", "portrayal"):
        f = f"{W_}/cache/release_{corpus}/{d}/shard_{k:05d}.parquet"
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"no board passages for {corpus} shard {k}")


def corpus_of(f):
    return next((c for c in PER_MENTION if f"/{c}/" in f), None)
DIMS = ["valence", "extremisme", "dynamique", "journaliste"]
SLOTS = ["agent", "agent2", "agent_drop", "patient", "patient2", "obl2"]
COLS = ["doc_id", "date", "newspaper", "full_name", "occurrence", "portrayal_n", "portrayal_pertinent"] + \
       [f"portrayal_{k}" for k in DIMS] + SLOTS
SPEECH = set("""dire déclarer affirmer estimer expliquer assurer annoncer ajouter indiquer souligner répondre lancer juger
rappeler préciser confier promettre dénoncer considérer penser insister avertir prévenir marteler écrire raconter
reconnaître admettre regretter déplorer saluer plaider répéter revendiquer fustiger tweeter commenter qualifier évoquer
mentionner citer prôner interroger demander proclamer clamer protester répliquer rétorquer confirmer démentir nier
soutenir prétendre jurer avouer lâcher glisser résumer conclure noter observer remarquer martèle""".split())

# first-round candidates, by election (first-round date)
ELECTIONS = {
 "1965-12-05": ["Charles de Gaulle", "François Mitterrand", "Jean Lecanuet", "Jean-Louis Tixier-Vignancour", "Pierre Marcilhacy", "Marcel Barbu"],
 "1969-06-01": ["Georges Pompidou", "Alain Poher", "Jacques Duclos", "Gaston Defferre", "Michel Rocard", "Louis Ducatel", "Alain Krivine"],
 "1974-05-05": ["Valéry Giscard d'Estaing", "François Mitterrand", "Jacques Chaban-Delmas", "Jean Royer", "Arlette Laguiller", "René Dumont",
                "Jean-Marie Le Pen", "Émile Muller", "Alain Krivine", "Bertrand Renouvin", "Jean-Claude Sebag", "Guy Héraud"],
 "1981-04-26": ["Valéry Giscard d'Estaing", "François Mitterrand", "Jacques Chirac", "Georges Marchais", "Brice Lalonde", "Arlette Laguiller",
                "Michel Crépeau", "Michel Debré", "Marie-France Garaud", "Huguette Bouchardeau"],
 "1988-04-24": ["François Mitterrand", "Jacques Chirac", "Raymond Barre", "Jean-Marie Le Pen", "André Lajoinie", "Antoine Waechter",
                "Pierre Juquin", "Arlette Laguiller", "Pierre Boussel"],
 "1995-04-23": ["Lionel Jospin", "Jacques Chirac", "Édouard Balladur", "Jean-Marie Le Pen", "Robert Hue", "Arlette Laguiller",
                "Philippe de Villiers", "Dominique Voynet", "Jacques Cheminade"],
 "2002-04-21": ["Jacques Chirac", "Jean-Marie Le Pen", "Lionel Jospin", "François Bayrou", "Arlette Laguiller", "Jean-Pierre Chevènement",
                "Noël Mamère", "Olivier Besancenot", "Jean Saint-Josse", "Alain Madelin", "Robert Hue", "Bruno Mégret", "Christiane Taubira",
                "Corinne Lepage", "Christine Boutin", "Daniel Gluckstein"],
 "2007-04-22": ["Nicolas Sarkozy", "Ségolène Royal", "François Bayrou", "Jean-Marie Le Pen", "Olivier Besancenot", "Philippe de Villiers",
                "Marie-George Buffet", "Dominique Voynet", "Arlette Laguiller", "José Bové", "Frédéric Nihous", "Gérard Schivardi"],
 "2012-04-22": ["François Hollande", "Nicolas Sarkozy", "Marine Le Pen", "Jean-Luc Mélenchon", "François Bayrou", "Eva Joly",
                "Nicolas Dupont-Aignan", "Philippe Poutou", "Nathalie Arthaud", "Jacques Cheminade"],
 "2017-04-23": ["Emmanuel Macron", "Marine Le Pen", "François Fillon", "Jean-Luc Mélenchon", "Benoît Hamon", "Nicolas Dupont-Aignan",
                "Jean Lassalle", "Philippe Poutou", "François Asselineau", "Nathalie Arthaud", "Jacques Cheminade"],
 "2022-04-10": ["Emmanuel Macron", "Marine Le Pen", "Jean-Luc Mélenchon", "Éric Zemmour", "Valérie Pécresse", "Yannick Jadot",
                "Jean Lassalle", "Fabien Roussel", "Nicolas Dupont-Aignan", "Anne Hidalgo", "Philippe Poutou", "Nathalie Arthaud"],
}
CANDIDATES = {name_key(n) for ns in ELECTIONS.values() for n in ns}


@functools.lru_cache(1)
def rubriques():
    return pd.concat([pd.read_parquet(f, columns=["doc_id", "rubrique"]) for f in RUBS]).set_index("doc_id")["rubrique"]


def momentum_free(ch, f):
    """Per character (doc_id, char_id): W_nm = sum w (1 - P(mom)), S_valence_nm = sum w (1 - P(mom)) (P(+) - P(-))."""
    if corpus_of(f):
        k = int(os.path.basename(f).split("-")[1].split(".")[0])
        p = pd.read_parquet(board_passages(corpus_of(f), k), columns=["doc_id", "chain", "p_pertinent", "p_valence_neg",
                                                                        "p_valence_pos", "p_dynamique_neg", "p_dynamique_pos"])
        w = p["p_pertinent"] * (1 - p["p_dynamique_neg"] - p["p_dynamique_pos"])
        t = pd.DataFrame({"doc_id": p["doc_id"], "char_id": p["chain"].astype(ch["char_id"].dtype), "W_nm": w,
                          "S_valence_nm": w * (p["p_valence_pos"] - p["p_valence_neg"])}).groupby(["doc_id", "char_id"]).sum()
        return ch.join(t, on=["doc_id", "char_id"]).fillna({"W_nm": 0.0, "S_valence_nm": 0.0})
    wn, sn = [], []
    for ps in ch["portrayal"]:
        a = b = 0.0
        for x in (ps if ps is not None else []):
            w = x["pertinent"] * (1 - x["dynamique_neg"] - x["dynamique_pos"]); a += w; b += w * (x["valence_pos"] - x["valence_neg"])
        wn.append(a); sn.append(b)
    return ch.assign(W_nm=wn, S_valence_nm=sn)


def words(a):
    return [] if a is None else [x["w"] for x in a]


def one(f):
    ch = pd.read_parquet(f, columns=COLS + ["char_id"] + ([] if corpus_of(f) else ["portrayal"]))
    ch = ch[ch["full_name"].notna()]
    ch["full_name"] = ch["full_name"].map(norm_apos)
    if "/lemonde/" in f:
        ch["newspaper"] = "lemonde"; ch["section"] = ch["doc_id"].map(rubriques()).fillna("inconnue")
    elif corpus_of(f):
        ch["section"] = ch["doc_id"].map(rubriques()).fillna("inconnue")
    else:
        ch["section"] = "politique"
    ch = momentum_free(ch, f)
    ch["key"] = ch["full_name"].map(name_key)
    ch = ch[(ch["key"] != "") & ch["date"].fillna("").str.match(r"\d{4}-\d{2}")].copy()   # undated articles: out
    ch["month"] = ch["date"].str[:7]
    ag, ag2, drop = (ch[s].map(words) for s in ("agent", "agent2", "agent_drop"))
    pa, pa2, ob = (ch[s].map(words) for s in ("patient", "patient2", "obl2"))
    # agent words minus the dropped ones (propp agents of passive subjects), plus the extended-rule agents
    agents = [[w for w in a if not (w in d and not d.remove(w))] + b for a, b, d in zip(ag, ag2, (list(x) for x in drop))]
    patients = [a + b for a, b in zip(pa, pa2)]
    ch["agent"] = [len(a) for a in agents]
    ch["agent_speech"] = [sum(w in SPEECH for w in a) for a in agents]
    ch["patient"] = [len(p) for p in patients]
    ch["obl"] = ob.map(len)
    ch["n_mentions"] = ch["occurrence"].astype(int)
    n = ch["portrayal_n"].fillna(0).astype(float)
    ch["n_segments"] = n
    ch["W"] = ch["portrayal_pertinent"].astype(float).fillna(0) * n
    for k in DIMS:
        ch[f"S_{k}"] = ch[f"portrayal_{k}"].astype(float).fillna(0) * ch["W"]
    num = ["n_mentions", "n_segments", "W"] + [f"S_{k}" for k in DIMS] + ["W_nm", "S_valence_nm"] + \
          ["agent", "agent_speech", "patient", "obl"]
    per = ch.groupby(["key", "doc_id"], sort=False).agg({**{c: "sum" for c in num}, "month": "first", "newspaper": "first",
                                                          "section": "first"})
    per["n_articles"] = 1
    per["WW"] = per["W"] ** 2
    per["WW_nm"], per["SS_valence_nm"], per["WS_valence_nm"] = per["W_nm"] ** 2, per["S_valence_nm"] ** 2, per["W_nm"] * per["S_valence_nm"]
    se = ["WW", "WW_nm", "SS_valence_nm", "WS_valence_nm"]
    for k in DIMS:
        per[f"SS_{k}"], per[f"WS_{k}"] = per[f"S_{k}"] ** 2, per["W"] * per[f"S_{k}"]; se += [f"SS_{k}", f"WS_{k}"]
    cells = per.groupby(["key", "month", "newspaper", "section"], sort=False)[["n_articles"] + num + se].sum().reset_index()
    names = ch.groupby(["key", "full_name"]).size().rename("n").reset_index()
    cv = Counter()
    m = ch["key"].isin(CANDIDATES).to_numpy()
    for key, date, a, p, in zip(ch["key"][m], ch["date"][m], (agents[i] for i in m.nonzero()[0]), (patients[i] for i in m.nonzero()[0])):
        for w in a: cv[(key, date[:7], "agent", w)] += 1
        for w in p: cv[(key, date[:7], "patient", w)] += 1
    verbs = pd.DataFrame([(*k, v) for k, v in cv.items()], columns=["key", "month", "slot", "lemma", "n"])
    return cells, names, verbs


def cells():
    fs = sorted(glob.glob(f"{W_}/release/data/characters_qwen4b/politique/*.parquet") +
                glob.glob(f"{W_}/release_lemonde/data/characters_qwen4b/lemonde/*.parquet") +
                [x for c in PER_MENTION for x in glob.glob(f"{W_}/release_{c}/data/characters_qwen4b/{c}/*.parquet")])
    print(len(fs), "files", flush=True)
    with mp.get_context("spawn").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
        parts = p.map(one, fs, chunksize=4)
    cs, ns, vs = zip(*parts)
    c = pd.concat(cs).groupby(["key", "month", "newspaper", "section"], sort=False).sum().reset_index()
    n = pd.concat(ns).groupby(["key", "full_name"])["n"].sum().reset_index()
    v = pd.concat(vs).groupby(["key", "month", "slot", "lemma"])["n"].sum().reset_index()
    os.makedirs(OUT, exist_ok=True)
    c.to_parquet(f"{OUT}/cells_all.parquet"); n.to_parquet(f"{OUT}/names_all.parquet"); v.to_parquet(f"{OUT}/verbs_candidates.parquet")
    json.dump(ELECTIONS, open(f"{OUT}/elections.json", "w"), ensure_ascii=False, indent=1)
    print("cells", len(c), "keys", c.key.nunique(), "months", c.month.min(), c.month.max(), "verbs", len(v), flush=True)


def top(N):
    c = pd.read_parquet(f"{OUT}/cells_all.parquet"); n = pd.read_parquet(f"{OUT}/names_all.parquet")
    disp = n.sort_values("n", ascending=False).drop_duplicates("key").set_index("key")["full_name"]
    tot = c.groupby("key")["n_mentions"].sum().sort_values(ascending=False)
    keep = tot.index[:N]
    d = c[c.key.isin(keep)].copy()
    for k in DIMS:
        d[k] = (d.pop(f"S_{k}") / d["W"]).where(d["W"] > 0).round(4)
    d["valence_nm"] = (d.pop("S_valence_nm") / d["W_nm"]).where(d["W_nm"] > 0).round(4)
    d["agency"] = (d["agent"] / (d["agent"] + d["patient"])).where(d["agent"] + d["patient"] > 0).round(4)
    d.insert(1, "name", d["key"].map(disp)); d.insert(2, "rank", d["key"].map({k: i + 1 for i, k in enumerate(keep)}))
    d["W"] = d["W"].round(3)
    d = d.drop(columns=[c for c in d.columns if c in ("WW", "WW_nm") or c.startswith(("SS_", "WS_"))])
    d = d.sort_values(["rank", "month", "newspaper", "section"]).reset_index(drop=True)
    d.to_parquet(f"{OUT}/monthly_top{N}.parquet"); d.to_csv(f"{OUT}/monthly_top{N}.csv.gz", index=False)
    print(len(d), "rows,", d.key.nunique(), "names, rank-1000 has", int(tot.iloc[N - 1]), "mentions", flush=True)


def sqlite(min_articles=100):
    """The API database: cells (sums, float32-rounded), names (key, display name, totals), one row per cell; only the
    names in at least min_articles articles (all newspapers and months)."""
    import sqlite3
    c = pd.read_parquet(f"{OUT}/cells_all.parquet"); n = pd.read_parquet(f"{OUT}/names_all.parquet")
    keep = c.groupby("key")["n_articles"].sum(); keep = keep.index[keep >= min_articles]
    c = c[c["key"].isin(keep)]; n = n[n["key"].isin(keep)]
    disp = n.sort_values("n", ascending=False).drop_duplicates("key").set_index("key")["full_name"]
    tot = c.groupby("key")[["n_articles", "n_mentions"]].sum()
    names = tot.join(disp).reset_index().rename(columns={"full_name": "name"})
    names["first_month"] = names["key"].map(c.groupby("key")["month"].min())
    names["last_month"] = names["key"].map(c.groupby("key")["month"].max())
    for col in c.columns:
        if c[col].dtype == "float64":
            c[col] = c[col].astype("float32")
    db = f"{OUT}/portraits.sqlite"
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    names.to_sql("names", con, index=False)
    c.to_sql("cells", con, index=False, chunksize=200000)
    con.executescript("""CREATE UNIQUE INDEX names_key ON names(key); CREATE INDEX names_mentions ON names(n_mentions);
                         CREATE INDEX cells_key ON cells(key, month); CREATE INDEX cells_month ON cells(month, newspaper, section);
                         VACUUM;""")
    con.close()
    # the site's precomputed tables (cells_year with E_/EW_ expected-score sums, totals_month), as on the API machine
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "portraits_precompute.py"), db], check=True)
    print(db, os.path.getsize(db) / 1e9, "GB,", len(c), "cells,", len(names), "names", flush=True)


if __name__ == "__main__":
    {"cells": cells, "top": lambda: top(int(sys.argv[2])), "sqlite": lambda: sqlite(int(sys.argv[2]) if len(sys.argv) > 2 else 100)}[sys.argv[1]]()
