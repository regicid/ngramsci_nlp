"""Public release of a corpus's coreference + portrayal results in regicid/propp-press (data/*_qwen4b/<corpus>/).

entities: the board's entities shard as is. characters: the board's characters shard + per-character portrayal (the
chains the portrayal stage scored: every singular person chain with score_all, named chains in the 2026-11-22 runs;
others get portrayal_n = 0 and nulls), and names from post-processing (people() at release time, for every character):
  full_name               the person's name, or null
  name_source             chain: the chain's own « First Last »; article: a family name (« X », « M. X », « Mme X ») named
                          by the article's one « First X »; family_name: « X » alone, the person it stands for that year
  portrayal_n             passages scored (one per sentence mentioning the person, with its context)
  portrayal_pertinent     mean P(the passage says something about the person)
  portrayal_valence       sum w (P(+) - P(-)) / sum w, w = P(pertinent); likewise extremisme, dynamique
  portrayal_journaliste   sum w P(journaliste) / sum w
  portrayal               the passages in text order: {kind, title, quote, pertinent, journaliste,
                          valence_neg, valence_pos, extremisme_neg, extremisme_pos, dynamique_neg, dynamique_pos}
                          (P(0) = 1 - neg - pos)
Both tables: newspaper = the outlet (doc_id prefix) instead of the corpus name (unless given, e.g. lemonde), and date
(YYYY-MM-DD) is added."""
from __future__ import annotations

import numpy as np
import pandas as pd

DIMS = ["valence", "extremisme", "dynamique"]


def outlet(doc_id: pd.Series) -> pd.Series:
    return doc_id.str.rsplit("_", n=1).str[0]


def with_meta(t: pd.DataFrame, dates: pd.Series, newspaper=None) -> pd.DataFrame:
    t = t.copy()
    t["newspaper"] = (pd.Series(newspaper, index=t.index) if newspaper else outlet(t["doc_id"].astype(str))).astype(object)
    t.insert(t.columns.get_loc("doc_id") + 1, "date", t["doc_id"].map(dates).astype("string").str[:10])
    return t


def name_sources(docs: pd.DataFrame, ents: pd.DataFrame, surnames=None) -> pd.DataFrame:
    """(doc_id, chain, full_name, name_source) of every named person chain, with the portrayal stage's naming."""
    from .stages.portrayal import people
    rows = []
    for d, g in ents.groupby("doc_id", sort=False):
        if d not in docs.index:
            continue
        y = str(docs.at[d, "date"])[:4] if "date" in docs.columns else ""
        for ch, name, _, src in people(docs.at[d, "text"], g, surnames=surnames, year=int(y) if y.isdigit() else None,
                                       with_source=True):
            rows.append((d, ch, name, src))
    return pd.DataFrame(rows, columns=["doc_id", "chain", "full_name", "name_source"])


def per_character(port: pd.DataFrame) -> pd.DataFrame:
    p = port.sort_values(["doc_id", "chain", "mention"], kind="stable")
    w = p["p_pertinent"].to_numpy()
    agg = pd.DataFrame({"doc_id": p["doc_id"].to_numpy(), "chain": p["chain"].to_numpy(), "w": w,
                        "journaliste": w * p["p_journaliste"].to_numpy()})
    for k in DIMS:
        agg[k] = w * (p[f"p_{k}_pos"].to_numpy() - p[f"p_{k}_neg"].to_numpy())
    g = agg.groupby(["doc_id", "chain"], sort=False)
    s = g.sum()
    out = pd.DataFrame({"full_name": p.groupby(["doc_id", "chain"], sort=False)["full_name"].first() if "full_name" in p
                        else None,
                        "portrayal_n": g.size().astype("int32"),
                        "portrayal_pertinent": (s["w"] / g.size()).astype("float32")})
    for k in DIMS + ["journaliste"]:
        out[f"portrayal_{k}"] = (s[k] / s["w"].where(s["w"] > 0)).astype("float32")
    r4 = lambda c: p[c].to_numpy().round(4)
    recs = pd.DataFrame({"kind": p["kind"].to_numpy(), "title": p["is_title"].to_numpy().astype(bool),
                         "quote": p["in_quote"].to_numpy().astype(bool), "pertinent": r4("p_pertinent"),
                         "journaliste": r4("p_journaliste"),
                         **{f"{k}_{s_}": r4(f"p_{k}_{s_}") for k in DIMS for s_ in ("neg", "pos")}})
    keys = list(zip(p["doc_id"], p["chain"]))
    lists: dict = {}
    for key, rec in zip(keys, recs.to_dict("records")):
        lists.setdefault(key, []).append(rec)
    out["portrayal"] = [lists[k] for k in out.index]
    return out.reset_index()


def characters(chars: pd.DataFrame, port: pd.DataFrame, dates: pd.Series, sources: pd.DataFrame | None = None,
               newspaper=None) -> pd.DataFrame:
    """sources: name_sources() output, the names (post-processing): every character gets its name from there, scored or
    not, and every scored chain keeps its scores, named or not. Without sources, the names the portrayal rows carry."""
    pc = per_character(port).rename(columns={"chain": "char_id"})
    pc["char_id"] = pc["char_id"].astype(chars["char_id"].dtype)
    names = pc[["doc_id", "char_id", "full_name"]].assign(name_source=None) if sources is None else \
        sources.rename(columns={"chain": "char_id"}).astype({"char_id": chars["char_id"].dtype})
    pc = pc.drop(columns=["full_name"])
    out = chars.merge(names, on=["doc_id", "char_id"], how="left", validate="one_to_one")
    out = out.merge(pc, on=["doc_id", "char_id"], how="left", validate="one_to_one")
    if len(out) != len(chars) or out["portrayal_n"].notna().sum() != len(pc) or out["portrayal_n"].sum() != len(port):
        raise ValueError("portrayal rows do not all match a character")
    out["portrayal_n"] = out["portrayal_n"].fillna(0).astype("int32")
    out["portrayal"] = [x if isinstance(x, list) else [] for x in out["portrayal"]]
    return with_meta(out, dates, newspaper)


def entities(ents: pd.DataFrame, dates: pd.Series, newspaper=None) -> pd.DataFrame:
    return with_meta(ents, dates, newspaper)
