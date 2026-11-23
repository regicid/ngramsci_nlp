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
  portrayal               (passages=True only, the 2026-11-22 releases) the passages in text order: {kind, title, quote,
                          pertinent, journaliste, valence_neg, valence_pos, extremisme_neg, ...} (P(0) = 1 - neg - pos)
entities, with port (from presse3 on): each mention a scored passage marks gets that passage's scores (mention_scores):
  portrayal_sent          the passage's centre sentence (0 = title; mentions of one sentence share one passage: same
                          scores, one portrayal_sent; collapse on (doc_id, COREF, portrayal_sent) to count passages)
  portrayal_kind, portrayal_title, portrayal_quote   name / coref / coref+name; the passage is the title; inside a quote
  p_pertinent, p_journaliste, p_valence_neg, p_valence_pos, p_extremisme_neg, ..., p_dynamique_pos
  Null: the mention is in no scored passage (possessive determiner, chain not scored, not PER, past the target's 700
  characters) or its passage could not be rebuilt.
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


def locate(port: pd.DataFrame, docs: pd.DataFrame, ents: pd.DataFrame, mode: str = "name") -> pd.DataFrame:
    """port + sent / mentions per passage, by rebuilding the passages the scorer cut (nlp.segments.build, deterministic;
    unchanged since the first run). docs, ents: portrayal.load_inputs() with the run's exclude_possessive. mode "name":
    the target regex is name_re(full_name of the rows) (the 2026-11-22 runs: portrayal/, portrayal_surnames/); "chain":
    chain_re of the chain's own proper-name mentions (score_all: portrayal_all/). Rows whose rebuilt passages do not match
    (count, kind, title, quote) get nulls; locate.mismatch counts the chains."""
    from .nlp.segments import build, name_re
    from .stages.portrayal import chain_re
    if "sent" in port and "mentions" in port:                 # written by the scorer itself
        return port
    per = ents[(ents["cat"] == "PER") & ents["COREF"].notna()]
    tok = {(d, int(a), int(b)): [int(s_), int(e_)] for d, a, b, s_, e_ in
           zip(per["doc_id"], per["onset"], per["offset"], per["start_token"], per["end_token"])}
    chains = {k: g for k, g in per.groupby(["doc_id", "COREF"], sort=False)}
    sent, ments, bad = [None] * len(port), [None] * len(port), 0
    pos = {i: j for j, i in enumerate(port.index)}
    for (d, ch), rows in port.groupby(["doc_id", "chain"], sort=False):
        g = chains.get((d, float(ch))) if (d, float(ch)) in chains else chains.get((d, ch))
        if g is None or d not in docs.index:
            bad += 1; continue
        text = docs.at[d, "text"]
        spans = [(int(a), int(b)) for a, b in zip(g["onset"], g["offset"])]
        if mode == "chain":
            rx = chain_re([text[int(a):int(b)] for a, b, p in zip(g["onset"], g["offset"], g["prop"]) if p == "PROP"])
        else:
            rx = name_re(rows["full_name"].iloc[0])
        segs = {s["mention"]: s for s in build(text, rx, spans)}
        ok = len(segs) == len(rows) and all(
            (m := segs.get(int(r.mention))) is not None and m["kind"] == r.kind and m["is_title"] == int(r.is_title)
            and m["in_quote"] == int(r.in_quote) for r in rows.itertuples())
        if not ok:
            bad += 1; continue
        for r in rows.itertuples():
            m = segs[int(r.mention)]; i = pos[r.Index]
            sent[i] = int(m["sent"]); ments[i] = [tok[(d, x, y)] for x, y in m["spans"] if (d, x, y) in tok]
    locate.mismatch = bad
    return port.assign(sent=pd.array(sent, dtype="Int32"), mentions=ments)


def per_character(port: pd.DataFrame, passages: bool = True) -> pd.DataFrame:
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
    if not passages:
        return out.reset_index()
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
               newspaper=None, passages: bool = True) -> pd.DataFrame:
    """sources: name_sources() output, the names (post-processing): every character gets its name from there, scored or
    not, and every scored chain keeps its scores, named or not. Without sources, the names the portrayal rows carry."""
    pc = per_character(port, passages).rename(columns={"chain": "char_id"})
    pc["char_id"] = pc["char_id"].astype(chars["char_id"].dtype)
    names = pc[["doc_id", "char_id", "full_name"]].assign(name_source=None) if sources is None else \
        sources.rename(columns={"chain": "char_id"}).astype({"char_id": chars["char_id"].dtype})
    pc = pc.drop(columns=["full_name"])
    out = chars.merge(names, on=["doc_id", "char_id"], how="left", validate="one_to_one")
    out = out.merge(pc, on=["doc_id", "char_id"], how="left", validate="one_to_one")
    if len(out) != len(chars) or out["portrayal_n"].notna().sum() != len(pc) or out["portrayal_n"].sum() != len(port):
        raise ValueError("portrayal rows do not all match a character")
    out["portrayal_n"] = out["portrayal_n"].fillna(0).astype("int32")
    if passages:
        out["portrayal"] = [x if isinstance(x, list) else [] for x in out["portrayal"]]
    return with_meta(out, dates, newspaper)


SCORE_COLS = ["p_pertinent", "p_journaliste"] + [f"p_{k}_{s}" for k in DIMS for s in ("neg", "pos")]


def mention_scores(port: pd.DataFrame) -> pd.DataFrame:
    """One row per (passage, chain mention it marks): doc_id, start_token, end_token, COREF and the passage's columns."""
    p = port[port["sent"].notna()]
    rows = [(d, int(t[0]), int(t[1]), int(ch), i) for i, (d, ch, ms) in enumerate(zip(p["doc_id"], p["chain"], p["mentions"]))
            for t in ms]
    m = pd.DataFrame(rows, columns=["doc_id", "start_token", "end_token", "COREF", "_i"])
    q = p.reset_index(drop=True)
    cols = {"portrayal_sent": q["sent"].astype("Int32"), "portrayal_kind": q["kind"],
            "portrayal_title": q["is_title"].astype(bool), "portrayal_quote": q["in_quote"].astype(bool),
            **{c: q[c].astype("float32") for c in SCORE_COLS}}
    return pd.concat([m.drop(columns="_i")] + [pd.Series(v.to_numpy()[m["_i"].to_numpy()], name=k) for k, v in cols.items()], axis=1)


def entities(ents: pd.DataFrame, dates: pd.Series, newspaper=None, port: pd.DataFrame | None = None) -> pd.DataFrame:
    """port (located: release.locate): every marked mention gets its passage's scores; each one must match a row."""
    if port is not None:
        ms = mention_scores(port)
        key = ["doc_id", "start_token", "end_token", "COREF"]
        ms = ms.astype({"start_token": ents["start_token"].dtype, "end_token": ents["end_token"].dtype, "COREF": ents["COREF"].dtype})
        if ms.duplicated(key).any():
            raise ValueError(f"{int(ms.duplicated(key).sum())} mentions in two passages")
        out = ents.merge(ms, on=key, how="left", validate="many_to_one")
        hit = out[["doc_id", "start_token", "end_token", "COREF"]][out["p_pertinent"].notna()].drop_duplicates()
        if len(out) != len(ents) or len(hit) != len(ms):
            raise ValueError(f"{len(ms) - len(hit)} scored mentions match no entities row")
        ents = out
    return with_meta(ents, dates, newspaper)
