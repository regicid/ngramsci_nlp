"""Portrayal of people: every person chain of an article (PER coreference chain with a full proper name, « Prénom Nom »)
gets one segment per mention sentence (nlp.segments), scored by the portrayal ensemble (n-gramsci/portrayal-camembert-large).
Default: seeds 0 and 1 (gold kappa equal to the 5-seed ensemble, 2.5x cheaper; 1 seed loses 0.015-0.045).
With `targets` (names) or `targets_file` (one name per line) in the corpus params, only those people are scored: chains
with a proper-name mention matching the name, in articles naming them in full. Without, every person.
Output: one row per (article, chain, mention sentence) with the head probabilities and segment flags. The passages
themselves are article text and are not written. Trained on politicians: see the model card for the heads that
transfer to other people."""
import collections
import os
import re

import pandas as pd

from ..nlp.segments import NAMEOK, build, name_re
from ..stage import Output, Stage, register
from ..textio import read_text

STOP = set("le la les l un une des du de au aux ce cet cette son sa ses".split())
HONORIFIC = {"Mme", "Mmes", "Mlle", "Madame", "Mademoiselle", "Monsieur", "Messieurs", "Me", "Maître", "Mgr", "Dr", "Docteur", "Pr"}
POSS = set("mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs".split())


def possessive(ents):
    """Possessive determiners of a chain ('son' in 'son discours'): with `exclude_possessive`, they neither make a
    segment nor get marked in one (the pronoun 'leur' in 'il leur parle' is not a determiner and stays)."""
    return ents["POS_tag"].eq("DET") & ents["text"].str.lower().str.strip().isin(POSS)


def load(ctx):
    from ..portrayal import HUB, Scorer
    sc = Scorer.from_hub(ctx.p("portrayal", "model", HUB), seeds=ctx.p("portrayal", "seeds", [0, 1]),
                         revision=ctx.p("portrayal", "revision"))
    f = ctx.p("portrayal", "surnames_file")                 # relative to the repo root
    if f:
        sc.surnames = Surnames.from_json(f if os.path.isabs(f) else os.path.join(os.path.dirname(__file__), "..", "..", "..", f))
    return sc


def full_name(surfaces):
    props = [t for t in surfaces if NAMEOK.match(t) and t.split()[0].lower().strip("'’") not in STOP]
    if not props:
        return None
    # « Ségolène Royal » over a more frequent « Mme Royal »: an honorific is not a first name (one person, one name)
    return max(collections.Counter(props).items(), key=lambda kv: (kv[0].split()[0] not in HONORIFIC, kv[1], len(kv[0])))[0]


SURNAME_RE = re.compile(r"^(?:(?:le|la|de|du|des|d['’])\s*)?[A-ZÀ-Ý][\w'’\-]+$", re.I)
TITLES = {"M.", "M", "MM.", "Mme", "Mmes", "Mlle", "Madame", "Monsieur", "Me", "Maître", "Mgr", "Dr", "Pr"}


def surname_of(name: str) -> str:
    """« Jean-Luc Mélenchon » -> « Mélenchon », « Marine Le Pen » -> « Le Pen » (everything after the first name)."""
    return name.split(" ", 1)[1] if " " in name else name


def bare_surname(surfaces):
    """Most frequent family name of a chain without a full name: « Mélenchon », « M. Mélenchon », « Le Pen »."""
    c = collections.Counter()
    for t in surfaces:
        w = t.split()
        if w and w[0] in TITLES:
            w = w[1:]
        t = " ".join(w)
        if t and t.split()[-1][0].isupper() and SURNAME_RE.match(t) and t.split()[0].lower() not in ("son", "sa", "ses", "ce", "cette"):
            c[t] += 1
    return c.most_common(1)[0][0] if c else None


def people(text, ents, min_mentions=1, surnames=None, year=None, only_surname=False, with_source=False):
    """[(chain id, full name, [(onset, offset)])] for the PER chains of one article.
    A chain with no « First Last » mention but a family name (« Mélenchon », « M. Mélenchon ») takes the article's one
    « <first> Mélenchon » if there is exactly one, else the person `surnames` (Surnames) gives for that name and year.
    only_surname: return only the chains named that way (to score them on top of an earlier run).
    with_source: 4-tuples, + where the name comes from: "chain" (the chain's own « First Last »), "article" (« Mme X » or a
    bare « X » resolved by the article's one « First X »), "family_name" (the person `surnames` gives for that year)."""
    named, bare = [], []
    per = ents[(ents["cat"] == "PER") & ents["COREF"].notna()]
    for ch, g in per.groupby("COREF"):
        if len(g) < min_mentions:
            continue
        # a mention running across the title/body separator glues two names (« SarkozyNicolas Sarkozy »): not a name
        surf = [text[int(a):int(b)].strip() for a, b, p in zip(g["onset"], g["offset"], g["prop"]) if p == "PROP"]
        surf = [t for t in surf if "///" not in t]
        spans = [(int(a), int(b)) for a, b in zip(g["onset"], g["offset"])]
        name = full_name(surf)
        if name:
            named.append((int(ch), name, spans))
        elif surnames is not None or only_surname:
            sn = bare_surname(surf)
            if sn:
                bare.append((int(ch), sn, spans))
    before = {ch: n for ch, n, _ in named}
    named = [(ch, n, sp, "chain" if before[ch] == n else "article") for ch, n, sp in resolve_honorifics(named)]
    from ..explorer import name_key
    found = []
    for ch, sn, spans in bare:
        k = name_key(sn)
        here = {n for _, n, _, _ in named if n.split()[0] not in HONORIFIC and name_key(surname_of(n)) == k}
        if len(here) == 1:
            found.append((ch, here.pop(), spans, "article"))
        elif not here and surnames is not None and (name := surnames.get(k, year)):
            found.append((ch, name, spans, "family_name"))
    out = found if only_surname else named + found
    return out if with_source else [x[:3] for x in out]


# words found before a family name that are not first names (« Général de Gaulle », « Vive de Gaulle », « Lycée Jaurès »)
NOT_FIRST = set("general president presidente vive marechal amiral colonel commandant capitaine lieutenant ministre senateur "
                "depute maire professeur pere abbe cardinal saint sainte mines lycee college rue place avenue boulevard quai "
                "aeroport fondation institut centre hopital ecole universite pont stade gare "
                # names the NER takes for people (« Debout la France », « Downing Street »…), found in the table 2026-11-22
                "debout downing scotland boko bottega correspondant correspondante nuit".split())


class Surnames:
    """Family name -> the one person it stands for in a given year: the first name holding >= `share` of the articles
    naming someone « <first> <family name> » in year-1..year+1, with >= `min_n` such articles. `exclude`: family names
    never resolved (Le Pen: Marine, Jean-Marie, Marion; places the NER may take for people). A family name that is also a first name (used as one in >= 10% as
    many articles: Philippe, Bertrand, Laurent) is never resolved either: a bare « Philippe » is usually someone's first
    name. Built from counts {(family key, year, full name): n articles}."""
    def __init__(self, table: dict):
        self.t = table                                   # {family key: {year: full name}}

    def get(self, key, year):
        return self.t.get(key, {}).get(int(year)) if year else None

    @classmethod
    def build(cls, counts: pd.DataFrame, share=0.9, min_n=20, window=1,
              exclude=("Le Pen", "Marseille", "Paris", "Lyon", "Toulouse", "Nice", "Bordeaux", "France")):
        """counts: columns key (family key), year, full_name, n."""
        from ..explorer import name_key
        ex = {name_key(x) for x in exclude}
        counts = counts[~counts["full_name"].map(lambda n: name_key(n.split()[0]) in NOT_FIRST)]
        first = counts.assign(f=counts["full_name"].map(lambda n: name_key(n.split()[0]))).groupby("f")["n"].sum()
        fam = counts.groupby("key")["n"].sum()
        ex |= {k for k, v in fam.items() if first.get(k, 0) >= 0.1 * v}
        c = counts[~counts["key"].isin(ex)]
        c = c.assign(nk=c["full_name"].map(name_key))
        disp = c.groupby("nk")["full_name"].agg(lambda s: s.value_counts().index[0])
        g = c.groupby(["key", "year", "nk"])["n"].sum().reset_index()
        parts = [g.assign(year=g["year"] + d) for d in range(-window, window + 1)]
        w = pd.concat(parts).groupby(["key", "year", "nk"])["n"].sum().reset_index()
        tot = w.groupby(["key", "year"])["n"].transform("sum")
        top = w[w["n"] == w.groupby(["key", "year"])["n"].transform("max")].assign(tot=tot)
        top = top[(top["n"] >= share * top["tot"]) & (top["tot"] >= min_n)].drop_duplicates(["key", "year"])
        table: dict = {}
        for r in top.itertuples():
            table.setdefault(r.key, {})[int(r.year)] = disp[r.nk]
        return cls(table)

    def to_json(self, path):
        import json
        json.dump(self.t, open(path, "w"), ensure_ascii=False)

    @classmethod
    def from_json(cls, path):
        import json
        return cls({k: {int(y): n for y, n in v.items()} for k, v in json.load(open(path)).items()})


def name_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """Portrayal-style rows (doc_id, full_name, year) -> {(family key, year, full name): articles}, honorific names out."""
    from ..explorer import name_key
    r = rows[["doc_id", "full_name", "year"]].drop_duplicates()
    r = r[~r["full_name"].str.split().str[0].isin(HONORIFIC)]
    r = r.assign(key=r["full_name"].map(lambda n: name_key(surname_of(n))))
    return r.groupby(["key", "year", "full_name"]).size().rename("n").reset_index()


def resolve_honorifics(found):
    """« Mme Royal » -> « Ségolène Royal » when the article names exactly one person « <first name> Royal » (in another
    chain): one person, one name across articles. Otherwise the honorific name is kept."""
    full = [n for _, n, _ in found if n.split()[0] not in HONORIFIC]
    out = []
    for ch, name, spans in found:
        first, *rest = name.split()
        if first in HONORIFIC and rest:
            cands = {n for n in full if n.endswith(" " + " ".join(rest)) and n != name}
            if len(cands) == 1:
                name = cands.pop()
        out.append((ch, name, spans))
    return out


def full_re(name):
    """The full name as written in an article ('Marine Le Pen', any whitespace): an article is kept for a target only if it
    names them in full once, as in the candidate study (a bare 'Le Pen' could be a relative)."""
    first, *rest = name.split()
    return re.compile(r"\s+".join([re.escape(first).replace("É", "[ÉE]")] + [re.escape(w) for w in rest]))


def targeted(text, ents, targets):
    """[(chain id or -1, target name, spans)] for the given people: every chain with a proper-name mention matching the
    target's name; the target's name matches are marked even when no chain was found (chain -1)."""
    per = ents[(ents["cat"] == "PER") & ents["COREF"].notna()]
    out = []
    for name, rx, frx in targets:
        if not frx.search(text):
            continue
        chains = {c for c, a, b, pr in zip(per["COREF"], per["onset"], per["offset"], per["prop"])
                  if pr == "PROP" and rx.fullmatch(text[int(a):int(b)].replace("///", "").strip())}
        spans = [(int(a), int(b)) for c, a, b in zip(per["COREF"], per["onset"], per["offset"]) if c in chains]
        out.append((int(min(chains)) if chains else -1, name, spans))
    return out


def load_targets(ctx):
    names = list(ctx.p("portrayal", "targets", []) or [])
    f = ctx.p("portrayal", "targets_file")
    if f:
        names += [l.strip() for l in open(f, encoding="utf-8") if l.strip() and not l.startswith("#")]
    return [(n, name_re(n), full_re(n)) for n in dict.fromkeys(names)]


def load_inputs(text_path, mentions_path, entities_path, exclude_possessive=False):
    """(docs indexed by doc_id, entities with onset/offset): what run() and the family-name counts read."""
    docs = read_text(text_path).set_index("doc_id")
    m = pd.read_parquet(mentions_path, columns=["doc_id", "start_token", "end_token", "onset", "offset"])
    e = pd.read_parquet(entities_path, columns=["doc_id", "start_token", "end_token", "cat", "prop", "COREF", "POS_tag", "text"])
    if exclude_possessive:
        e = e[~possessive(e)]
    e["COREF"] = pd.to_numeric(e["COREF"], errors="coerce")
    for c in ("start_token", "end_token"):
        e[c] = pd.to_numeric(e[c], errors="coerce"); m[c] = m[c].astype(int)
    return docs, e.merge(m, on=["doc_id", "start_token", "end_token"])


def run(k, inp, out, state, ctx):
    docs, e = load_inputs(inp["text"], inp["mentions"], inp["entities"], ctx.p("portrayal", "exclude_possessive", False))
    rows, meta = [], []
    targets = load_targets(ctx)                       # none -> every person; else only these people
    for d, g in e.groupby("doc_id", sort=False):
        if d not in docs.index:
            continue
        text = docs.at[d, "text"]
        year = str(docs.at[d, "date"])[:4] if "date" in docs.columns else None
        found = targeted(text, g, targets) if targets else people(
            text, g, ctx.p("portrayal", "min_mentions", 1), surnames=getattr(state, "surnames", None),
            year=int(year) if year and year.isdigit() else None, only_surname=ctx.p("portrayal", "only_surname", False))
        for ch, name, spans in found:
            for s in build(text, name_re(name), spans):
                rows.append({"title": s["title"], "passage": s["passage"], "full_name": name})
                meta.append({"doc_id": d, "chain": ch, "full_name": name, "n_mentions": len(spans),
                             **{x: s[x] for x in ("mention", "is_title", "kind", "label_flag", "in_quote")}})
    P = state.score(rows, batch=ctx.p("portrayal", "batch", 64)) if rows else []
    df = pd.DataFrame(meta)
    for h in ("pertinent", "journaliste"):
        df[f"p_{h}"] = [p[h] for p in P]
    for h in ("valence", "extremisme", "dynamique"):
        for j, c in enumerate(("neg", "zero", "pos")):
            df[f"p_{h}_{c}"] = [p[h][j] for p in P]
    df.to_parquet(out["portrayal"], index=False)
    return {"model": state.revision, "docs": int(df["doc_id"].nunique()) if len(df) else 0, "people": int(df.groupby(["doc_id", "chain"]).ngroups) if len(df) else 0,
            "segments": len(df)}


register(Stage("portrayal", inputs=["text", "mentions", "entities"], outputs=[Output("portrayal", "portrayal/shard_{k:05d}.parquet")],
               env="portrayal", gpu=True, ttl_hours=3, load=load, run=run, doc=__doc__))
