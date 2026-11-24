"""Portrayal of people: person chains of an article get one segment per mention sentence (nlp.segments), scored by the
portrayal ensemble (n-gramsci/portrayal-camembert-large).
score_all (the default from 2026-11-23 on): every singular PER chain is scored, named or not, with a name-free input (the
chain's own proper-name mentions become the model's placeholder); naming is post-processing (people(), with_source),
written alongside as full_name / name_source and recomputable at release time without scoring again. Without score_all
(the 2026-11-22 runs): only chains people() names are scored, with that name as the target.
Default: seeds 0 and 1 (gold kappa equal to the 5-seed ensemble, 2.5x cheaper; 1 seed loses 0.015-0.045).
With `targets` (names) or `targets_file` (one name per line) in the corpus params, only those people are scored: chains
with a proper-name mention matching the name, in articles naming them in full. Without, every person.
Output: one row per (article, chain, mention sentence) with the head probabilities, segment flags, sent (the centre
sentence, 0 = title) and mentions (the chain mentions the passage scores, [start_token, end_token]; from 2026-11-23 on). The passages
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


SURNAME_RE = re.compile(r"^(?:(?:le|la|de|du|des|d['’])\s*)?[A-ZÀ-Ý][\w'’\-]+$", re.I)
TITLES = {"M.", "M", "MM.", "Mme", "Mmes", "Mlle", "Madame", "Mademoiselle", "Monsieur", "Messieurs", "Me", "Maître", "Mgr",
          "Dr", "Docteur", "Pr"}
FEMININE = {"Mme", "Mmes", "Mlle", "Madame", "Mademoiselle"}


def strip_title(t: str) -> tuple[str, str | None]:
    """« Mme Simone Veil » -> (« Simone Veil », « Mme »): titles are never part of a name."""
    w, title = t.split(), None
    while w and w[0] in TITLES:
        title = title or w[0]; w = w[1:]
    return " ".join(w), title


def full_name(surfaces, compounds=frozenset()):
    """The chain's most frequent « First Last » mention, titles removed. A compound family name (« Giscard d'Estaing »,
    « Mendès France »: Surnames.compounds) is not « First Last »."""
    from ..explorer import name_key
    props = []
    for t in surfaces:
        t, _ = strip_title(t)
        if " " in t and NAMEOK.match(t) and t.split()[0].lower().strip("'’") not in STOP and name_key(t) not in compounds:
            props.append(t)
    if not props:
        return None
    return max(collections.Counter(props).items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def surname_of(name: str) -> str:
    """« Jean-Luc Mélenchon » -> « Mélenchon », « Marine Le Pen » -> « Le Pen » (everything after the first name)."""
    return name.split(" ", 1)[1] if " " in name else name


def bare_surname(surfaces, compounds=frozenset(), with_title=False):
    """Most frequent family name of a chain without a full name: « Mélenchon », « M. Mélenchon », « Le Pen », and compound
    family names (« M. Giscard d'Estaing »); also a single name (« Lula », « Madonna »). with_title: (name, True if it only
    ever came with a feminine title, True if it ever came with a title)."""
    from ..explorer import name_key
    c, fem = collections.Counter(), collections.defaultdict(set)
    for t in surfaces:
        t, title = strip_title(t)
        if not t or t.split()[0].lower() in ("son", "sa", "ses", "ce", "cette"):
            continue
        if name_key(t) in compounds or (t.split()[-1][0].isupper() and SURNAME_RE.match(t)):
            c[t] += 1; fem[t].add(title)
    if not c:
        return (None, False, False) if with_title else None
    sn = c.most_common(1)[0][0]
    if not with_title:
        return sn
    return sn, all(t in FEMININE for t in fem[sn]), any(t is not None for t in fem[sn])


NAME_PARTICLES = set("de du des d da das do dos del della di von van der den ter le la ben bin ibn el al y".split())


def name_keys(name: str) -> set:
    """Keys a bare word can stand for in a full name: the family name (everything after the first name) and each later
    word alone. « Luiz Inácio Lula da Silva »: inacio lula da silva, inacio, lula, silva; « Bachar al-Assad »: al assad,
    assad; « Marine Le Pen »: le pen, pen."""
    from ..explorer import name_key
    keys = {name_key(surname_of(name))}
    for w in name.split()[1:]:
        k = name_key(w)
        if not k or k in NAME_PARTICLES:
            continue
        keys.add(k)
        p = k.split()
        if len(p) > 1 and p[0] in NAME_PARTICLES:
            keys.add(" ".join(p[1:]))
    keys.discard("")
    return keys


def _chains(text, ents, compounds, min_mentions=1):
    """Named chains [(chain, « First Last », spans)] and bare ones [(chain, family or single name, spans, fem, titled)]."""
    named, bare = [], []
    per = ents[(ents["cat"] == "PER") & ents["COREF"].notna()]
    for ch, g in per.groupby("COREF"):
        if len(g) < min_mentions:
            continue
        # a mention running across the title/body separator glues two names (« SarkozyNicolas Sarkozy »): not a name
        surf = [text[int(a):int(b)].strip() for a, b, p in zip(g["onset"], g["offset"], g["prop"]) if p == "PROP"]
        surf = [t for t in surf if "///" not in t]
        spans = [(int(a), int(b)) for a, b in zip(g["onset"], g["offset"])]
        name = full_name(surf, compounds)
        if name:
            named.append((int(ch), name, spans))
        else:
            sn, fem, titled = bare_surname(surf, compounds, with_title=True)
            if sn:
                bare.append((int(ch), sn, spans, fem, titled))
    return named, bare


def people(text, ents, min_mentions=1, surnames=None, year=None, only_surname=False, with_source=False):
    """[(chain id, full name, [(onset, offset)])] for the PER chains of one article.
    A chain with no « First Last » mention but a bare name (« Mélenchon », « M. Mélenchon », « Mme Veil », « Giscard
    d'Estaing », « Lula ») is named, in this order, by:
      article      the article's one full name holding that word: its family name (« Jean-Luc Mélenchon »), or a later word
                   that is not a first name (« Lula » -> « Luiz Inácio Lula da Silva »; never « Luis » -> « José Luis
                   Rodríguez Zapatero »). A word also used as a first name (« Philippe ») takes « Édouard Philippe » only
                   if the chain has a title (« M. Philippe ») or nobody in the article has that first name;
      alias        the hand alias table (data/aliases.json, « Lula » -> « Luiz Inácio Lula da Silva »);
      family_name  the person `surnames` (Surnames) gives for that word and year: a family name, a later word of a full
                   name, or a single name nobody has a first name for (« Madonna »). Names that only come with a feminine
                   title are left out (« Mme Chirac » is not the year's dominant Chirac).
    Titles are never part of a name (« Mme Simone Veil » -> Simone Veil); full names are mapped through the aliases
    (« Lula da Silva » -> « Luiz Inácio Lula da Silva »).
    only_surname: return only the chains named from a bare name (to score them on top of an earlier run).
    with_source: 4-tuples, + where the name comes from: "chain" (the chain's own « First Last »), "article", "alias",
    "family_name"."""
    from ..explorer import name_key
    compounds = getattr(surnames, "compounds", frozenset())
    canon = surnames.canon if surnames is not None else (lambda n: n)
    named, bare = _chains(text, ents, compounds, min_mentions)
    named = [(ch, canon(n), sp, "chain") for ch, n, sp in named]
    firsts = {name_key(n.split()[0]) for _, n, _, _ in named}             # first names of this article's people
    common_first = getattr(surnames, "firstnames", frozenset()) | firsts
    found = []
    for ch, sn, spans, fem, titled in bare:
        k = name_key(sn)
        fam = {n for _, n, _, _ in named if name_key(surname_of(n)) == k}
        inner = {n for _, n, _, _ in named if k in name_keys(n)} - fam if k not in common_first else set()
        here = fam | inner
        if k in firsts and not titled:
            here = set()                                                   # « Philippe » next to « Philippe Martinez »
        if len(here) == 1:
            found.append((ch, here.pop(), spans, "article"))
        elif here:
            continue
        elif not fem and surnames is not None and k in surnames.aliases:
            found.append((ch, surnames.aliases[k], spans, "alias"))
        elif not fem and surnames is not None and (name := surnames.get(k, year)):
            found.append((ch, name, spans, "family_name"))
    out = found if only_surname else named + found
    return out if with_source else [x[:3] for x in out]


def name_rows(text, ents, compounds=frozenset()):
    """What the family-name table counts in one article: {(name, kind)}, kind "full" (a chain's « First Last »), "bare"
    (a chain's bare name, not only with a feminine title) or "other" (a one-word proper name the NER tags as something
    else than a person: « Carrefour », « Lille »; single names must be mostly people)."""
    named, bare = _chains(text, ents, compounds)
    out = {(n, "full") for _, n, _ in named} | {(sn, "bare") for _, sn, _, fem, _ in bare if not fem}
    o = ents[(ents["cat"] != "PER") & (ents["prop"] == "PROP")]
    for a, b in zip(o["onset"], o["offset"]):                  # every capitalised word: « le groupe Renault »
        for t in text[int(a):int(b)].replace("///", " ").split():
            t = t.strip(".,;:!?()«»\"")
            if t and t[0].isupper() and SURNAME_RE.match(t):
                out.add((t, "other"))
    return out


# words found before a family name that are not first names (« Général de Gaulle », « Vive de Gaulle », « Lycée Jaurès »)
NOT_FIRST = set("general president presidente vive marechal amiral colonel commandant capitaine lieutenant ministre senateur "
                "depute maire professeur pere abbe cardinal saint sainte mines lycee college rue place avenue boulevard quai "
                "aeroport fondation institut centre hopital ecole universite pont stade gare "
                # names the NER takes for people (« Debout la France », « Downing Street »…), found in the table 2026-11-22
                "debout downing scotland boko bottega correspondant correspondante nuit".split())
# single words the NER may tag as people that are never someone's name on their own (mononym route)
NOT_PERSON = set("dieu allah seigneur marianne matignon elysee bercy beauvau quai washington moscou pekin londres berlin "
                 "bruxelles rome madrid kiev kyiv washington europe afrique amerique twitter facebook google".split())


def load_aliases(path) -> dict:
    """data/aliases.json {canonical name: [other names]} -> {name key: canonical name} (the canonical's own key too)."""
    import json
    from ..explorer import name_key
    out = {}
    for canon, others in json.load(open(path, encoding="utf-8")).items():
        for n in [canon, *others]:
            out[name_key(n)] = canon
    return out


class Surnames:
    """Bare name -> the one person it stands for in a given year: the full name holding >= `share` of the articles
    naming someone « ... <word> ... » with that word after the first name (its family name or a later word: « Lula » in
    « Luiz Inácio Lula da Silva ») in year-1..year+1, with >= `min_n` such articles; or the word itself (« Madonna ») when
    it names >= `min_n` bare chains and full names holding it are < `mono_ratio` as many. `exclude`: words never resolved
    (Le Pen: Marine, Jean-Marie, Marion; places the NER may take for people). A word also used as a first name (in >= 10%
    as many articles: Philippe, Bertrand, Laurent) is never resolved either: a bare « Philippe » is usually someone's first
    name. `firstnames`: words used as a first name in >= `min_n` articles (people() never takes them as a later word).
    `aliases` {name key: canonical name}: one person's spellings (« Lula da Silva », « Luiz Inácio Lula »), applied to the
    counts and to every name people() gives. Built from counts {(year, name, kind): n articles}."""
    def __init__(self, table: dict, compounds=(), firstnames=(), aliases=None):
        self.t = table                                   # {key: {year: name}}
        self.compounds = frozenset(compounds)            # keys of compound family names (« giscard d estaing »)
        self.firstnames = frozenset(firstnames)
        self.aliases = dict(aliases or {})

    def get(self, key, year):
        return self.t.get(key, {}).get(int(year)) if year else None

    def canon(self, name):
        from ..explorer import name_key
        return self.aliases.get(name_key(name), name)

    @classmethod
    def build(cls, counts: pd.DataFrame, share=0.9, min_n=20, window=1, mono_ratio=0.1, min_person=0.8, min_full=3,
              aliases=None,
              exclude=("Le Pen", "Marseille", "Paris", "Lyon", "Toulouse", "Nice", "Bordeaux", "France")):
        """counts: columns year, full_name, n, kind ("full": a « First Last », the default; "bare": a chain's bare name,
        name_rows(); "other": the word tagged as something else than a person). A single name must be a person in
        >= `min_person` of its articles (bare / (bare + other)); when a few full names hold it (>= `min_full` articles,
        one holding >= `share`), it is that full name (« Hitler » -> « Adolf Hitler »). Compound family names « A B » are found first: « A » is (almost) never a first name, i.e. >= `share`
        of the full names starting with « A » are « A B », and « First A B » exists (>= `min_n` articles each):
        « Giscard d'Estaing », « Mendès France », « Couve de Murville »."""
        from ..explorer import name_key
        aliases = dict(aliases or {})
        canon = lambda n: aliases.get(name_key(n), n)
        if "kind" not in counts.columns:
            counts = counts.assign(kind="full")
        bare = counts[counts["kind"] == "bare"]
        other = counts[counts["kind"] == "other"]
        counts = counts[counts["kind"] == "full"]
        counts = counts.assign(full_name=counts["full_name"].map(lambda n: canon(strip_title(n)[0]))).query("full_name != ''")
        tot = counts.assign(nk=counts["full_name"].map(name_key)).groupby("nk")["n"].sum()
        first = counts.assign(f=counts["full_name"].map(lambda n: name_key(n.split()[0]))).groupby("f")["n"].sum()
        compounds = set()
        for nk, n in tot.items():
            w = nk.split()
            if len(w) >= 3 and n >= min_n:
                r = " ".join(w[1:])
                if tot.get(r, 0) >= min_n and tot.get(r, 0) >= share * first.get(w[1], 0):
                    compounds.add(r)
        # a compound written alone (« Giscard d'Estaing ») is a family name, not a full name: out of the counts
        counts = counts[~counts["full_name"].map(name_key).isin(compounds)]
        ex = {name_key(x) for x in exclude}
        counts = counts[~counts["full_name"].map(lambda n: name_key(n.split()[0]) in NOT_FIRST)]
        first = counts.assign(f=counts["full_name"].map(lambda n: name_key(n.split()[0]))).groupby("f")["n"].sum()
        firstnames = {k for k, v in first.items() if v >= min_n}
        counts = counts.assign(key=counts["full_name"].map(lambda n: sorted(name_keys(n)))).explode("key")
        bare = bare.assign(name=bare["full_name"].map(lambda n: strip_title(n)[0])).query("name != ''")
        bare = bare.assign(key=bare["name"].map(name_key), name=bare["name"].map(canon))
        fam = counts.groupby("key")["n"].sum().add(bare.groupby("key")["n"].sum(), fill_value=0)
        ex |= {k for k, v in fam.items() if first.get(k, 0) >= 0.1 * v}
        c = counts[~counts["key"].isin(ex)]
        c = c.assign(nk=c["full_name"].map(name_key))
        disp = c.groupby(["nk", "full_name"])["n"].sum().sort_values().groupby(level=0).tail(1)
        disp = pd.Series(disp.index.get_level_values(1), index=disp.index.get_level_values(0))

        def windowed(g, cols):
            parts = [g.assign(year=g["year"] + d) for d in range(-window, window + 1)]
            return pd.concat(parts).groupby(cols)["n"].sum().reset_index()

        w = windowed(c.groupby(["key", "year", "nk"])["n"].sum().reset_index(), ["key", "year", "nk"])
        tot = w.groupby(["key", "year"])["n"].transform("sum")
        top = w[w["n"] == w.groupby(["key", "year"])["n"].transform("max")].assign(tot=tot)
        top = top[(top["n"] >= share * top["tot"]) & (top["tot"] >= min_n)].drop_duplicates(["key", "year"])
        table: dict = {}
        for r in top.itertuples():
            table.setdefault(r.key, {})[int(r.year)] = disp[r.nk]
        # single names (« Madonna »): many bare chains, (almost) no full name holding the word
        b = bare[~bare["key"].isin(ex | NOT_PERSON | set(compounds)) & ~bare["key"].map(lambda k: k in NOT_FIRST).astype(bool)]
        b = b[~b["name"].map(lambda n: n.isupper() or any(ch.isdigit() for ch in n)).astype(bool)]   # RFI, M6: acronyms
        if len(other):
            on = other.assign(key=other["full_name"].map(lambda n: name_key(strip_title(n)[0]))).groupby("key")["n"].sum()
            bn = b.groupby("key")["n"].sum()
            person = bn / bn.add(on.reindex(bn.index).fillna(0))
            b = b[b["key"].map(person).fillna(1) >= min_person]
        if len(b):
            bdisp = b.groupby(["key", "name"])["n"].sum().sort_values().groupby(level=0).tail(1)
            bdisp = dict(zip(bdisp.index.get_level_values(0), bdisp.index.get_level_values(1)))
            bw = windowed(b.groupby(["key", "year"])["n"].sum().reset_index(), ["key", "year"])
            fw = windowed(counts.groupby(["key", "year"])["n"].sum().reset_index(), ["key", "year"])
            bw = bw.merge(fw, on=["key", "year"], how="left", suffixes=("", "_full")).fillna({"n_full": 0})
            bw = bw[(bw["n"] >= min_n) & (bw["n_full"] < mono_ratio * bw["n"])]
            # a few full names holding the word, one of them dominant: that person (« Adolf Hitler »)
            few = w.assign(tot=w.groupby(["key", "year"])["n"].transform("sum"))
            few = few[(few["n"] >= share * few["tot"]) & (few["n"] >= min_full)].drop_duplicates(["key", "year"])
            few = few[few["nk"].map(lambda nk: nk.split()[0] in firstnames)]      # « Adolf Hitler », not « L'action Carrefour »
            few = {(r.key, int(r.year)): disp[r.nk] for r in few.itertuples()}
            for r in bw.itertuples():
                table.setdefault(r.key, {}).setdefault(int(r.year), few.get((r.key, int(r.year)), bdisp[r.key]))
        return cls(table, compounds, firstnames, aliases)

    def to_json(self, path):
        import json
        json.dump({"table": self.t, "compounds": sorted(self.compounds), "firstnames": sorted(self.firstnames)},
                  open(path, "w"), ensure_ascii=False)

    @classmethod
    def from_json(cls, path, aliases="aliases.json"):
        """`aliases`: the alias file, relative to the table's folder (data/aliases.json), read if it exists."""
        import json
        d = json.load(open(path))
        if "table" not in d:                                   # 2026-11-22 format: the table alone
            d = {"table": d, "compounds": []}
        a = os.path.join(os.path.dirname(os.path.abspath(path)), aliases) if aliases else None
        return cls({k: {int(y): n for y, n in v.items()} for k, v in d["table"].items()}, d["compounds"],
                   d.get("firstnames", ()), load_aliases(a) if a and os.path.exists(a) else None)


def name_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows (doc_id, full_name, year[, kind]) -> {(year, name, kind): articles}, honorific full names out."""
    r = rows.assign(kind=rows["kind"] if "kind" in rows.columns else "full")
    r = r[["doc_id", "full_name", "year", "kind"]].dropna().drop_duplicates()
    r = r[(r["kind"] == "bare") | ~r["full_name"].str.split().str[0].isin(HONORIFIC)]
    return r.groupby(["year", "full_name", "kind"]).size().rename("n").reset_index()


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


def chain_re(surfaces) -> re.Pattern:
    """The chain's own proper-name mentions (any whitespace inside), titles optional: the name-free target of score_all.
    No proper-name mention: a pattern that never matches (mentions are kept as written)."""
    alts = set()
    for t in surfaces:
        t = t.replace("///", " ").strip()
        if t:
            alts.add(t); alts.add(strip_title(t)[0])
    alts = sorted((a for a in alts if a), key=len, reverse=True)
    if not alts:
        return re.compile(r"(?!)")
    return re.compile("|".join(r"\s+".join(map(re.escape, a.split())) for a in alts))


def plural(g) -> bool:
    """A chain is plural when it has more plural than singular mentions (propp's characters `number`)."""
    n = g["number"] if "number" in g else pd.Series(dtype=object)
    return int((n == "Plural").sum()) > int((n == "Singular").sum())


def load_inputs(text_path, mentions_path, entities_path, exclude_possessive=False):
    """(docs indexed by doc_id, entities with onset/offset): what run() and the family-name counts read."""
    docs = read_text(text_path).set_index("doc_id")
    m = pd.read_parquet(mentions_path, columns=["doc_id", "start_token", "end_token", "onset", "offset"])
    e = pd.read_parquet(entities_path, columns=["doc_id", "start_token", "end_token", "cat", "prop", "COREF", "POS_tag", "text", "number"])
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
        tok = {(int(a), int(b)): [int(x), int(y)] for a, b, x, y in zip(g["onset"], g["offset"], g["start_token"], g["end_token"])} \
            if "start_token" in g else {}
        year = str(docs.at[d, "date"])[:4] if "date" in docs.columns else None
        y = int(year) if year and year.isdigit() else None
        sn = getattr(state, "surnames", None)
        if ctx.p("portrayal", "score_all", False) and not targets:
            names = {c: (n, src) for c, n, _, src in people(text, g, ctx.p("portrayal", "min_mentions", 1), surnames=sn, year=y,
                                                           with_source=True)}
            per = g[(g["cat"] == "PER") & g["COREF"].notna()]
            for ch, h in per.groupby("COREF"):
                if len(h) < ctx.p("portrayal", "min_mentions", 1) or plural(h):
                    continue
                ch = int(ch); spans = [(int(a), int(b)) for a, b in zip(h["onset"], h["offset"])]
                rx = chain_re([text[int(a):int(b)] for a, b, p in zip(h["onset"], h["offset"], h["prop"]) if p == "PROP"])
                name, src = names.get(ch, (None, None))
                for s in build(text, rx, spans):
                    rows.append({"title": s["title"], "passage": s["passage"], "pattern": rx.pattern})
                    meta.append({"doc_id": d, "chain": ch, "full_name": name, "name_source": src, "n_mentions": len(spans),
                                 **{x: s[x] for x in ("mention", "is_title", "kind", "label_flag", "in_quote", "sent")},
                                 "mentions": [tok[x] for x in s["spans"] if x in tok]})
            continue
        found = targeted(text, g, targets) if targets else people(
            text, g, ctx.p("portrayal", "min_mentions", 1), surnames=sn, year=y,
            only_surname=ctx.p("portrayal", "only_surname", False))
        for ch, name, spans in found:
            for s in build(text, name_re(name), spans):
                rows.append({"title": s["title"], "passage": s["passage"], "full_name": name})
                meta.append({"doc_id": d, "chain": ch, "full_name": name, "n_mentions": len(spans),
                             **{x: s[x] for x in ("mention", "is_title", "kind", "label_flag", "in_quote", "sent")},
                             "mentions": [tok[x] for x in s["spans"] if x in tok]})
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
