"""Portrayal of people: every person chain of an article (PER coreference chain with a full proper name, « Prénom Nom »)
gets one segment per mention sentence (nlp.segments), scored by the portrayal ensemble (n-gramsci/portrayal-camembert-large).
Default: seeds 0 and 1 (gold kappa equal to the 5-seed ensemble, 2.5x cheaper; 1 seed loses 0.015-0.045).
With `targets` (names) or `targets_file` (one name per line) in the corpus params, only those people are scored: chains
with a proper-name mention matching the name, in articles naming them in full. Without, every person.
Output: one row per (article, chain, mention sentence) with the head probabilities and segment flags. The passages
themselves are article text and are not written. Trained on politicians: see the model card for the heads that
transfer to other people."""
import collections
import re

import pandas as pd

from ..nlp.segments import NAMEOK, build, name_re
from ..stage import Output, Stage, register
from ..textio import read_text

STOP = set("le la les l un une des du de au aux ce cet cette son sa ses".split())


def load(ctx):
    from ..portrayal import HUB, Scorer
    return Scorer.from_hub(ctx.p("portrayal", "model", HUB), seeds=ctx.p("portrayal", "seeds", [0, 1]))


def full_name(surfaces):
    props = [t for t in surfaces if NAMEOK.match(t) and t.split()[0].lower().strip("'’") not in STOP]
    if not props:
        return None
    return max(collections.Counter(props).items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def people(text, ents, min_mentions=1):
    """[(chain id, full name, [(onset, offset)])] for the PER chains of one article."""
    out = []
    per = ents[(ents["cat"] == "PER") & ents["COREF"].notna()]
    for ch, g in per.groupby("COREF"):
        if len(g) < min_mentions:
            continue
        surf = [text[int(a):int(b)].replace("///", "").strip() for a, b, p in zip(g["onset"], g["offset"], g["prop"]) if p == "PROP"]
        name = full_name(surf)
        if name:
            out.append((int(ch), name, [(int(a), int(b)) for a, b in zip(g["onset"], g["offset"])]))
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


def run(k, inp, out, state, ctx):
    docs = read_text(inp["text"]).set_index("doc_id")
    m = pd.read_parquet(inp["mentions"], columns=["doc_id", "start_token", "end_token", "onset", "offset"])
    e = pd.read_parquet(inp["entities"], columns=["doc_id", "start_token", "end_token", "cat", "prop", "COREF"])
    e["COREF"] = pd.to_numeric(e["COREF"], errors="coerce")
    for c in ("start_token", "end_token"):
        e[c] = pd.to_numeric(e[c], errors="coerce"); m[c] = m[c].astype(int)
    e = e.merge(m, on=["doc_id", "start_token", "end_token"])
    rows, meta = [], []
    targets = load_targets(ctx)                       # none -> every person; else only these people
    for d, g in e.groupby("doc_id", sort=False):
        if d not in docs.index:
            continue
        text = docs.at[d, "text"]
        found = targeted(text, g, targets) if targets else people(text, g, ctx.p("portrayal", "min_mentions", 1))
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
    return {"docs": int(df["doc_id"].nunique()) if len(df) else 0, "people": int(df.groupby(["doc_id", "chain"]).ngroups) if len(df) else 0,
            "segments": len(df)}


register(Stage("portrayal", inputs=["text", "mentions", "entities"], outputs=[Output("portrayal", "portrayal/shard_{k:05d}.parquet")],
               env="portrayal", gpu=True, ttl_hours=3, load=load, run=run, doc=__doc__))
