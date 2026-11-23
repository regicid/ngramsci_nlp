"""Quote links: each REBEL quote's speaker -> the entity mention it names -> its Qwen coreference chain -> gender.

Post-processing of `quotes` with the chains of `assemble` (the matching of our earlier quote analyses, on the new coref):
  1. the quote is found in the article (its first 60/30/15 characters, case, apostrophes and spacing normalised),
     searched first in the 1,500-character unit REBEL read;
  2. the speaker string (`source`) is found as whole words, the occurrence nearest the quote and outside it, in the
     same unit (names may also be found elsewhere in the article: `link` = "article");
  3. the entity mention that covers it best (character-span Jaccard; ties: PER with a chain) is the speaker;
  4. its chain gives chain_gender / chain_score / chain_conflict (the characters table: our chain-level gender),
     and gender.gendering.genderize(source, chain...) the speaker's gender: the source's own cues (first name,
     honorific, job) first, the chain only for cue-less sources (pronouns, surnames alone).
One row per quote, in the order of the quotes shard, with the quote columns plus the link and gender columns.
CPU only, no model; needs the article text, so it runs on the site that holds the shard."""
import re

import pandas as pd

from ..stage import Output, Stage, register
from ..textio import read_text

APOS, QUOTES = "’ʼ‘`´", "«»“”„"
PRONOUNS = {"il", "elle", "ils", "elles", "on", "je", "j'", "nous", "vous", "lui", "celui-ci", "celle-ci"}
COLS = ["link", "q_start", "q_end", "src_start", "src_end", "mention", "mention_cat", "chain", "chain_size",
        "chain_name", "chain_number", "chain_gender", "chain_score", "chain_conflict",
        "gender_firm", "gender_firm_basis", "gender_inclusive", "gender_pronoun_det", "gender_conflict", "gender_score"]


def norm(s):
    """(normalised text, index of each of its characters in s): lower case, one apostrophe and one quotation mark,
    whitespace runs (and the title separator ///) as one space, no space after an apostrophe."""
    s = s.replace("///", "   ")
    out, idx = [], []
    for i, c in enumerate(s):
        c = "'" if c in APOS else '"' if c in QUOTES else " " if c.isspace() else (c.lower()[:1] or c)
        if c == " " and (not out or out[-1] in " '"):
            continue
        out.append(c); idx.append(i)
    return "".join(out), idx


def word_re(source):
    """Whole-word pattern of a normalised source, spaces optional (detokenisation differences)."""
    toks = norm(source)[0].split()
    return re.compile(r"(?<!\w)" + " ?".join(map(re.escape, toks)) + r"(?!\w)") if toks else None


def unit_starts(nt, text):
    """Start (in the normalised text) of each unit REBEL read, found in order."""
    from .quotes import units
    starts, cur = [], 0
    for _, u in units(text):
        head = norm(u)[0][:40].strip()
        p = nt.find(head, cur) if head else -1
        starts.append(p if p >= 0 else cur); cur = max(cur, starts[-1])
    return starts


def find_quote(nt, content, lo, hi):
    c = norm(content)[0].strip(' "') if isinstance(content, str) else ""
    for L in (60, 30, 15):
        h = c[:L]
        if not h:
            break
        p = nt.find(h, lo, hi + len(h))
        if p < 0:
            p = nt.find(h)
        if p >= 0:
            return p, min(len(nt), p + len(c))
    return -1, -1


def find_source(nt, source, qs, qe, lo, hi):
    """(start, end, where) of the occurrence of the source nearest the quote, outside it."""
    rx = word_re(source) if isinstance(source, str) else None
    if rx is None:
        return -1, -1, ""
    pron = norm(source)[0].strip() in PRONOUNS

    def best(a, b):
        ms = [(m.start(), m.end()) for m in rx.finditer(nt, a, b)]
        if qs >= 0:
            ms = [(s, e) for s, e in ms if e <= qs or s >= qe]
            ms.sort(key=lambda m: qs - m[1] if m[1] <= qs else m[0] - qe)
        return ms[0] if ms else None

    m = best(lo, hi)
    if m:
        return m[0], m[1], "unit"
    if not pron and (m := best(0, len(nt))):
        return m[0], m[1], "article"
    return -1, -1, ""


def link_doc(text, quotes, ents, chars):
    """Linking for one article. quotes: its rows of the quotes shard (unit, content, source); ents: its entities with
    onset/offset (character offsets in text), cat, prop, COREF; chars: its characters rows (char_id = COREF).
    Returns one dict per quote (COLS without the gender_* columns)."""
    nt, idx = norm(text)
    starts = unit_starts(nt, text) + [len(nt)]
    def to_orig(p, end=False):                       # normalised position -> character offset in text
        if end:
            return idx[p - 1] + 1 if 0 < p <= len(idx) else -1
        return idx[p] if 0 <= p < len(idx) else -1
    ch = chars.set_index("char_id") if len(chars) else None
    on, off = ents["onset"].to_numpy(), ents["offset"].to_numpy()
    rows = []
    for u, content, source in zip(quotes["unit"], quotes["content"], quotes["source"]):
        u = int(u) if pd.notna(u) and int(u) < len(starts) - 1 else 0
        lo, hi = starts[u], starts[u + 1] + 200
        qs, qe = find_quote(nt, content, lo, hi)
        ss, se, where = find_source(nt, source, qs, qe, lo, hi)
        r = dict.fromkeys(COLS); r.update(link=where, q_start=to_orig(qs), q_end=to_orig(qe, True),
                                          src_start=to_orig(ss), src_end=to_orig(se, True))
        if ss >= 0 and len(ents):
            a, b = r["src_start"], r["src_end"]
            inter = (pd.Series(off).clip(upper=b) - pd.Series(on).clip(lower=a)).clip(lower=0).to_numpy()
            jac = inter / ((off - on) + (b - a) - inter)
            if jac.max() > 0:
                cand = ents.assign(_j=jac)[jac == jac.max()]
                cand = cand.assign(_k=(cand["cat"] == "PER").astype(int) + cand["COREF"].notna().astype(int))
                e = cand.sort_values(["_k", "onset"], ascending=[False, True]).iloc[0]
                r.update(mention=text[int(e["onset"]):int(e["offset"])], mention_cat=e["cat"])
                if pd.notna(e["COREF"]):
                    c = int(e["COREF"]); g = ents[ents["COREF"] == c]
                    props = [text[int(x):int(y)].replace("///", " ").strip() for x, y, p in zip(g["onset"], g["offset"], g["prop"]) if p == "PROP"]
                    r.update(chain=c, chain_size=len(g), chain_name=max(props, key=len) if props else None)
                    if ch is not None and c in ch.index:
                        x = ch.loc[c]
                        r.update(chain_number=x.get("number"), chain_gender=x.get("gender_firm"),
                                 chain_score=x.get("gender_score"), chain_conflict=x.get("gender_conflict"))
        rows.append(r)
    return rows


def load(ctx):
    from ..gender import gendering
    from ..gender.regender_book import JOBS_CSV, NAMES_CSV
    gendering.init_resources(NAMES_CSV, JOBS_CSV)
    return gendering


def run(k, inp, out, state, ctx):
    from .portrayal import load_inputs
    gendering = state
    docs, ents = load_inputs(inp["text"], inp["mentions"], inp["entities"])
    q = pd.read_parquet(inp["quotes"]); q["doc_id"] = q["doc_id"].astype(str)
    chars = pd.read_parquet(inp["characters"], columns=["doc_id", "char_id", "number", "gender_firm", "gender_score", "gender_conflict"])
    ent_by = {d: g.sort_values("onset").reset_index(drop=True) for d, g in ents.groupby("doc_id", sort=False)}
    char_by = {d: g for d, g in chars.groupby("doc_id", sort=False)}
    empty_e = ents.iloc[:0]; empty_c = chars.iloc[:0]
    rows, order = [], []
    for d, g in q.groupby("doc_id", sort=False):
        order += list(g.index)
        rows += (link_doc(docs.at[d, "text"], g, ent_by.get(d, empty_e), char_by.get(d, empty_c)) if d in docs.index
                 else [dict.fromkeys(COLS) for _ in range(len(g))])
    links = pd.DataFrame(rows, columns=COLS, index=order).reindex(q.index)
    for i, (s, cg, cs, cc) in enumerate(zip(q["source"], links["chain_gender"], links["chain_score"], links["chain_conflict"])):
        gd = gendering.genderize(s, chain_gender=cg if cg in ("Male", "Female") else None,
                                 chain_score=None if pd.isna(cs) else float(cs), chain_conflict=bool(cc) if pd.notna(cc) else False)
        for c in ("gender_firm", "gender_firm_basis", "gender_inclusive", "gender_pronoun_det", "gender_conflict", "gender_score"):
            links.iat[i, links.columns.get_loc(c)] = gd[c]
    res = pd.concat([q.reset_index(drop=True), links.reset_index(drop=True)], axis=1)
    for c in ("q_start", "q_end", "src_start", "src_end", "chain", "chain_size"):
        res[c] = pd.to_numeric(res[c]).astype("Int32")
    for c in ("chain_score", "gender_score"):
        res[c] = pd.to_numeric(res[c]).astype("float32")
    for c in ("chain_conflict", "gender_conflict"):
        res[c] = res[c].astype("boolean")
    res.to_parquet(out["quote_links"], index=False)
    lk = res["link"].fillna("")
    return {"quotes": len(res), "found": int((res["q_start"] >= 0).sum()), "source_found": int((lk != "").sum()),
            "mention": int(res["mention"].notna().sum()), "chain": int(res["chain"].notna().sum()),
            "gendered": int(res["gender_firm"].isin(["Male", "Female"]).sum())}


register(Stage("quote_links", inputs=["text", "quotes", "mentions", "entities", "characters"],
               outputs=[Output("quote_links", "quote_links/shard_{k:05d}.parquet")],
               env="assemble", gpu=False, ttl_hours=2, load=load, run=run, doc=__doc__))
