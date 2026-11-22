"""Characters: Qwen coreference -> propp features and attributes -> characters, with our gender inference.

Per article: propp's add_features_to_entities on the mentions, coreference chains from the Qwen windows (nlp.coref),
propp's extract_attributes (agent / patient / modifier / possessive words of each mention) and
generate_characters_dict (min_occurrences=1: people mentioned once are kept), gender re-inferred by gender.regender_book,
plus the extended attribute rules of nlp.ext_rules (passives, control verbs, coordination, ...).
CPU only, ~0.6 core-s per article; `processes` (default: the job's CPUs) splits a shard's articles over processes. Needs the private token tables, so it runs on the site that parsed the shard."""
import os
from collections import Counter, defaultdict

import pandas as pd

from ..nlp.coref import coref_ids
from ..nlp.windows import CORE_COLS, COREF_CATS
from ..stage import Output, Stage, register
from ..textio import read_jsonl, read_text

ELISION = {"d’": "de", "l’": "le", "qu’": "que", "s’": "se", "n’": "ne", "j’": "je", "c’": "ce",
           "m’": "me", "t’": "te", "jusqu’": "jusque", "lorsqu’": "lorsque", "puisqu’": "puisque",
           "quoiqu’": "quoique"}


def load(ctx):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from ..gender import gendering
    from ..gender.regender_book import JOBS_CSV, NAMES_CSV
    gendering.init_resources(NAMES_CSV, JOBS_CSV)
    return None


def schemas():
    import pyarrow as pa
    from ..nlp.tables import CHAR_SCHEMA
    A3 = pa.list_(pa.struct([("w", pa.string()), ("i", pa.int32()), ("r", pa.string())]))
    A2 = pa.list_(pa.struct([("w", pa.string()), ("i", pa.int32())]))
    s = CHAR_SCHEMA
    for f in ("agent2", "patient2", "mod2", "poss2", "obl2"):
        s = s.append(pa.field(f, A3))
    return s.append(pa.field("agent_drop", A2))


def assemble_doc(doc_id, paper, m, tdf, wins, outs, fix_apos=False):
    """One article -> (entities df, character rows, n_bad_windows)."""
    from propp_fr import add_features_to_entities, extract_attributes, generate_characters_dict
    from coreference import extract_predicted_COREF_ids_from_output as extract
    from ..gender.regender_book import regender_book
    from ..nlp.ext_rules import extended
    from ..nlp.tables import ENT_COLS, char_row
    if fix_apos:                                    # spaCy lemmatises curly-apostrophe elisions as themselves ("d’")
        lm = tdf["lemma"].astype(str)
        tdf["lemma"] = lm.where(~lm.str.endswith("’"), lm.str.lower().map(lambda x: ELISION.get(x, x)))
    m = m.sort_values("mid")
    ents = m[CORE_COLS].reset_index(drop=True)
    ents["confidence"] = pd.to_numeric(ents["confidence"], errors="coerce")
    if ents["end_token"].max() >= len(tdf):
        raise ValueError("token mismatch")
    if "onset" in m and len(m):                     # tokens re-parsed apart from the mentions must line up with them
        on = pd.to_numeric(m["onset"], errors="coerce").to_numpy()
        if (tdf["byte_onset"].to_numpy()[ents["start_token"].astype(int).to_numpy()] != on).any():
            raise ValueError("token mismatch (mention offsets)")
    ents = add_features_to_entities(ents, tdf)
    mids = m["mid"].tolist()
    per_mids = [mid for mid, p in zip(mids, ents["cat"].isin(COREF_CATS).tolist()) if p]
    pos = {mid: i for i, mid in enumerate(mids)}
    cid, bad = coref_ids(per_mids, wins, outs, {mid: ents.at[pos[mid], "text"] for mid in per_mids},
                         {mid: ents.at[pos[mid], "prop"] for mid in per_mids}, extract)
    ents["COREF"] = [cid.get(mid) for mid in mids]
    tdf2 = extract_attributes(ents, tdf)
    chars = regender_book(generate_characters_dict(tdf2, ents, min_occurrences=1))
    new, drop = extended(tdf2, ents["head_id"])
    e = ents.reindex(columns=ENT_COLS).astype(str).replace({"None": "", "nan": "", "<NA>": ""})
    lem = tdf2["lemma"].str.lower()
    for slot in ("agent", "patient", "mod", "poss"):              # per-mention attributes, keyed by head_id
        col = tdf2[f"char_att_{slot}"]; att = defaultdict(list)
        for w, i, h in zip(lem[col.notna()], tdf2.loc[col.notna(), "token_ID_within_document"], col.dropna()):
            att[int(h)].append({"w": w, "i": int(i)})
        e[f"att_{slot}"] = [att.get(int(h), []) if str(h) not in ("", "nan") else [] for h in ents["head_id"]]
    hk = [int(h) if str(h) not in ("", "nan", "None") else None for h in ents["head_id"]]
    for slot in ("agent", "patient", "mod", "poss", "obl"):
        e[f"att2_{slot}"] = [list(new[h][slot]) if h in new else [] for h in hk]
    e["att2_agent_drop"] = [list(drop[h]) if h in drop else [] for h in hk]
    e.insert(0, "doc_id", doc_id); e.insert(0, "newspaper", paper)
    heads_of = defaultdict(list)
    for h, c, cat in zip(hk, ents["COREF"], ents["cat"]):
        if cat in COREF_CATS and h is not None and c is not None and str(c) not in ("", "nan", "None"):
            heads_of[int(float(c))].append(h)
    rows = []
    for c in chars:
        r = char_row(c, paper, doc_id); hs = heads_of.get(int(c.get("id")), [])
        for slot in ("agent", "patient", "mod", "poss", "obl"):
            r[f"{slot}2"] = [a for h in hs for a in new.get(h, {}).get(slot, [])]
        r["agent_drop"] = [a for h in hs for a in drop.get(h, [])]
        rows.append(r)
    return e, rows, bad


def _assemble_many(items, fix_apos):
    """Worker body: assemble a list of articles. Returns (entity dfs, character rows, counts, error messages)."""
    ent_dfs, rows, st, errs = [], [], Counter(), []
    for d, paper, m, tdf, wins, outs in items:
        try:
            e, r, bad = assemble_doc(d, paper, m, tdf, wins, outs, fix_apos=fix_apos)
            ent_dfs.append(e); rows += r; st["bad_windows"] += bad; st["docs_ok"] += 1
        except Exception as ex:
            st["errors"] += 1; errs.append(f"{d}: {type(ex).__name__}: {str(ex)[:200]}")
    return ent_dfs, rows, st, errs


def run(k, inp, out, state, ctx):
    import pyarrow as pa, pyarrow.parquet as pq
    from ..nlp.tables import ent_table
    docs = read_text(inp["text"])
    paper_of = dict(zip(docs["doc_id"], docs["media"])) if "media" in docs else {}
    ments = pd.read_parquet(inp["mentions"])
    win_by, out_by = defaultdict(list), defaultdict(dict)
    for w in read_jsonl(inp["windows"]):
        win_by[w["file"]].append(w)
    for r in read_jsonl(inp["qwen"]):
        if r.get("output") is not None:
            out_by[r["file"]][r["w"]] = r["output"]
    ment_by = {d: g for d, g in ments.groupby("doc_id", sort=False)}
    ids = [d for d in docs["doc_id"] if d in ment_by]
    tok = pq.read_table(inp["tokens"], filters=[("doc_id", "in", ids)]).to_pandas() if ids else pd.DataFrame(columns=["doc_id"])
    tok_by = {d: g.drop(columns="doc_id").reset_index(drop=True) for d, g in tok.groupby("doc_id", sort=False)}
    del tok
    todo = [(d, paper_of.get(d, ctx.corpus.name), ment_by[d], tok_by.pop(d), win_by.get(d, []), out_by.get(d, {}))
            for d in ids if d in tok_by]
    fix = ctx.p("assemble", "fix_apos", False)
    n = max(1, int(ctx.p("assemble", "processes", os.environ.get("SLURM_CPUS_PER_TASK", 1))))
    if n == 1 or len(todo) < 2 * n:
        parts = [_assemble_many(todo, fix)]
    else:                                             # articles are independent: split them over n processes
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        chunks = [todo[i::n] for i in range(n)]
        with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=load, initargs=(None,)) as ex:
            parts = list(ex.map(_assemble_many, chunks, [fix] * n))
    ent_dfs, char_rows, st = [], [], Counter(no_tokens=len(ids) - len(todo))
    for e, r, c, errs in parts:
        ent_dfs += e; char_rows += r; st.update(c)
        for m in errs:
            ctx.log(m)
    if ent_dfs:
        pq.write_table(ent_table(ent_dfs), out["entities"], compression="zstd")
    else:
        pd.DataFrame(columns=["newspaper", "doc_id"]).to_parquet(out["entities"])
    pq.write_table(pa.Table.from_pylist(char_rows, schema=schemas()), out["characters"], compression="zstd")
    return dict(st, docs=len(ids))


register(Stage(
    "assemble", inputs=["text", "tokens", "mentions", "windows", "qwen"], env="assemble", gpu=False, ttl_hours=6,
    load=load, run=run, doc=__doc__,
    outputs=[Output("entities", "entities/shard_{k:05d}.parquet"), Output("characters", "characters/shard_{k:05d}.parquet")]))
