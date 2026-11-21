"""Arrow schemas of the published tables (entities = one row per mention, characters = one row per coreference
chain), shared with regicid/propp-press."""
import pandas as pd
import pyarrow as pa

# ---------------------------------------------------------------- characters
MENTIONS = pa.list_(pa.struct([("n", pa.string()), ("c", pa.int32())]))
WORDS = pa.list_(pa.struct([("w", pa.string()), ("i", pa.int32())]))

CHAR_SCHEMA = pa.schema([
    ("newspaper", pa.string()), ("doc_id", pa.string()), ("char_id", pa.int32()),
    ("occurrence", pa.int32()), ("mention_ratio", pa.float32()),
    ("gender_firm", pa.string()), ("gender_inclusive", pa.string()),
    ("gender_pronoun", pa.string()), ("gender_score", pa.float32()),
    ("gender_coverage", pa.float32()), ("gender_n_gendered", pa.int32()),
    ("gender_basis", pa.string()), ("gender_conflict", pa.bool_()),
    ("number", pa.string()), ("number_ratio", pa.float32()), ("p_singular", pa.float32()),
    ("propp_gender", pa.string()), ("propp_ratio", pa.float32()),
    ("propp_p_male", pa.float32()), ("propp_p_female", pa.float32()),
    ("mentions_proper", MENTIONS), ("mentions_common", MENTIONS), ("mentions_pronoun", MENTIONS),
    ("agent", WORDS), ("patient", WORDS), ("mod", WORDS), ("poss", WORDS),
])


def char_row(c, paper, doc_id):
    g, n, k, gp, m = (c.get(x) or {} for x in ("gender", "number", "count", "gender_propp", "mentions"))
    ni, gpi = n.get("inference") or {}, gp.get("inference") or {}
    return {
        "newspaper": paper, "doc_id": doc_id, "char_id": c.get("id"),
        "occurrence": k.get("occurrence"), "mention_ratio": k.get("mention_ratio"),
        "gender_firm": g.get("firm"), "gender_inclusive": g.get("inclusive"),
        "gender_pronoun": g.get("pronoun"), "gender_score": g.get("score"),
        "gender_coverage": g.get("coverage"), "gender_n_gendered": g.get("n_gendered"),
        "gender_basis": g.get("basis"), "gender_conflict": g.get("conflict"),
        "number": n.get("argmax"), "number_ratio": n.get("ratio"), "p_singular": ni.get("Singular"),
        "propp_gender": gp.get("argmax"), "propp_ratio": gp.get("ratio"),
        "propp_p_male": gpi.get("Male"), "propp_p_female": gpi.get("Female"),
        **{f"mentions_{t}": m.get(t) or [] for t in ("proper", "common", "pronoun")},
        **{r: c.get(r) or [] for r in ("agent", "patient", "mod", "poss")},
    }


# ------------------------------------------------------------------ entities
ENT_COLS = [
    "start_token", "end_token", "cat", "confidence", "text", "mention_len",
    "paragraph_ID", "sentence_ID", "start_token_ID_within_sentence",
    "out_to_in_nested_level", "in_to_out_nested_level", "nested_entities_count",
    "temp_index", "head_id", "head_word", "head_dependency_relation",
    "head_syntactic_head_ID", "POS_tag", "prop", "number", "gender",
    "grammatical_person", "COREF",
]
ENT_INT = [
    "start_token", "end_token", "mention_len", "paragraph_ID", "sentence_ID",
    "start_token_ID_within_sentence", "out_to_in_nested_level", "in_to_out_nested_level",
    "nested_entities_count", "temp_index", "head_id", "head_syntactic_head_ID",
    "grammatical_person", "COREF",
]


def ent_table(dfs):
    df = pd.concat(dfs, ignore_index=True)
    for col in ENT_INT:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").astype("float32")
    return pa.Table.from_pandas(df, preserve_index=False)
