"""spaCy parse only -> the private token tables, for shards whose mentions and windows exist already but whose token
tables were not kept (e.g. made by older scripts). Same parse as parse_ner, so the tokens line up with its mentions
(assemble checks it). Site-local: inputs and outputs are private, so claims are disk locks on this site, not hub commits."""
import time

from ..stage import Stage, register
from ..textio import read_text
from .parse_ner import TOKENS, parse, write_tokens


def load(ctx):
    import spacy
    ctx.log(f"spacy gpu: {spacy.prefer_gpu()}")
    nlp = spacy.load(ctx.p("parse_ner", "spacy_model", "fr_dep_news_trf")); nlp.max_length = 500_000
    return nlp


def run(k, inp, out, nlp, ctx):
    t0 = time.time()
    docs = read_text(inp["text"])
    tdfs, err = parse(docs["text"].tolist(), nlp, ctx.log)
    write_tokens(tdfs, docs["doc_id"].tolist(), out["tokens"])
    return {"docs": len(docs), "errors": err, "parse_s": round(time.time() - t0)}


register(Stage("parse", inputs=["text"], outputs=[TOKENS], env="propp", gpu=True, ttl_hours=2, load=load, run=run, doc=__doc__))
