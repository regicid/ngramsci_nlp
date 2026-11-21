"""Quotes: speaker, cue and quote type with the GenderedNews REBEL model (BARThez fine-tuned on FRACAS, Ange Richard;
checkpoint on Zenodo, record 15228575), run with CTranslate2 (beam 3, length-sorted batches; ~7x faster than HF).
Input units: by default the article truncated to 512 BARThez tokens, as in the original pipeline; split="chunks"
covers whole articles in pieces of <= 480 tokens cut at sentence ends.
One row per quote: content, quote_type, cue, source, source_type (+ the unit it came from)."""
import os
import re

import pandas as pd

from ..stage import Output, Stage, register
from ..textio import read_text

SPECIAL = ["<per>", "<peop>", "<pron>", "<org>", "<cue>", "<dirquot>", "<indquot>", "<mixquot>"]


def tokenizer(pack):
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(pack, "barthez", "tokenizer.json"), bos_token="<s>", eos_token="</s>",
                                  unk_token="<unk>", pad_token="<pad>", mask_token="<mask>", cls_token="<s>", sep_token="</s>",
                                  additional_special_tokens=["<triplet>"])
    tok.add_tokens(SPECIAL, special_tokens=True)
    return tok


def convert(pack, dest):
    """REBEL state dict -> HF model -> CTranslate2 model dir (once per site)."""
    import ctranslate2, torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    b, tok = os.path.join(pack, "barthez"), tokenizer(pack)
    cfg = AutoConfig.from_pretrained(b, decoder_start_token_id=0, early_stopping=False, no_repeat_ngram_size=0)
    m = AutoModelForSeq2SeqLM.from_pretrained(b, config=cfg); m.resize_token_embeddings(len(tok))
    m.load_state_dict(torch.load(os.path.join(pack, "rebel_state_dict.pt"), map_location="cpu"), strict=True)
    hf = dest + "_hf"; m.save_pretrained(hf); tok.save_pretrained(hf)
    ctranslate2.converters.TransformersConverter(hf).convert(dest, force=True)


def load(ctx):
    import ctranslate2
    pack = ctx.site.models.get("rebel") if ctx.site else ctx.p("quotes", "pack")
    if not pack:
        raise SystemExit("no REBEL model: set models.rebel in the site file (dir with barthez/ and rebel_state_dict.pt)")
    ct2 = (ctx.site.models.get("rebel_ct2") if ctx.site else None) or os.path.join(pack, "ct2")
    if not os.path.exists(os.path.join(ct2, "model.bin")):
        ctx.log(f"converting REBEL to CTranslate2 in {ct2}"); convert(pack, ct2)
    dev = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    ct = ctx.p("quotes", "compute_type", "float16" if dev == "cuda" else "int8")
    return tokenizer(pack), ctranslate2.Translator(ct2, device=dev, compute_type=ct)


def units(text, tok, split="truncate", max_tok=480):
    """[(unit_index, text)]: the article itself (truncated by the tokenizer) or sentence-cut chunks."""
    text = text.replace("///", "\n", 1)
    if split != "chunks":
        return [(0, text)]
    sents = re.split(r"(?<=[.!?»])\s+", text)
    out, cur, n = [], [], 0
    for s in sents:
        L = len(tok(s, add_special_tokens=False)["input_ids"])
        if cur and n + L > max_tok:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += L
    if cur:
        out.append(" ".join(cur))
    return list(enumerate(out))


def to_quotes(rels):
    """REBEL relations -> quotes: each 'cité' relation (speaker -> quote), with the cue of the 'indique' relation
    pointing at the same quote text."""
    cues = {}
    for r in rels:
        if r.get("type") == "indique" and r.get("head_type") == "<cue>":
            cues.setdefault(r.get("tail"), r.get("head"))
    return [{"content": r["tail"], "quote_type": r.get("tail_type"), "cue": cues.get(r["tail"]),
             "source": r.get("head"), "source_type": r.get("head_type")}
            for r in rels if r.get("type") == "cité" and r.get("tail")]


def predict(texts, tok, tr, batch=128, beam=3):
    from ..nlp.rebel_parse import extract_triplets_rcv2
    src = [tok.convert_ids_to_tokens(tok(t, truncation=True, max_length=512)["input_ids"]) for t in texts]
    order = sorted(range(len(src)), key=lambda i: -len(src[i])); preds = [None] * len(src)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        res = tr.translate_batch([src[i] for i in idx], beam_size=beam, max_decoding_length=1024, length_penalty=0.0,
                                 no_repeat_ngram_size=0, max_batch_size=batch)
        for i, r in zip(idx, res):
            ids = tok.convert_tokens_to_ids(r.hypotheses[0])
            preds[i] = extract_triplets_rcv2("<s>" + tok.decode(ids, skip_special_tokens=False))
    return preds


def run(k, inp, out, state, ctx):
    tok, tr = state
    docs = read_text(inp["text"])
    split = ctx.p("quotes", "split", "truncate")
    us = [(d, u, t) for d, x in zip(docs["doc_id"], docs["text"]) for u, t in units(x, tok, split)]
    preds = predict([t for _, _, t in us], tok, tr, batch=ctx.p("quotes", "batch", 128), beam=ctx.p("quotes", "beam", 3))
    rows = [dict(doc_id=d, unit=u, **q) for (d, u, _), p in zip(us, preds) for q in to_quotes(p)]
    cols = ["doc_id", "unit", "content", "quote_type", "cue", "source", "source_type"]
    pd.DataFrame(rows, columns=cols).to_parquet(out["quotes"], index=False)
    return {"docs": len(docs), "units": len(us), "quotes": len(rows)}


register(Stage("quotes", inputs=["text"], outputs=[Output("quotes", "quotes/shard_{k:05d}.parquet")],
               env="quotes", gpu=True, ttl_hours=3, load=load, run=run, doc=__doc__))
