"""spaCy parse + propp mention detection + Qwen windows, in one pass over a shard (one GPU process; run 2 per V100).

1. fr_dep_news_trf parse on GPU, in length-sorted chunks with ~200k chars per spaCy batch (OOM -> smaller batches);
   articles of 100k+ chars go through propp's chunked generate_tokens_df.
2. propp NER (CamemBERT-large windows + BiLSTM-CRF, nested levels merged) with nlp.ner_fast: same mentions as
   propp's generate_entities_df (fp32: identical; fp16: 99.97 % identical), ~7x faster.
3. k=10 coreference windows over PER mentions (nlp.windows).
Outputs: tokens (private: the token table holds the text), mentions, windows (sent to Qwen), ner_done (timings)."""
import json
import time
import traceback

import pandas as pd

from ..nlp.windows import CORE_COLS, WIN_K, build_windows
from ..stage import Output, Stage, register
from ..textio import read_text, write_jsonl_gz

LONG = 100_000


def load(ctx):
    import spacy, torch
    from ..nlp import ner_fast
    from ..nlp.propp_compat import load_ner_models
    from propp_fr import load_tokenizer_and_embedding_model
    gpu = spacy.prefer_gpu()
    ctx.log(f"spacy gpu: {gpu}")
    nlp = spacy.load(ctx.p("parse_ner", "spacy_model", "fr_dep_news_trf")); nlp.max_length = 500_000
    md = load_ner_models()
    tok, emb = load_tokenizer_and_embedding_model(md["base_model_name"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = emb.to(dev).eval()
    torch.backends.cudnn.enabled = False          # cuDNN LSTM: HOST_ALLOCATION_FAILED on V100s; propp's LSTM is tiny
    ner_fast.patch_head(md, device=dev)
    return {"nlp": nlp, "md": md, "tok": tok, "emb": emb}


def parse(texts, nlp, log=print):
    """-> list of propp token tables (None where parsing failed), n_errors."""
    import torch
    from propp_fr import generate_tokens_df
    from ..nlp.propp_compat import tokens_table
    tdfs, n_err = [None] * len(texts), 0
    short = sorted((i for i, t in enumerate(texts) if len(t) < LONG), key=lambda i: len(texts[i]))
    for c in range(0, len(short), 512):
        chunk = short[c:c + 512]; L = max(len(texts[i]) for i in chunk)
        b0 = max(2, min(64, 200_000 // max(L, 1)))
        for bs in (b0, max(1, b0 // 8), 1):
            try:
                for i, d in zip(chunk, nlp.pipe((texts[i] for i in chunk), batch_size=bs)):
                    try:
                        tdfs[i] = tokens_table(d).reset_index(drop=True)
                    except Exception:
                        n_err += 1
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); log(f"parse OOM at batch {bs}, retrying smaller")
    for i in (i for i in range(len(texts)) if len(texts[i]) >= LONG):
        try:
            tdfs[i] = generate_tokens_df(texts[i], nlp, verbose=0).reset_index(drop=True)
        except Exception:
            n_err += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return tdfs, n_err


def ner(texts, ids, tdfs, state, fp16=True, k_win=WIN_K, log=print):
    """-> mentions DataFrame, window dicts, n_errors."""
    import torch
    from ..nlp import ner_fast
    tok, emb, md = state["tok"], state["emb"], state["md"]
    rows, wins, n_err = [], [], 0
    ok = [i for i in range(len(texts)) if tdfs[i] is not None and len(tdfs[i])]
    for b in range(0, len(ok), 64):
        blk = ok[b:b + 64]
        try:
            ents = ner_fast.ner_block([texts[i] for i in blk], [tdfs[i] for i in blk], tok, emb, md, fp16=fp16)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); ents = []
            for i in blk:                                  # retry the block document by document
                try:
                    ents += ner_fast.ner_block([texts[i]], [tdfs[i]], tok, emb, md, fp16=fp16)
                except Exception:
                    ents.append(None); n_err += 1; torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc(); n_err += len(blk); continue
        for i, e in zip(blk, ents):
            if e is None or len(e) == 0:
                continue
            try:
                tdf = tdfs[i]; m = e[CORE_COLS].reset_index(drop=True)
                m["start_token"] = m["start_token"].astype(int); m["end_token"] = m["end_token"].astype(int)
                m.insert(0, "mid", range(len(m)))
                m["onset"] = tdf["byte_onset"].to_numpy()[m["start_token"].to_numpy()]
                m["offset"] = tdf["byte_offset"].to_numpy()[m["end_token"].to_numpy()]
                m.insert(0, "doc_id", ids[i]); rows.append(m)
                wins += build_windows(ids[i], texts[i], list(zip(tdf["byte_onset"], tdf["byte_offset"])), list(tdf["word"]), m, k_win)
            except Exception as ex:
                n_err += 1; log(f"{ids[i]}: {type(ex).__name__}: {str(ex)[:200]}")
    cols = ["doc_id", "mid"] + CORE_COLS + ["onset", "offset"]
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)), wins, n_err


def write_tokens(tdfs, ids, path):
    import pyarrow as pa, pyarrow.parquet as pq
    from ..nlp.propp_compat import to_str_columns
    writer, schema, batch = None, None, []

    def flush():
        nonlocal writer, schema, batch
        if not batch:
            return
        tab = pa.Table.from_pandas(pd.concat(batch, ignore_index=True), preserve_index=False)
        if writer is None:
            schema = tab.schema; writer = pq.ParquetWriter(path, schema, compression="zstd")
        writer.write_table(tab.cast(schema)); batch = []
    for d, t in zip(ids, tdfs):
        if t is not None and len(t):
            batch.append(to_str_columns(t.copy()).assign(doc_id=d))
            if len(batch) >= 500:
                flush()
    flush()
    if writer is None:
        pd.DataFrame(columns=["doc_id"]).to_parquet(path)
    else:
        writer.close()


def run(k, inp, out, state, ctx):
    import torch
    t0 = time.time()
    docs = read_text(inp["text"])
    texts, ids = docs["text"].tolist(), docs["doc_id"].tolist()
    tdfs, e1 = parse(texts, state["nlp"], ctx.log)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t_parse = time.time() - t0
    ments, wins, e2 = ner(texts, ids, tdfs, state, fp16=not ctx.p("parse_ner", "fp32", False),
                          k_win=ctx.p("parse_ner", "k", WIN_K), log=ctx.log)
    write_tokens(tdfs, ids, out["tokens"])
    ments.to_parquet(out["mentions"], index=False, compression="zstd")
    write_jsonl_gz(wins, out["windows"])
    info = {"docs": len(texts), "errors": e1 + e2, "mentions": len(ments), "windows": len(wins), "parse_s": round(t_parse),
            "total_s": round(time.time() - t0), "s_doc": round((time.time() - t0) / max(1, len(texts)), 4),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    json.dump(info, open(out["ner_done"], "w"))
    return info


register(Stage(
    "parse_ner", inputs=["text"], env="propp", gpu=True, ttl_hours=3, load=load, run=run, doc=__doc__,
    outputs=[Output("tokens", "tokens/shard_{k:05d}.parquet", private=True),
             Output("mentions", "mentions/shard_{k:05d}.ner.parquet"),
             Output("windows", "windows_k10/shard_{k:05d}.ner.jsonl.gz"),
             Output("ner_done", "ner_done/shard_{k:05d}.json")],
    done="ner_done"))
