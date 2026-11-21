"""Section classifier: 6 rubriques (culture, international, politique, société, sport, économie) with
regicid/bert-rubriques (CamemBERT-base fine-tune). GPU fp16 if available, else CPU; length-sorted batches."""
import numpy as np
import pandas as pd

from ..stage import Output, Stage, register
from ..textio import read_text

MODEL, TOKENIZER = "regicid/bert-rubriques", "camembert/camembert-base"   # the fine-tune repo ships no tokenizer


def load(ctx):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    m = AutoModelForSequenceClassification.from_pretrained(ctx.p("rubriques", "model", MODEL)).to(dev).eval()
    if dev == "cuda":
        m = m.half()
    return tok, m, dev


def classify(texts, tok, model, dev, batch=64, max_length=512):
    import torch
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    texts = [t.replace("///", "\n", 1) for t in texts]
    order = np.argsort([len(t) for t in texts])[::-1]
    probs = np.zeros((len(texts), len(labels)), dtype=np.float32)
    with torch.inference_mode():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            enc = tok([texts[i][:6000] for i in idx], truncation=True, max_length=max_length, padding=True,
                      return_tensors="pt").to(dev)
            probs[idx] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
    return labels, probs


def run(k, inp, out, state, ctx):
    tok, model, dev = state
    docs = read_text(inp["text"])
    labels, probs = classify(docs["text"].tolist(), tok, model, dev, batch=ctx.p("rubriques", "batch", 64))
    df = pd.DataFrame({"doc_id": docs["doc_id"], "rubrique": [labels[i] for i in probs.argmax(1)] if len(docs) else []})
    for j, l in enumerate(labels):
        df[f"p_{l}"] = probs[:, j]
    df.to_parquet(out["rubriques"], index=False)
    return {"docs": len(df)}


register(Stage("rubriques", inputs=["text"], outputs=[Output("rubriques", "rubriques/shard_{k:05d}.parquet")],
               env="rubriques", gpu=True, ttl_hours=2, load=load, run=run, doc=__doc__))
