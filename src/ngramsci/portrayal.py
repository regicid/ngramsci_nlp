"""Portrayal scorer: the CamemBERT-large student of codebook v4 (n-gramsci/portrayal-camembert-large), five seeds averaged.

Input rows: {title, passage, full_name}; the target's mentions in the passage are wrapped in [[ ]]. Mentions matching the
target's name (and the name in the title) are replaced by the placeholder the model was trained with, other marked
mentions (pronouns, descriptions) are kept as written. Output per row: P(1) for pertinent and journaliste, [P(-1), P(0), P(1)]
for valence, extremisme, dynamique."""
from __future__ import annotations

import json
import os
import re

import numpy as np

HUB = "n-gramsci/portrayal-camembert-large"
BASE = "almanach/camembert-large"
HEADS = {"pertinent": 2, "valence": 3, "extremisme": 3, "dynamique": 3, "journaliste": 2}
PH, MAXLEN = "Dominique Martin", 384


def net(base=BASE):
    import torch
    from torch import nn
    from transformers import AutoModel

    class Net(nn.Module):                                  # as in training: target mean ++ input mean -> one linear head each
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(base); h = self.enc.config.hidden_size
            self.drop = nn.Dropout(0.1)
            self.heads = nn.ModuleDict({k: nn.Linear(2 * h, n) for k, n in HEADS.items()})

        def forward(self, ids, att, tm):
            x = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
            tgt = (x * tm.unsqueeze(-1)).sum(1) / tm.sum(1, keepdim=True)
            win = (x * att.unsqueeze(-1)).sum(1) / att.sum(1, keepdim=True)
            return {k: h(torch.cat([tgt, win], -1)) for k, h in self.heads.items()}
    return Net()


def build_input(row, rx):
    pre = f"Titre : {rx.sub(PH, row.get('title') or '')}\n"
    body = re.sub(r"\[\[([^\]]+)\]\]", lambda m: f"[[{PH}]]" if rx.search(m.group(1)) else m.group(0), row["passage"])
    return pre + body, [(len(pre) + m.start(), len(pre) + m.end()) for m in re.finditer(r"\[\[[^\]]+\]\]", body)]


class Scorer:
    def __init__(self, state_dicts, base=BASE, device=None):
        import torch
        from transformers import AutoTokenizer
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(base)
        self.models = []
        for sd in state_dicts:
            m = net(base); m.load_state_dict(sd); m = m.to(self.dev).eval()
            self.models.append(m.half() if self.dev == "cuda" else m)

    @classmethod
    def from_hub(cls, repo=HUB, seeds=None, **kw):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        seeds = cfg["seeds"] if seeds is None else seeds
        sds = [{k: v.float() for k, v in load_file(hf_hub_download(repo, f"seed{s}/model.safetensors")).items()} for s in seeds]
        return cls(sds, base=cfg.get("base_model", BASE), **kw)

    def score(self, rows, batch=64):
        import torch
        from .nlp.segments import name_re
        rxs = {}
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["passage"]))
        acc = {k: np.zeros((len(rows), n), dtype=np.float32) for k, n in HEADS.items()}
        with torch.inference_mode():
            for s in range(0, len(order), batch):
                idx = order[s:s + batch]
                T, S = zip(*[build_input(rows[i], rxs.setdefault(rows[i]["full_name"], name_re(rows[i]["full_name"]))) for i in idx])
                e = self.tok(list(T), truncation=True, max_length=MAXLEN, padding=True, return_offsets_mapping=True, return_tensors="pt")
                off = e.pop("offset_mapping").numpy(); att = e["attention_mask"]
                tm = torch.zeros_like(att, dtype=torch.float32)
                for j, sp in enumerate(S):
                    for t, (a, b) in enumerate(off[j]):
                        if b > a and any(a < y and b > x for x, y in sp):
                            tm[j, t] = 1
                    if tm[j].sum() == 0:
                        tm[j] = att[j].float()
                dt = torch.float16 if self.dev == "cuda" else torch.float32
                ids, att_d, tm_d = e["input_ids"].to(self.dev), att.to(self.dev), tm.to(self.dev, dt)
                for m in self.models:
                    for k, v in m(ids, att_d, tm_d).items():
                        acc[k][idx] += torch.softmax(v.float(), -1).cpu().numpy()
        n = len(self.models)
        return [{"pertinent": float(acc["pertinent"][i, 1] / n), "journaliste": float(acc["journaliste"][i, 1] / n),
                 **{k: [float(x) for x in acc[k][i] / n] for k in ("valence", "extremisme", "dynamique")}} for i in range(len(rows))]


def export(run_dirs, out_dir, base=BASE):
    """Training checkpoints (model.pt state dicts from finetune_segment.py) -> fp16 safetensors per seed + config.json."""
    import torch
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    for s, d in enumerate(run_dirs):
        sd = torch.load(os.path.join(d, "model.pt"), map_location="cpu")
        os.makedirs(os.path.join(out_dir, f"seed{s}"), exist_ok=True)
        save_file({k: v.half().contiguous() for k, v in sd.items()}, os.path.join(out_dir, f"seed{s}", "model.safetensors"))
    json.dump({"base_model": base, "heads": HEADS, "placeholder": PH, "max_length": MAXLEN, "seeds": list(range(len(run_dirs))),
               "pooling": "mean over target spans ++ mean over input", "classes": {"valence": [-1, 0, 1], "extremisme": [-1, 0, 1],
               "dynamique": [-1, 0, 1], "pertinent": [0, 1], "journaliste": [0, 1]}}, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
