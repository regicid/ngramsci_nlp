"""Fast propp NER (same models and outputs as propp's generate_entities_df): takes the spaCy token tables instead of
re-parsing, batches the CamemBERT sliding windows across documents (length-sorted, dynamic padding, optional fp16
autocast), vectorises the subword<->token mapping and pooling, and reads the CRF confidences in one gather instead of
one .item() per token. The BiLSTM is unpacked, so the head keeps propp's grouping (32 consecutive sentences of one
document per batch) to give identical predictions.
Window boundaries come from propp's own get_boudaries_list, so the windows are the same as propp's."""
import types
import numpy as np, pandas as pd, torch
from propp_fr.propp_fr_generate_tokens_embeddings_tensor import get_boudaries_list
from propp_fr.propp_fr_mentions_detection_module import (predict_BIOES_tags, remove_BIO_illegal_transitions,
                                                         extract_entities_from_BIO_tag_list, combine_predicted_entities)


def viterbi_batched(crf, h, mask):
    """TorchCRF.viterbi_decode with the backtrack batched on the GPU (same recursion, same argmax)."""
    B, T, _ = h.size(); lens = mask.sum(1)
    score = [crf.start_trans.data + h[:, 0]]; path = []
    for t in range(1, T):
        best, arg = (score[t - 1].unsqueeze(2) + crf.trans_matrix + h[:, t].unsqueeze(1)).max(1)
        score.append(best); path.append(arg)
    S = torch.stack(score)                                                 # (T, B, L)
    ar = torch.arange(B, device=h.device)
    cur = (S[lens - 1, ar] + crf.end_trans).max(-1)[1]                     # best last label per sentence
    out = torch.zeros(B, T, dtype=torch.long, device=h.device); out[ar, lens - 1] = cur
    for t in range(T - 2, -1, -1):                                         # label_t = path[t][label_{t+1}] while t < len-1
        act = t < lens - 1
        nxt = path[t][ar, cur]; cur = torch.where(act, nxt, cur); out[:, t] = torch.where(act, cur, out[:, t])
    return out, lens


def fast_crf_forward(self, embeddings, attention_mask=None, labels=None):
    """NERModel.forward in eval mode: batched Viterbi, confidences gathered at once (same values as propp's)."""
    x = self.locked_dropout(embeddings); x = self.embedding2nn(x)
    self.lstm.flatten_parameters(); h, _ = self.lstm(x); h = self.locked_dropout(h); logits = self.linear(h)
    tags, lens = viterbi_batched(self.crf, logits, attention_mask.bool())
    conf = torch.softmax(logits, dim=-1).gather(2, tags.unsqueeze(2)).squeeze(2)
    tags, conf, lens = tags.cpu().tolist(), conf.cpu().tolist(), lens.cpu().tolist()
    return [t[:n] for t, n in zip(tags, lens)], [c[:n] for c, n in zip(conf, lens)]


def patch_head(md, device="cuda"):
    for m in md["models"]:
        m["model"].to(device).eval(); m["model"].forward = types.MethodType(fast_crf_forward, m["model"])


def entities_fast(tdf, tet, md, batch_size=32):
    """propp's generate_entities_df with numpy sentence grouping, cumulative-sum confidences and array-sliced text.
    Same sentence order (sorted sentence_ID), same 32-sentence batches, same BIOES clean-up / extraction / level merge."""
    sid = tdf["sentence_ID"].to_numpy(); order = np.argsort(sid, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(sid[order])) + 1)
    sents = [tet[torch.as_tensor(g, device=tet.device)] for g in groups]
    dfs = []
    for ner in md["models"]:
        preds, confs = predict_BIOES_tags(ner["model"], sents, batch_size=batch_size)
        flat = [l for seq in preds for l in remove_BIO_illegal_transitions([ner["id2label"][i] for i in seq])]
        e = extract_entities_from_BIO_tag_list(flat)
        cs = np.concatenate([[0.0], np.cumsum([c for seq in confs for c in seq])])
        st, en = e["start_token"].to_numpy(int), e["end_token"].to_numpy(int)
        e["confidence"] = (cs[en + 1] - cs[st]) / (en - st + 1) if len(e) else []
        dfs.append(e)
    E = combine_predicted_entities(dfs, priority="confidence")
    words = tdf["word"].to_numpy(object)
    E["text"] = [" ".join(words[a:b + 1]) for a, b in zip(E["start_token"].to_numpy(int), E["end_token"].to_numpy(int))]
    return E


def prep(text, tdf, tok, win=512, overlap=0.5):
    """subword ids, window boundaries, and (token, subword) pairs exactly as propp's find_subwords_for_tokens."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    on = np.fromiter((o[0] for o in offs), int, len(offs)); off = np.fromiter((o[1] for o in offs), int, len(offs))
    to, tf = tdf["byte_onset"].to_numpy(int), tdf["byte_offset"].to_numpy(int)
    s = np.maximum(np.searchsorted(on, to, side="right") - 1, 0)
    e = np.maximum(s, np.searchsorted(on, tf, side="left"))
    ln = e - s; tk = np.repeat(np.arange(len(to)), ln)
    sb = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(s, ln)
    keep = ~((off[sb] <= to[tk]) | (on[sb] >= tf[tk]))
    bounds = get_boudaries_list(tdf, sub_word_offsets=offs, sliding_window_size=win, sliding_window_overlap=overlap)
    return ids, bounds, tk[keep], sb[keep]


@torch.inference_mode()
def embed_docs(items, emb, pad_id, fp16=True, max_batch_tokens=24576, device="cuda"):
    """items: list of (ids, bounds, tok_idx, sub_idx, n_tokens) -> list of CPU float32 token-embedding tensors."""
    H = emb.config.hidden_size
    sums = [torch.zeros(len(it[0]), H, device=device) for it in items]
    cnts = [torch.zeros(len(it[0]), device=device) for it in items]
    wins = sorted(((d, a, b) for d, it in enumerate(items) for a, b in it[1]), key=lambda w: w[2] - w[1], reverse=True)
    i = 0
    while i < len(wins):
        L = wins[i][2] - wins[i][1]; nb = max(1, max_batch_tokens // max(L, 1)); batch = wins[i:i + nb]; i += nb
        x = torch.full((len(batch), L), pad_id, dtype=torch.long)
        for j, (d, a, b) in enumerate(batch): x[j, :b - a] = torch.as_tensor(items[d][0][a:b])
        x = x.to(device); att = (x != pad_id).long()
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            hs = emb(x, attention_mask=att).last_hidden_state
        hs = hs.float()
        for j, (d, a, b) in enumerate(batch):
            sums[d][a:b] += hs[j, :b - a]; cnts[d][a:b] += 1
    out = []
    for d, it in enumerate(items):
        mean_sub = sums[d] / cnts[d].clamp(min=1).unsqueeze(1)
        tk = torch.as_tensor(it[2], device=device); sb = torch.as_tensor(it[3], device=device)
        te = torch.zeros(it[4], H, device=device).index_add_(0, tk, mean_sub[sb])
        c = torch.bincount(tk, minlength=it[4]).clamp(min=1).unsqueeze(1)
        out.append(te / c)                                                  # stays on the GPU
    return out


def ner_block(texts, tdfs, tok, emb, md, fp16=True):
    """NER for a block of documents (tdfs = propp token tables, index 0..n-1). Returns one entities df per doc."""
    items = []
    for t, d in zip(texts, tdfs):
        ids, bounds, tk, sb = prep(t, d, tok, win=512 if tok.model_max_length > 100000 else tok.model_max_length)
        items.append((ids, bounds, tk, sb, len(d)))
    E = embed_docs(items, emb, int(tok.pad_token_id), fp16=fp16)
    return [entities_fast(d, e, md) if len(d) else pd.DataFrame() for d, e in zip(tdfs, E)]
