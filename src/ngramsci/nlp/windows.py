"""Qwen coreference windows: ~500-char segments cut at sentence ends (never inside a mention), k segments per
window with one segment of overlap, PER mentions marked inline as '{___ mention}'. Same markup as the coreference
package's preprocess_coreference (what the lattice-nlp Qwen model was trained on)."""

CORE_COLS = ["start_token", "end_token", "cat", "confidence", "text"]   # output of propp's generate_entities_df
COREF_CATS = ["PER"]                                                  # entity types sent to coreference
SEG_CHARS = 500
WIN_K = 10                    # segments per window (one-segment overlap): ~5000 chars


def propp_tokens(nlp_tokenizer, text):
    """Token list exactly as propp numbers it: spaCy tokens, newline-containing tokens dropped."""
    words, spans = [], []
    for t in nlp_tokenizer(text):
        if "\n" in t.text_with_ws:
            continue
        words.append(t.text)
        spans.append((t.idx, t.idx + len(t.text)))
    return words, spans


def segments(text, spans, words, mentions, seg_chars=SEG_CHARS):
    """Split a document into ~seg_chars segments at sentence-ish boundaries (after . ! ? or a
    newline), never inside a mention. mentions: list of (start_token, end_token). Returns list of
    (char_start, char_end)."""
    n = len(spans)
    if n == 0:
        return []
    inside = [0] * (n + 1)                       # token i is strictly inside a multi-token mention
    for s, e in mentions:
        if e > s:
            inside[s] += 1; inside[e] -= 1
    cover, c = [False] * n, 0
    for i in range(n):
        c += inside[i]; cover[i] = c > 0       # cutting after token i would split a mention
    can_cut = [False] * n
    for i in range(n - 1):
        gap = text[spans[i][1]:spans[i + 1][0]]
        can_cut[i] = (words[i] in (".", "!", "?") or "\n" in gap) and not cover[i]
    segs, start, last_cut = [], 0, None
    for i in range(n):
        if spans[i][1] - spans[start][0] > seg_chars and last_cut is not None:
            segs.append((spans[start][0], spans[last_cut][1]))
            start, last_cut = last_cut + 1, None
        if can_cut[i] and i >= start:
            last_cut = i
    segs.append((spans[start][0], spans[n - 1][1]))
    return segs


def windows(segs, k=None):
    """k consecutive segments per window, consecutive windows share one segment."""
    k = k or WIN_K
    if not segs:
        return []
    if len(segs) <= k:
        return [(segs[0][0], segs[-1][1])]
    return [(segs[i][0], segs[min(i + k, len(segs)) - 1][1]) for i in range(0, len(segs) - 1, k - 1)]


def window_input(text, w0, w1, mentions):
    """mentions: list of (mid, onset, offset) inside [w0, w1). Same markup as
    coreference.preprocess_coreference: '{___ ' before the first char, '}' after the last."""
    ms = sorted(mentions, key=lambda m: (m[1], -(m[2] - m[1])))
    chars = list(text[w0:w1])
    for _, on, off in ms:
        chars[on - w0] = "{___ " + chars[on - w0]
        chars[off - 1 - w0] = chars[off - 1 - w0] + "}"
    return "".join(chars), [m[0] for m in ms]


def build_windows(doc_id, text, spans, words, ments, k=WIN_K):
    """ments: DataFrame rows with mid,start_token,end_token,cat,onset,offset. Returns window dicts."""
    per = ments[ments["cat"].isin(COREF_CATS)]
    segs = segments(text, spans, words, list(zip(ments["start_token"], ments["end_token"])))
    out = []
    for wi, (a, b) in enumerate(windows(segs, k)):
        inw = [(m, on, off) for m, on, off in zip(per["mid"], per["onset"], per["offset"]) if on >= a and off <= b]
        if not inw:
            continue
        inp, ids = window_input(text, a, b, inw)
        out.append({"file": doc_id, "w": wi, "input": inp, "mention_ids": [int(i) for i in ids]})
    return out
