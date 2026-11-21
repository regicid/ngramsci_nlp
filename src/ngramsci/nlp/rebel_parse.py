"""Parse the REBEL quote model's generated text into relations.

Output format of the GenderedNews REBEL model (BARThez fine-tuned on FRACAS): a sequence of triplets
    <triplet> TAIL <tail_tag> HEAD <head_tag> RELATION [HEAD <head_tag> RELATION ...]
where TAIL is the quote (or a speaker), HEAD the speaker or cue, tags are <per> <peop> <pron> <org> <cue> <dirquot>
<indquot> <mixquot>, and RELATION is the French label `cité` (quoted in) or `indique` (indicates).

Relations come out as dicts {head, head_type, tail, tail_type, type}, identical to those of the GenderedNews pipeline
(checked on 1,000 articles), including two of its conventions:
  * a group that is not exactly (tail_tag, HEAD, head_tag, RELATION) is still returned, with the relation word stored in
    `tail_type`, the first tag in `head_type`, the first text in `head`, '' elsewhere;
  * a tag with no text before it is ignored.
"""
TAGS = ("<per>", "<peop>", "<pron>", "<org>", "<cue>", "<dirquot>", "<indquot>", "<mixquot>")
RELATIONS = ("cité", "indique")


def _words(text):
    for junk in ("fr_XX", "tp_XX"):
        text = text.replace(junk, "")
    text = text.replace("<", " <")                            # tags are glued to the preceding word by the tokenizer
    for junk in ("<s>", "</s>", "<pad>"):
        text = text.replace(junk, "")
    return text.split()


def _items(words):
    """Words of one triplet -> alternating [text, tag, text, tag, ..., (text)]. A tag right after another tag is dropped."""
    items, buf = [], []
    for w in words:
        if w in TAGS:
            if buf:
                items += [" ".join(buf), w]; buf = []
        else:
            buf.append(w)
    if buf:
        items.append(" ".join(buf))
    return items


def _relation(tail, group):
    if len(group) == 4:
        tail_tag, head, head_tag, rel = group
        return {"head": head, "head_type": head_tag, "tail": tail, "tail_type": tail_tag, "type": rel}
    out = {"head": "", "head_type": "", "tail": tail, "tail_type": "", "type": ""}
    for it in group:
        if it in RELATIONS:
            out["tail_type"] = it
        elif it in TAGS:
            out["head_type"] = out["head_type"] or it
        elif not out["head"]:
            out["head"] = it
    return out


def parse(generated: str) -> list:
    words = _words(generated.strip())
    if not words:
        return []
    if "<triplet>" in words:
        cuts = [i for i, w in enumerate(words) if w == "<triplet>"]
        triplets = [words[a + 1:b] for a, b in zip(cuts, cuts[1:] + [len(words)])]   # text before the first marker: ignored
    else:
        triplets = [words]
    rels = []
    for t in triplets:
        items = _items(t)
        if len(items) == 1:
            rels.append({"head": "", "head_type": "", "tail": items[0], "tail_type": "", "type": ""})
        elif len(items) > 1:
            tail, group = items[0], []
            for it in items[1:]:
                group.append(it)
                if it in RELATIONS:                            # a relation word closes a group
                    rels.append(_relation(tail, group)); group = []
    return rels
