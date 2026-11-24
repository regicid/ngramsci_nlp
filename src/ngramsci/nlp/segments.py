"""Mention segments for the portrayal scorer (codebook v4): the sentence mentioning the target + its neighbouring
sentences (±1), each neighbour given to one mention only (the closer; ties to the earlier), the title as separate
context. Embedded « À lire aussi » links are cut; a passage starting inside a quotation opened earlier gets a note.
Every mention of the target's coreference chain is wrapped in [[ ]]."""
import re

split = re.compile(r"(?<=[.!?»])\s+(?=[«A-ZÉÈÀÎÔ\"“])")
LABEL = re.compile(r"extr[êe]me[- ]?(?:droite|gauche)|ultra[- ]?(?:droite|gauche)|radical|populis|nationalis|r[ée]volutionnaire|"
                   r"factieu|arc r[ée]publicain|fascis|n[ée]onazi|identitaire", re.I)
LINK = re.compile(r">?\s*(?:(?:À|A) (?:lire|LIRE|voir|VOIR)\b|(?:Lire|LIRE) (?:aussi|AUSSI|également|ÉGALEMENT)\b)")
NOTE = "[Note: this passage is part of a quotation that begins earlier in the article.]"
# lower-case name particles, several in a row allowed (« Ursula von der Leyen », « Luiz Inácio Lula da Silva »,
# « Bachar al-Assad »)
PARTICLES = "de|du|des|d['’]|da|das|do|dos|del|della|di|von|van|der|den|ter|le|la|ben|bin|el|al|y"
NAMEOK = re.compile(rf"^[A-ZÀ-Ý][\w'’\-]+(?:\s+(?:(?:{PARTICLES})\s+)*(?:d['’]\s*|(?:al|el|ben|bin|ibn)-)?[A-ZÀ-Ý][\w'’\-]+)+$")


def name_re(full_name: str) -> re.Pattern:
    """'Marine Le Pen' -> matches 'Marine Le Pen', 'Le Pen', 'Mme Le Pen' (any whitespace inside the surname)."""
    first, *rest = full_name.split()
    f = re.escape(first).replace("É", "[ÉE]")
    if not rest:                                          # a single word: match it alone (never an empty pattern)
        return re.compile(rf"(?:(?:M\.|Mme|Monsieur|Madame)\s+)?{f}")
    s = r"\s+".join(map(re.escape, rest))
    return re.compile(rf"(?:(?:M\.|Mme|Monsieur|Madame)\s+)?(?:{f}\s+)?{s}")


def cut_link(s):
    m = LINK.search(s)
    return (s[:m.start()].strip(), m.start()) if m else (s, None)


def open_quote(sents, maxlen=6):
    """Per sentence: index of the sentence where a still-open « began (None if none). Sentence 0 is the title.
    A « open for more than maxlen sentences is a stray, not a quote."""
    state, depth, op = [], 0, None
    for i, s in enumerate(sents):
        state.append(op if depth > 0 and i > 0 and i - op <= maxlen else None)
        if i == 0:
            continue
        for ch in s:
            if ch == "«":
                if depth == 0:
                    op = i
                depth += 1
            elif ch == "»" and depth > 0:
                depth -= 1
                if depth == 0:
                    op = None
    return state


def mark(s, rx):
    return rx.sub(lambda m: f"[[{m.group(0)}]]", s) if "[[" not in s else s


def sentences(text):
    """Title + body sentences with absolute char spans (scrape format title///body)."""
    i = text.find("///"); title, b0 = (text[:i], i + 3) if i >= 0 else ("", 0)
    out = [(0, len(title))]; body = text[b0:]; st = 0
    for m in split.finditer(body):
        out.append((b0 + st, b0 + m.start())); st = m.end()
    out.append((b0 + st, len(text)))
    return out


def marked(text, a, b, spans):
    s, cur = [], a
    for x, y in sorted(spans):
        s += [text[cur:x], "[[", text[x:y], "]]"]; cur = y
    s.append(text[cur:b])
    return "".join(s).strip()


def in_target(a, spans, limit=700):
    """The spans (sorted) whose closing ]] falls within the first `limit` characters of marked(text, a, b, spans), i.e.
    those the scored target really marks (marked() strips leading whitespace: ignored, sentences start clean)."""
    out, cur, n = [], a, 0
    for x, y in sorted(spans):
        n += (x - cur) + 4 + (y - x); cur = y
        if n <= limit:
            out.append((x, y))
    return out


def build(text, rx, chain_spans):
    """Segments for one person in one article. rx: the person's name regex; chain_spans: (onset, offset) of all the
    mentions of their coreference chain. Returns dicts: mention (index), is_title, title, passage, kind, label_flag, in_quote,
    sent (index of the centre sentence; 0 = title) and spans (the chain mentions the target marks: chain spans that
    overlap a marked span; name-regex matches outside the chain are not in it)."""
    S = sentences(text); clean = [text[a:b].strip() for a, b in S]
    per_sent = {}
    for j, (a, b) in enumerate(S):
        sp = [(m.start() + a, m.end() + a) for m in rx.finditer(text[a:b])]
        for x, y in sorted(chain_spans, key=lambda t: t[1] - t[0]):          # shortest first; skip overlaps
            if a <= x < b and y <= b and not any(x < q and y > p for p, q in sp):
                sp.append((x, y))
        if sp:
            per_sent[j] = sp
    pos = sorted(per_sent); inq = open_quote(clean); out = []
    for j, i in enumerate(pos):
        a, b = S[i]; tgt, _ = cut_link(marked(text, a, b, per_sent[i])[:700])
        if not tgt or "[[" not in tgt:
            continue
        kinds = {("name" if rx.search(text[x:y]) else "coref") for x, y in per_sent[i]}
        if i == 0:
            seg, ttl, start = [tgt], "(this passage is the article title)", 0
        else:
            seg, start = [tgt], i
            for jj, before in ((i - 1, True), (i + 1, False)):
                if jj < 1 or jj >= len(clean):
                    continue
                others = [p for p in pos if p != i]
                if jj in others:
                    continue
                d = abs(jj - i)
                if any(abs(jj - p) < d or (abs(jj - p) == d and p < i) for p in others):
                    continue
                s = mark(cut_link(clean[jj])[0], rx)
                if s:
                    seg = [s] + seg if before else seg + [s]
                    if before:
                        start = jj
            ttl = clean[0]
            if inq[i] is not None and inq[i] < start:
                seg = [NOTE] + seg
        passage = " ".join(seg)
        near = any(abs(m.start() - n.start()) < 80 for n in re.finditer(r"\[\[", passage) for m in LABEL.finditer(passage))
        shown = in_target(a, per_sent[i])
        # a chain mention belongs to the passage of the sentence it starts in (one glued across the title///body
        # separator overlaps name matches on both sides)
        own = [(x, y) for x, y in chain_spans if a <= x < b and any(x < q and y > p for p, q in shown)]
        out.append(dict(mention=j, is_title=int(i == 0), title=ttl, passage=passage, kind="+".join(sorted(kinds)),
                        label_flag=int(near), in_quote=int(passage.startswith(NOTE)), sent=i, spans=sorted(own)))
    return out
