"""Extended attribute rules (v3), kept separate from propp's own rules.

Input: a propp token table (after extract_attributes: has char_att_agent/patient/mod/poss) and the entities table
(head_id of each mention). Output: per mention head, new attribute words by slot with the rule that produced them,
plus the agents propp attached wrongly to passive subjects (to drop).

Rules (user's spec, Nov 2026):
 1 pass_subj   nsubj:pass, or nsubj of a verb with an aux:pass child -> patient (+ that verb in agent_drop)
 2 pass_agent  obl:agent -> agent
 3 control     subject of V, V has an xcomp VERB child, V not causative/perception, V has no object -> agent
 4 conj        a mention head that is conj of another mention head inherits its attributes (old + new)
 5 participle  VERB participle (VerbForm=Part) with dep acl/amod attached to the mention -> mod
 6 attrib_verb subject of rester/sembler/paraître/devenir/demeurer + xcomp ADJ/NOUN/participle -> mod
 7 elided_de   mention is nmod of a NOUN with a case child "d’" (curly apostrophe, lemmatised "d’") -> poss
 8 dont        "dont" depends on a NOUN inside a relative clause attached to the mention -> poss
 9 poss_verb   subject of disposer/posséder/détenir/jouir + its object (obj/obl:arg/obl NOUN) -> poss
10 oblique     mention is iobj or obl:arg of V (not sembler/paraître) -> obl, stored as "verb+preposition"
"""
from collections import defaultdict

CAUSATIVE = {"faire", "laisser", "voir", "entendre", "regarder", "sentir"}
ATTRIB = {"rester", "sembler", "paraître", "devenir", "demeurer"}
POSSV = {"disposer", "posséder", "détenir", "jouir"}
NO_OBL = {"sembler", "paraître"}
SLOTS = ("agent", "patient", "mod", "poss")


def _morph(m, key):
    m = str(m or "")
    for kv in m.split("|"):
        if kv.startswith(key + "="):
            return kv.split("=", 1)[1]
    return None


def extended(tdf, head_ids):
    """tdf: token table indexed by position with token_ID_within_document == position (propp convention).
    head_ids: iterable of mention head token ids. Returns {head: {slot: [ {w, i, r} ]}} and {head: [ {w, i} ]} drops."""
    tid = tdf["token_ID_within_document"].astype(int).tolist()
    lem = tdf["lemma"].astype(str).str.lower().tolist(); word = tdf["word"].astype(str).str.lower().tolist()
    pos = tdf["POS_tag"].astype(str).tolist(); dep = tdf["dependency_relation"].astype(str).tolist()
    head = tdf["syntactic_head_ID"].astype(int).tolist(); morph = tdf["morph"].astype(str).tolist() if "morph" in tdf else [""] * len(tid)
    idx = {t: k for k, t in enumerate(tid)}
    kids = defaultdict(list)
    for k, h in enumerate(head):
        if h != tid[k]:
            kids[h].append(k)
    def ch(t, deps=None):
        return [k for k in kids.get(t, []) if deps is None or dep[k] in deps]
    heads = {int(h) for h in head_ids if h is not None and str(h) not in ("", "nan") and int(h) in idx}
    new = defaultdict(lambda: defaultdict(list)); drop = defaultdict(list)
    def add(h, slot, k, rule, w=None):
        new[h][slot].append({"w": w if w is not None else lem[k], "i": int(tid[k]), "r": rule})
    for h in heads:
        k = idx[h]; d = dep[k]; v = idx.get(head[k])
        # 1 passive subject
        if v is not None and pos[v] in ("VERB", "AUX"):
            if d == "nsubj:pass":
                add(h, "patient", v, "pass_subj")
            elif d == "nsubj" and ch(tid[v], {"aux:pass"}):
                add(h, "patient", v, "pass_subj"); drop[h].append({"w": lem[v], "i": int(tid[v])})
        # 2 passive agent
        if d == "obl:agent" and v is not None and pos[v] == "VERB":
            add(h, "agent", v, "pass_agent")
        # 3 control verbs, 6 attributive verbs, 9 possession verbs (all need the mention as subject of v)
        if d in ("nsubj", "nsubj:pass") and v is not None:
            has_obj = bool(ch(tid[v], {"obj", "iobj"}))
            for x in ch(tid[v], {"xcomp"}):
                if pos[x] == "VERB" and _morph(morph[x], "VerbForm") == "Inf" and lem[v] not in CAUSATIVE and not has_obj:
                    add(h, "agent", x, "control")
                if lem[v] in ATTRIB and (pos[x] in ("ADJ", "NOUN") or _morph(morph[x], "VerbForm") == "Part"):
                    add(h, "mod", x, "attrib_verb")
            if lem[v] in POSSV:
                for x in ch(tid[v], {"obj", "obl:arg", "obl"}):
                    if pos[x] == "NOUN":
                        add(h, "poss", x, "poss_verb")
        # 5 participle modifiers
        for x in ch(h, {"acl", "amod"}):
            if pos[x] == "VERB" and _morph(morph[x], "VerbForm") == "Part":
                add(h, "mod", x, "participle")
        # 7 elided d’ (possession)
        if d == "nmod" and v is not None and pos[v] == "NOUN" and any(word[c] == "d’" for c in ch(h, {"case"})):
            add(h, "poss", v, "elided_de")
        # 8 dont
        for r in ch(h, {"acl:relcl"}):
            stack = [r]; seen = set()
            while stack:
                y = stack.pop()
                if y in seen: continue
                seen.add(y)
                for c in kids.get(tid[y], []):
                    if lem[c] == "dont" and pos[idx.get(head[c], c)] == "NOUN":
                        add(h, "poss", idx[head[c]], "dont")
                    if dep[c] != "acl:relcl":
                        stack.append(c)
        # 10 oblique
        if d in ("iobj", "obl:arg") and v is not None and pos[v] == "VERB" and lem[v] not in NO_OBL:
            case = [lem[c] for c in ch(h, {"case"})]
            add(h, "obl", v, "oblique", w=f"{lem[v]}+{case[0] if case else 'à'}")
    # 4 coordination: inherit the first conjunct's attributes (propp's + new)
    old = defaultdict(lambda: defaultdict(list))
    for slot in SLOTS:
        col = f"char_att_{slot}"
        if col in tdf:
            for k, a in enumerate(tdf[col].tolist()):
                if a is not None and str(a) not in ("", "nan", "-1") and int(a) in heads:
                    old[int(a)][slot].append({"w": lem[k], "i": int(tid[k])})
    for h in heads:
        k = idx[h]
        if dep[k] == "conj" and head[k] in heads:
            h1 = head[k]
            for slot in SLOTS:
                have = {a["i"] for a in old[h][slot]} | {a["i"] for a in new[h][slot]}
                for a in old[h1][slot] + [x for x in new[h1][slot] if x["r"] != "conj"]:
                    if a["i"] not in have:
                        new[h][slot].append({"w": a["w"], "i": a["i"], "r": "conj"}); have.add(a["i"])
    return new, drop
