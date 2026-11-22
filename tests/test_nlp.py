"""Pure-Python NLP pieces (no models): windows, coreference linking, quote assembly, portrayal segments."""
import re

import pandas as pd

from ngramsci.nlp.coref import coref_ids
from ngramsci.nlp.segments import build, name_re
from ngramsci.nlp.windows import build_windows, segments, window_input, windows
from ngramsci.stages.quotes import to_quotes

TEXT = ("Claire Dubois arrive à Lyon. Elle rencontre Paul Martin. Il la salue. "
        "Le maire de Lyon parle ensuite longuement du budget de la ville.")


def toks(text):
    spans = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
    return spans, [text[a:b] for a, b in spans]


def test_window_markup():
    inp, ids = window_input(TEXT, 0, len(TEXT), [(0, 0, 13), (1, 29, 33), (2, 44, 55)])
    assert inp.startswith("{___ Claire Dubois}") and "{___ Elle}" in inp and "{___ Paul Martin}" in inp
    assert ids == [0, 1, 2]


def test_segments_never_cut_a_mention():
    spans, words = toks(TEXT)
    segs = segments(TEXT, spans, words, [(0, 1)], seg_chars=10)
    assert all(not (a < spans[0][1] < b and a > spans[0][0]) for a, b in segs)
    assert segs[0][0] == 0 and segs[-1][1] == len(TEXT)
    assert windows(segs, k=2)[0][0] == 0


def test_build_windows_only_per():
    spans, words = toks(TEXT)
    m = pd.DataFrame({"mid": [0, 1, 2], "start_token": [0, 6, 8], "end_token": [1, 6, 9], "cat": ["PER", "PER", "LOC"],
                      "onset": [0, 29, 44], "offset": [13, 33, 54]})
    w = build_windows("d", TEXT, spans, words, m)
    assert len(w) == 1 and w[0]["mention_ids"] == [0, 1]


def test_coref_ids_links_windows_and_names():
    wins = [{"w": 0, "input": "", "mention_ids": [0, 1, 2]}, {"w": 1, "input": "", "mention_ids": [2, 3]}]
    outs = {0: "a", 1: "b"}
    local = {"a": [0, 0, 1], "b": [5, 5]}                     # window 1: mention 3 joins mention 2's cluster
    res, bad = coref_ids([0, 1, 2, 3, 4], wins, outs, {m: "x" for m in range(5)}, {m: "PRON" for m in range(5)},
                         lambda inp, o: local[o])
    assert bad == 0 and res[0] == res[1] and res[2] == res[3] and res[0] != res[2] and res[4] not in (res[0], res[2])
    res2, _ = coref_ids([0, 1], [], {}, {0: "Paul Martin", 1: "Paul Martin"}, {0: "PROP", 1: "PROP"}, None)
    assert res2[0] == res2[1]                                  # identical proper names merged


def test_to_quotes():
    rels = [{"head": "Paul Martin", "head_type": "<per>", "tail": "« Nous gagnerons »", "tail_type": "<dirquot>", "type": "cité"},
            {"head": "a déclaré", "head_type": "<cue>", "tail": "« Nous gagnerons »", "tail_type": "<dirquot>", "type": "indique"}]
    q = to_quotes(rels)
    assert q == [{"content": "« Nous gagnerons »", "quote_type": "<dirquot>", "cue": "a déclaré", "source": "Paul Martin", "source_type": "<per>"}]


def test_portrayal_segments():
    text = "Claire Dubois en campagne///Claire Dubois arrive à Lyon. Elle rencontre Paul Martin. La foule applaudit."
    rx = name_re("Claire Dubois")
    assert rx.fullmatch("Mme Dubois") and rx.fullmatch("Dubois") and not rx.fullmatch("")
    i = text.index("Elle")
    segs = build(text, rx, [(i, i + 4)])
    assert segs[0]["is_title"] == 1
    body = [s for s in segs if not s["is_title"]]
    assert body[0]["passage"] == "[[Claire Dubois]] arrive à Lyon."     # next sentence is another mention's
    assert body[1]["passage"] == "[[Elle]] rencontre Paul Martin. La foule applaudit."   # free neighbour -> closest mention
    assert body[1]["kind"] == "coref" and body[0]["kind"] == "name"


def test_portrayal_targets():
    from ngramsci.stages.portrayal import full_re, people, targeted
    text = "Le Pen en campagne///Marine Le Pen arrive à Lyon. Elle rencontre Paul Martin. Mme Le Pen sourit."
    on = lambda s: text.index(s)
    ents = pd.DataFrame({"cat": ["PER"] * 4, "COREF": [0, 0, 1, 0], "prop": ["PROP", "PRON", "PROP", "PROP"],
                         "onset": [on("Marine"), on("Elle"), on("Paul"), on("Mme")],
                         "offset": [on("Marine") + 13, on("Elle") + 4, on("Paul") + 11, on("Mme") + 10]})
    assert sorted(n for _, n, _ in people(text, ents)) == ["Marine Le Pen", "Paul Martin"]      # everyone
    t = targeted(text, ents, [("Marine Le Pen", name_re("Marine Le Pen"), full_re("Marine Le Pen"))])
    assert [(c, n, len(s)) for c, n, s in t] == [(0, "Marine Le Pen", 3)]                     # only her, whole chain
    assert targeted(text.replace("Marine Le Pen arrive", "Elle arrive"), ents, [("Marine Le Pen", name_re("Marine Le Pen"),
                    full_re("Marine Le Pen"))]) == []                                         # never named in full


def test_rebel_parse():
    from ngramsci.nlp.rebel_parse import parse
    out = parse("<s><triplet> « Nous gagnerons »<dirquot> Paul Martin<per> cité<triplet> « Nous gagnerons »<dirquot> a déclaré<cue> indique</s>")
    assert out == [{"head": "Paul Martin", "head_type": "<per>", "tail": "« Nous gagnerons »", "tail_type": "<dirquot>", "type": "cité"},
                   {"head": "a déclaré", "head_type": "<cue>", "tail": "« Nous gagnerons »", "tail_type": "<dirquot>", "type": "indique"}]
    assert to_quotes(out)[0]["cue"] == "a déclaré"
    # a short group keeps the relation word in tail_type (reference parser's convention)
    assert parse("<triplet> elle viendra<indquot> indique") == [
        {"head": "", "head_type": "<indquot>", "tail": "elle viendra", "tail_type": "indique", "type": ""}]
    assert parse("<s></s><pad>") == []
    assert parse("<triplet> rien") == [{"head": "", "head_type": "", "tail": "rien", "tail_type": "", "type": ""}]


def test_quote_units():
    from ngramsci.stages.quotes import units
    text = "Titre///" + " ".join(f"Phrase numéro {i} assez longue pour remplir." for i in range(100))
    us = units(text, split="chunks", max_chars=300)
    assert all(len(t) <= 300 for _, t in us) and len(us) > 10
    assert " ".join(t for _, t in us).replace("\n", " ").split() == text.replace("///", " ").split()   # nothing lost
    assert units(text, split="truncate") == [(0, text.replace("///", "\n", 1))]


def test_portrayal_family_names():
    """« Mélenchon » alone: the article's « Jean-Luc Mélenchon », else the person the name stands for that year; Le Pen never."""
    from ngramsci.stages.portrayal import Surnames, bare_surname, people
    counts = pd.DataFrame([("melenchon", 2017, "Jean-Luc Mélenchon", 95), ("melenchon", 2017, "Sophie Mélenchon", 2),
                           ("martin", 2017, "Paul Martin", 30), ("martin", 2017, "Jean Martin", 25),
                           ("le pen", 2017, "Marine Le Pen", 500), ("philippe", 2017, "Edouard Philippe", 300),
                           ("martinez", 2017, "Philippe Martinez", 60),
                           ("de gaulle", 1960, "Charles de Gaulle", 90), ("de gaulle", 1960, "Général de Gaulle", 8),
                           ("de gaulle", 1960, "Pierre de Gaulle", 3)], columns=["key", "year", "full_name", "n"])
    sn = Surnames.build(counts)
    assert sn.get("melenchon", 2017) == sn.get("melenchon", 2018) == "Jean-Luc Mélenchon"       # ±1 year window
    assert sn.get("melenchon", 2020) is None and sn.get("martin", 2017) is None and sn.get("le pen", 2017) is None
    assert sn.get("philippe", 2017) is None and sn.get("martinez", 2017) == "Philippe Martinez"   # also a first name
    assert bare_surname(["de Gaulle", "le général"]) == "de Gaulle" and sn.get("de gaulle", 1960) == "Charles de Gaulle"
    assert bare_surname(["M. Mélenchon", "Mélenchon", "le leader insoumis"]) == "Mélenchon"
    text = "Mélenchon à Lyon///M. Mélenchon parle. Il rencontre Paul Martin. Martin répond."
    on = lambda s: text.index(s)
    ents = pd.DataFrame({"cat": ["PER"] * 5, "COREF": [0, 0, 0, 1, 2], "prop": ["PROP", "PROP", "PRON", "PROP", "PROP"],
                         "onset": [on("Mélenchon"), on("M. Mélenchon"), on("Il"), on("Paul"), on("Martin répond")],
                         "offset": [on("Mélenchon") + 9, on("M. Mélenchon") + 12, on("Il") + 2, on("Paul") + 11, on("Martin répond") + 6]})
    got = {(c, n) for c, n, _ in people(text, ents, surnames=sn, year=2017)}
    assert got == {(0, "Jean-Luc Mélenchon"), (1, "Paul Martin"), (2, "Paul Martin")}          # 2: the article's Martin
    assert {n for _, n, _ in people(text, ents)} == {"Paul Martin"}                             # old behaviour without table
    assert {c for c, _, _ in people(text, ents, surnames=sn, year=2017, only_surname=True)} == {0, 2}
