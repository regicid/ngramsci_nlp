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
    assert {(c, src) for c, _, _, src in people(text, ents, surnames=sn, year=2017, with_source=True)} == \
        {(0, "family_name"), (1, "chain"), (2, "article")}


def _ents(text, chains):
    """chains: [(chain id, surface, prop, number)] -> entities rows with onsets (first occurrence after the previous)."""
    rows, pos = [], 0
    for ch, s, prop, num in chains:
        a = text.index(s, pos); pos = a + 1
        rows.append({"cat": "PER", "COREF": ch, "prop": prop, "number": num, "onset": a, "offset": a + len(s)})
    return pd.DataFrame(rows)


def test_compound_family_names_and_titles():
    from ngramsci.stages.portrayal import Surnames, people
    counts = pd.DataFrame([("x", 1975, "Valéry Giscard d'Estaing", 200), ("x", 1975, "Giscard d'Estaing", 150),
                           ("x", 1975, "Anne-Aymone Giscard d'Estaing", 5), ("x", 1975, "Simone Veil", 80),
                           ("x", 1975, "Jacques Chirac", 300), ("x", 1975, "Jean-Luc Mélenchon", 50),
                           ("x", 1975, "Jean-Luc Godard", 40)], columns=["key", "year", "full_name", "n"])
    sn = Surnames.build(counts)
    assert "giscard d estaing" in sn.compounds and "luc melenchon" not in sn.compounds
    assert sn.get("giscard d estaing", 1975) == "Valéry Giscard d'Estaing" and sn.get("chirac", 1975) == "Jacques Chirac"
    text = "Budget///M. Giscard d'Estaing parle. Giscard d'Estaing insiste. Mme Simone Veil répond. Mme Veil sourit. Mme Chirac aussi."
    ents = _ents(text, [(0, "M. Giscard d'Estaing", "PROP", "Singular"), (1, "Giscard d'Estaing", "PROP", "Singular"),
                        (2, "Mme Simone Veil", "PROP", "Singular"), (3, "Mme Veil", "PROP", "Singular"),
                        (4, "Mme Chirac", "PROP", "Singular")])
    got = {c: (n, src) for c, n, _, src in people(text, ents, surnames=sn, year=1975, with_source=True)}
    assert got[0] == got[1] == ("Valéry Giscard d'Estaing", "family_name")
    assert got[2] == ("Simone Veil", "chain") and got[3] == ("Simone Veil", "article")
    assert 4 not in got                                            # « Mme Chirac » is not the year's dominant Chirac


def test_lula_mononyms_and_first_name_guard(tmp_path):
    """« Lula »: a later word of the article's full name, the alias table, or the year's table; single names (« Madonna »);
    « Philippe » next to a « Philippe Martinez » stays unnamed; particles (da, von der, al-)."""
    import json
    from ngramsci.nlp.segments import NAMEOK
    from ngramsci.stages.portrayal import Surnames, people
    assert all(NAMEOK.match(n) for n in ["Luiz Inácio Lula da Silva", "Ursula von der Leyen", "Bachar al-Assad"])
    counts = pd.DataFrame([(2022, "Luiz Inácio Lula da Silva", "full", 60), (2022, "Lula da Silva", "full", 30),
                           (2022, "Lula", "bare", 400), (2022, "Madonna", "bare", 50), (2022, "Dieu", "bare", 90),
                           (2022, "José Luis Rodríguez Zapatero", "full", 40), (2022, "Luis Fernandez", "full", 30),
                           (2022, "Edouard Philippe", "full", 300), (2022, "Philippe Martinez", "full", 60),
                           (2022, "Silva", "bare", 30), (2022, "Marina Silva", "full", 25),
                           (2022, "Carrefour", "bare", 40), (2022, "Carrefour", "other", 300),
                           (2022, "Hitler", "bare", 200), (2022, "Adolf Hitler", "full", 8), (2022, "Adolf Martin", "full", 30),
                           (2022, "RFI", "bare", 50), (2022, "Neymar", "bare", 80), (2022, "Agent Neymar", "full", 4)],
                          columns=["year", "full_name", "kind", "n"])
    al = {"lula da silva": "Luiz Inácio Lula da Silva", "lula": "Luiz Inácio Lula da Silva",
          "luiz inacio lula da silva": "Luiz Inácio Lula da Silva"}
    sn = Surnames.build(counts, aliases=al)
    assert sn.get("lula", 2022) == "Luiz Inácio Lula da Silva" and sn.get("zapatero", 2022) == "José Luis Rodríguez Zapatero"
    assert sn.get("madonna", 2022) == "Madonna" and sn.get("dieu", 2022) is None
    assert sn.get("carrefour", 2022) is None and sn.get("hitler", 2022) == "Adolf Hitler"
    assert sn.get("rfi", 2022) is None and sn.get("neymar", 2022) == "Neymar"
    assert sn.get("luis", 2022) is None and sn.get("philippe", 2022) is None and sn.get("silva", 2022) is None
    assert "philippe" in sn.firstnames and "lula" not in sn.firstnames
    p = tmp_path / "surnames.json"; sn.to_json(str(p))
    json.dump({"Luiz Inácio Lula da Silva": ["Lula", "Lula da Silva"]}, open(tmp_path / "aliases.json", "w"))
    sj = Surnames.from_json(str(p))
    assert sj.canon("Lula da Silva") == "Luiz Inácio Lula da Silva" and sj.firstnames == sn.firstnames

    text = "Brésil///Luiz Inácio Lula da Silva arrive. Lula parle. Luis sourit. Madonna chante."
    ents = _ents(text, [(0, "Luiz Inácio Lula da Silva", "PROP", "Singular"), (1, "Lula", "PROP", "Singular"),
                        (2, "Luis", "PROP", "Singular"), (3, "Madonna", "PROP", "Singular")])
    got = {c: (n, src) for c, n, _, src in people(text, ents, surnames=sj, year=2022, with_source=True)}
    assert got[1] == ("Luiz Inácio Lula da Silva", "article") and 2 not in got and got[3] == ("Madonna", "family_name")
    got = {c: (n, src) for c, n, _, src in people("Brésil///Lula parle.", _ents("Brésil///Lula parle.", [(0, "Lula", "PROP", "Singular")]),
                                                  surnames=sj, year=2015, with_source=True)}
    assert got == {0: ("Luiz Inácio Lula da Silva", "alias")}
    from ngramsci.stages.portrayal import name_rows
    text = "Bourse///Madonna chante. Nike recule. Le groupe américain licencie. Madonna rit. L'entraîneur du club sourit."
    ents = _ents(text, [(0, "Madonna", "PROP", "Singular"), (1, "Nike", "PROP", "Singular"),
                        (1, "Le groupe américain", "NOM", "Singular"), (0, "Madonna", "PROP", "Singular"),
                        (0, "L'entraîneur du club", "NOM", "Singular")])
    assert name_rows(text, ents) == {("Madonna", "bare"), ("Nike", "other")}
    sn2 = Surnames({"nike": {2022: "Nike"}, "madonna": {2022: "Madonna"}})
    assert {n for _, n, _ in people(text, ents, surnames=sn2, year=2022)} == {"Madonna"}
    text = "Grève///Edouard Philippe reçoit Philippe Martinez. Philippe répond. M. Philippe insiste."
    ents = _ents(text, [(0, "Edouard Philippe", "PROP", "Singular"), (1, "Philippe Martinez", "PROP", "Singular"),
                        (2, "Philippe", "PROP", "Singular"), (3, "M. Philippe", "PROP", "Singular")])
    got = {c: n for c, n, _ in people(text, ents, surnames=sn, year=2022)}
    assert 2 not in got and got[3] == "Edouard Philippe"


def test_apostrophes_elisions_and_dieudonne():
    """Apostrophes unified (« d‘Estaing » = « d'Estaing », one key); « L'équipe de France » is no name; « Dieudonné », a
    single name more often than a first name, resolves, and next to one other Dieudonné is that person."""
    from ngramsci.explorer import name_key
    from ngramsci.stages.portrayal import Surnames, full_name, people
    assert name_key("Valéry Giscard d’Estaing") == name_key("Valéry Giscard d‘Estaing") == name_key("Valéry Giscard d'Estaing") \
        == "valery giscard d estaing"
    assert full_name(["L’équipe de France"]) is None and full_name(["L'administration Trump"]) is None
    text = "Paris///Valéry Giscard d‘Estaing arrive. Valéry Giscard d’Estaing parle."
    ents = _ents(text, [(0, "Valéry Giscard d‘Estaing", "PROP", "Singular"), (0, "Valéry Giscard d’Estaing", "PROP", "Singular")])
    assert [n for _, n, _ in people(text, ents)] == ["Valéry Giscard d'Estaing"]
    counts = pd.DataFrame([(2014, "Dieudonné", "bare", 200), (2014, "Dieudonné M'Bala M'Bala", "full", 30),
                           (2014, "Dieudonné Niangouna", "full", 20), (2014, "Dieudonné Sinon", "full", 20),
                           (2014, "Philippe", "bare", 50), (2014, "Philippe Martinez", "full", 300)],
                          columns=["year", "full_name", "kind", "n"])
    al = {"dieudonne": "Dieudonné M'Bala M'Bala", "dieudonne m bala m bala": "Dieudonné M'Bala M'Bala"}
    sn = Surnames.build(counts, aliases=al)
    assert sn.get("dieudonne", 2014) == "Dieudonné M'Bala M'Bala" and sn.get("philippe", 2014) is None
    for text, want in [("Humour///Dieudonné joue.", "Dieudonné M'Bala M'Bala"),
                       ("Humour///Dieudonné M’Bala M’Bala arrive. Dieudonné joue.", "Dieudonné M'Bala M'Bala"),
                       ("Théâtre///Dieudonné Niangouna arrive. Dieudonné joue.", "Dieudonné Niangouna")]:
        mentions = [(0, "Dieudonné M’Bala M’Bala", "PROP", "Singular")] if "M’Bala" in text else \
                   [(0, "Dieudonné Niangouna", "PROP", "Singular")] if "Niangouna" in text else []
        ents = _ents(text, mentions + [(1, "Dieudonné", "PROP", "Singular")])
        got = {c: n for c, n, _ in people(text, ents, surnames=sn, year=2014)}
        assert got[1] == want, (text, got)


def test_compound_found_from_bare_counts():
    """Counts taken with a compound list hold « Giscard d'Estaing » alone as a bare name, not a full one: still a compound."""
    from ngramsci.stages.portrayal import Surnames
    counts = pd.DataFrame([(1975, "Valéry Giscard d'Estaing", "full", 200), (1975, "Giscard d'Estaing", "full", 5),
                           (1975, "Giscard d'Estaing", "bare", 400), (1975, "Giscard Samba", "full", 30)],
                          columns=["year", "full_name", "kind", "n"])
    assert "giscard d estaing" in Surnames.build(counts).compounds


def test_surnames_json_formats(tmp_path):
    import json
    from ngramsci.stages.portrayal import Surnames
    p = tmp_path / "s.json"; json.dump({"veil": {"1975": "Simone Veil"}}, open(p, "w"))       # 2026-11-22 format
    assert Surnames.from_json(str(p)).get("veil", 1975) == "Simone Veil"
    Surnames({"veil": {1975: "Simone Veil"}}, ["giscard d estaing"]).to_json(str(p))
    s = Surnames.from_json(str(p)); assert s.compounds == {"giscard d estaing"} and s.get("veil", 1975) == "Simone Veil"


def test_score_all_name_free():
    """score_all: every singular chain is scored, named or not; the target is the chain's own proper-name mentions."""
    import types
    import ngramsci.stages.portrayal as P
    text = "Réunion///Le ministre arrive. Il parle à Paul Martin. Les manifestants crient. Ils attendent."
    ents = _ents(text, [(0, "Le ministre", "NOM", "Singular"), (0, "Il", "PRON", "Singular"), (1, "Paul Martin", "PROP", "Singular"),
                        (2, "Les manifestants", "NOM", "Plural"), (2, "Ils", "PRON", "Plural")])
    seen = []
    def score(rows, batch=64):
        seen.extend(rows); return [{"pertinent": 1.0, "journaliste": 0.0, "valence": [0, 1, 0], "extremisme": [0, 1, 0], "dynamique": [0, 1, 0]}] * len(rows)
    docs = pd.DataFrame({"doc_id": ["d"], "text": [text], "date": ["2020-01-01"]}).set_index("doc_id")
    orig = P.load_inputs
    P.load_inputs = lambda *a, **k: (docs, ents.assign(doc_id="d"))
    try:
        ctx = types.SimpleNamespace(p=lambda st, key, d=None: {"score_all": True}.get(key, d))
        import tempfile, os
        out = os.path.join(tempfile.mkdtemp(), "p.parquet")
        P.run(0, {"text": "", "mentions": "", "entities": ""}, {"portrayal": out}, types.SimpleNamespace(score=score, revision="stub"), ctx)
        df = pd.read_parquet(out)
    finally:
        P.load_inputs = orig
    assert set(df.chain) == {0, 1}                                  # the plural chain is not scored
    assert df[df.chain == 0].full_name.isna().all() and (df[df.chain == 1].full_name == "Paul Martin").all()
    assert all("pattern" in r and "full_name" not in r for r in seen)
    assert re.compile(seen[-1]["pattern"]).search("Paul Martin") and not re.compile(next(r for r in seen if "ministre" in r["passage"])["pattern"]).search("Le ministre")


def _quote_case():
    text = ("Titre///Claire Dubois arrive à Lyon. « Le budget est   prêt », a-t-elle déclaré. "
            "Paul Martin l’approuve. Il estime que « la ville peut investir ».")
    spans = {"Claire Dubois": text.find("Claire"), "elle": text.find("elle"), "Paul Martin": text.find("Paul"),
             "Il": text.find("Il "), "Lyon": text.find("Lyon")}
    ents = pd.DataFrame([dict(onset=a, offset=a + len(t), text=t, cat="LOC" if t == "Lyon" else "PER",
                              prop="PROP" if t[0].isupper() and t != "Il" else "PRON",
                              COREF={"Claire Dubois": 0, "elle": 0, "Paul Martin": 1, "Il": 1}.get(t))
                         for t, a in spans.items()]).sort_values("onset").reset_index(drop=True)
    chars = pd.DataFrame([dict(char_id=0, number="Singular", gender_firm="Female", gender_score=0.0, gender_conflict=False),
                          dict(char_id=1, number="Singular", gender_firm="Male", gender_score=1.0, gender_conflict=False)])
    quotes = pd.DataFrame([dict(unit=0, content="Le budget est prêt", source="elle"),
                           dict(unit=0, content="la ville peut investir", source="il"),       # case differs from the text
                           dict(unit=0, content="introuvable", source="Paul Martin"),
                           dict(unit=0, content="la ville peut investir", source="Jacques")])
    return text, quotes, ents, chars


def test_quote_links():
    from ngramsci.stages.quote_links import link_doc, norm
    assert norm("L’ assemblée\n\n a dit")[0] == "l'assemblée a dit"
    text, quotes, ents, chars = _quote_case()
    r = link_doc(text, quotes, ents, chars)
    assert text[r[0]["q_start"]:r[0]["q_end"]] == "Le budget est   prêt"
    assert r[0]["mention"] == "elle" and r[0]["chain"] == 0 and r[0]["chain_gender"] == "Female"
    assert r[0]["chain_name"] == "Claire Dubois" and r[0]["chain_size"] == 2 and r[0]["link"] == "unit"
    assert r[1]["mention"] == "Il" and r[1]["chain_name"] == "Paul Martin" and r[1]["chain_gender"] == "Male"
    assert r[2]["q_start"] == -1 and r[2]["mention"] == "Paul Martin" and r[2]["chain"] == 1   # quote lost, speaker found
    assert r[3]["link"] == "" and r[3]["mention"] is None and r[3]["chain"] is None


def test_locate_rebuilds_sentences_and_mentions():
    """release.locate gives each passage its centre sentence and the chain mentions it scores, the same as the scorer
    writes from 2026-11-23 on; a possessive-only sentence (excluded upstream) has no passage."""
    import types
    import ngramsci.stages.portrayal as P
    from ngramsci import release as Rl
    text = "Réunion///Paul Martin arrive. Il parle, puis il repart. La foule crie. Martin sourit."
    ents = _ents(text, [(0, "Paul Martin", "PROP", "Singular"), (0, "Il", "PRON", "Singular"), (0, "il", "PRON", "Singular"),
                        (0, "Martin", "PROP", "Singular")])
    ents = ents.assign(doc_id="d", start_token=range(0, 4 * len(ents), 4), end_token=range(1, 4 * len(ents), 4))
    score = lambda rows, batch=64: [{"pertinent": 1.0, "journaliste": 0.0, "valence": [0, 1, 0], "extremisme": [0, 1, 0], "dynamique": [0, 1, 0]}] * len(rows)
    docs = pd.DataFrame({"doc_id": ["d"], "text": [text], "date": ["2020-01-01"]}).set_index("doc_id")
    orig = P.load_inputs
    P.load_inputs = lambda *a, **k: (docs, ents)
    try:
        import tempfile, os
        out = os.path.join(tempfile.mkdtemp(), "p.parquet")
        P.run(0, {"text": "", "mentions": "", "entities": ""}, {"portrayal": out}, types.SimpleNamespace(score=score, revision="stub"),
              types.SimpleNamespace(p=lambda st, key, d=None: d))
        df = pd.read_parquet(out)
    finally:
        P.load_inputs = orig
    assert list(df["sent"]) == [1, 2, 4]                       # sentence 3 (« La foule crie ») does not mention him
    assert [len(m) for m in df["mentions"]] == [1, 2, 1]        # « Il … il » share one passage
    again = Rl.locate(df.drop(columns=["sent", "mentions"]), docs, ents, mode="name")
    assert Rl.locate.mismatch == 0 and list(again["sent"]) == list(df["sent"])
    assert [sorted(map(list, m)) for m in again["mentions"]] == [sorted(map(list, m)) for m in df["mentions"]]
    e = Rl.entities(ents, pd.Series({"d": "2020-01-01"}), port=again)
    assert e["p_pertinent"].notna().all() and len(e) == len(ents)
    assert list(e["portrayal_sent"]) == [1, 2, 2, 4]           # « Il … il »: one passage, two mention rows
    c = Rl.per_character(again, passages=False)
    assert "portrayal" not in c and int(c["portrayal_n"].iloc[0]) == 3
