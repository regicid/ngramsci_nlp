"""Small additions to propp-fr 0.1.0 (PyPI, MIT licence, https://github.com/lattice-8094/propp) that this pipeline needs.

* tokens_table(doc): vectorised version of propp's generate_tokens_df_from_spacy_doc (same columns and values),
  from the propp-fr development copy we ran on the clusters.
* load_ner_models(): the mention-detection models without propp's load_models (which also loads spaCy, the propp
  coreference model, and may reinstall spaCy for CUDA), with a CPU-safe unpickler: the published pickles hold CUDA
  tensors and fail to load on machines without a GPU.
"""
import io
import pickle
import types

import numpy as np
import pandas as pd
import torch

NER_MODEL = "AntoineBourgois/propp-fr_NER_camembert-large_FAC_GPE_LOC_PER_TIME_VEH"


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_ner_models(name=NER_MODEL):
    from propp_fr import propp_fr_mentions_detection_module as mdm
    if not torch.cuda.is_available():
        mdm.pickle = types.SimpleNamespace(loads=lambda b: _CPUUnpickler(io.BytesIO(b)).load(),
                                           load=lambda f: _CPUUnpickler(f).load(), dump=pickle.dump,
                                           UnpicklingError=pickle.UnpicklingError)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(name, "final_model.pkl")         # HF cache (works offline), not propp's ./<repo>/ copy
    return mdm.load_mentions_detection_model(path)


def to_str_columns(t: pd.DataFrame) -> pd.DataFrame:
    """spaCy MorphAnalysis objects -> str, so the table can be written to parquet."""
    for c in t.columns:
        if t[c].dtype == object and len(t) and not isinstance(t[c].iloc[0], (str, type(None))):
            t[c] = t[c].astype(str)
    return t


def tokens_table(doc):
    """
    Version optimisée de generate_tokens_df_from_spacy_doc.
    - Aucune libération mémoire (gc.collect / torch.cuda.empty_cache).
    - Une seule passe d'extraction sur le doc spaCy.
    - paragraph_ID / sentence_ID / token_ID_within_sentence calculés en numpy.
    - Seule reste une petite boucle Python sur des listes pour la récurrence
      sur is_sent_start (dépend de la valeur finale du token précédent).
    """
    n = len(doc)
    columns = ['paragraph_ID', 'sentence_ID', 'token_ID_within_sentence',
               'token_ID_within_document', 'word', 'lemma',
               'byte_onset', 'byte_offset', 'POS_tag',
               'dependency_relation', 'syntactic_head_ID', 'morph']
    if n == 0:
        return pd.DataFrame(columns=columns)

    # ---- 1. Extraction d'attributs en une seule passe ----
    words           = [None] * n
    lemmas          = [None] * n
    pos_tags        = [None] * n
    morphs          = [None] * n
    dep_rels        = [None] * n
    token_ids       = np.empty(n, dtype=np.int64)
    byte_onsets     = np.empty(n, dtype=np.int64)
    byte_offsets    = np.empty(n, dtype=np.int64)
    head_ids        = np.empty(n, dtype=np.int64)
    is_newline      = np.empty(n, dtype=bool)
    is_title        = np.empty(n, dtype=bool)
    is_punct_spacy  = np.empty(n, dtype=bool)
    is_sent_start_s = np.empty(n, dtype=bool)

    for i, t in enumerate(doc):
        text = t.text
        words[i]           = text
        lemmas[i]          = t.lemma_
        pos_tags[i]        = t.pos_
        morphs[i]          = t.morph
        dep_rels[i]        = t.dep_
        token_ids[i]       = t.i
        byte_onsets[i]     = t.idx
        byte_offsets[i]    = t.idx + len(text)
        head_ids[i]        = t.head.i
        is_newline[i]      = '\n' in t.text_with_ws
        is_title[i]        = t.is_title
        is_punct_spacy[i]  = t.is_punct
        is_sent_start_s[i] = bool(t.is_sent_start)

    # ---- 2. Masques booléens vectorisés ----
    is_punct_terminal = np.fromiter(
        (w in ('.', '!', '?') for w in words), dtype=bool, count=n
    )

    # tableaux "précédents" via shift
    prev_newline = np.empty(n, dtype=bool)
    prev_newline[0]  = True              # = previous_is_newline_char initial
    prev_newline[1:] = is_newline[:-1]

    prev_punct = np.empty(n, dtype=bool)
    prev_punct[0]  = False               # = previous_is_punct initial
    prev_punct[1:] = is_punct_terminal[:-1]

    # composante non-récursive de la condition d'override
    # cond_b[i] = (is_title[i] or is_punct[i]) and is_sent_start_spacy[i] and prev_punct[i]
    cond_b = (is_title | is_punct_spacy) & is_sent_start_s & prev_punct

    # ---- 3. Récurrence : f[i] = prev_newline[i] OR (cond_b[i] AND NOT f[i-1]) ----
    pn  = prev_newline.tolist()
    cb  = cond_b.tolist()
    fss = [False] * n
    prev_f = False
    for i in range(n):
        if pn[i] or (cb[i] and not prev_f):
            fss[i] = True
            prev_f = True
        else:
            prev_f = False
    final_is_sent_start = np.asarray(fss, dtype=bool)

    # ---- 4. paragraph_ID / sentence_ID / token_ID_within_sentence vectorisés ----
    # paragraph_ID = nombre de newlines dans [0, i)
    paragraph_ids = np.zeros(n, dtype=np.int64)
    if n > 1:
        np.cumsum(is_newline[:-1], out=paragraph_ids[1:])

    # sentence_ID = cumsum(sent_start) - 1 (l'original démarre à -1)
    sentence_ids = np.cumsum(final_is_sent_start) - 1

    # token_ID_within_sentence = i - dernier indice de sent_start <= i
    indices = np.arange(n, dtype=np.int64)
    last_sent_start = np.where(final_is_sent_start, indices, -1)
    last_sent_start = np.maximum.accumulate(last_sent_start)
    token_ids_within_sent = indices - last_sent_start

    # ---- 5. Filtrage des tokens newline ----
    mask = ~is_newline
    keep_idx = np.where(mask)[0].tolist()

    # remap token_ID_within_document -> 0..k-1
    kept_token_ids = token_ids[mask]
    new_token_ids  = np.arange(len(kept_token_ids), dtype=np.int64)
    id_mapping = dict(zip(kept_token_ids.tolist(), new_token_ids.tolist()))

    kept_head_ids = head_ids[mask]
    new_head_ids = np.fromiter(
        (id_mapping.get(int(h), int(h)) for h in kept_head_ids),
        dtype=np.int64, count=len(kept_head_ids)
    )

    # ---- 6. DataFrame final ----
    return pd.DataFrame({
        'paragraph_ID':              paragraph_ids[mask],
        'sentence_ID':               sentence_ids[mask],
        'token_ID_within_sentence':  token_ids_within_sent[mask],
        'token_ID_within_document':  new_token_ids,
        'word':                      [words[i]    for i in keep_idx],
        'lemma':                     [lemmas[i]   for i in keep_idx],
        'byte_onset':                byte_onsets[mask],
        'byte_offset':               byte_offsets[mask],
        'POS_tag':                   [pos_tags[i] for i in keep_idx],
        'dependency_relation':       [dep_rels[i] for i in keep_idx],
        'syntactic_head_ID':         new_head_ids,
        'morph':                     [morphs[i]   for i in keep_idx],
    })
