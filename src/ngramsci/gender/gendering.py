#!/usr/bin/env python3
# coding: utf-8
"""
gendering.py
============
Source-level gender inference, designed to be simple enough to describe in a
methods section. Produces TWO labels per quote source plus one independent
validation signal:

  1. `gender_firm`      — the committed label. It is decided by the source's
                          OWN gendered cues, treated co-equally:
                            * firstname (INSEE sex ratio),
                            * honorific (M./Mme/Monsieur/Madame/mère/sœur/…),
                            * job title (lexicon score, epicene titles excluded).
                          If the cues present all point one way, that gender is
                          returned; if any two disagree (one Male, one Female),
                          it ABSTAINS (Unknown). Only when the source carries no
                          cue at all does it fall back to the coref chain.

  2. `gender_inclusive` — permissive. Identical to `gender_firm`, but when the
                          firm label abstains it falls back to the surface
                          pronoun/determiner cue (il/elle/le/la). This raises
                          coverage at the cost of relying on weaker signals.

  3. `gender_pronoun_det` — the pronoun/determiner cue on its own (il/elle,
                          le/la), computed for EVERY source regardless of the
                          firm decision. It is never used to *build* the firm
                          label, which makes it a clean independent signal to
                          cross-tabulate against `gender_firm` for validation
                          ("when firm says Male, does the surface cue agree?").

Design commitments (all stated to be reviewer-auditable):
  - Symmetric M/F treatment at every step.
  - Abstain (return Unknown) on any internal conflict rather than guessing.
  - The source's own cues (firstname / honorific / job) are co-equal and are
    consulted BEFORE the coref chain; the chain is a fallback for cue-less
    sources only.
  - Epicene professions and institutional nouns never fire a gender on their
    own (they defer to firstname / honorific / upstream coref). The epicene
    list is the lever for neutralising unreliable role titles (e.g. ministre,
    secrétaire): a title placed there can no longer assert a gender or trigger
    a spurious conflict against a name.

Labels are one of: 'Male', 'Female', 'Unknown', 'Other' (multi-speaker).

Usage from the calling script
------------------------------
    import gendering
    gendering.init_resources(NAMES_CSV, JOBS_CSV)   # once per worker
    out = gendering.genderize(source, chain_gender=cg)
    # out is a dict: gender_firm, gender_firm_basis,
    #                gender_inclusive, gender_pronoun_det
"""

import re
import sys
import unicodedata
from typing import Optional

import pandas as pd

# ──────────────────────────────────────────────
# RESOURCES (populated by init_resources)
# ──────────────────────────────────────────────
_NAMES_DICT: Optional[dict] = None   # firstname (normalised) → sex ratio (0=F, 1=M)
_JOBS_DICT:  Optional[dict] = None   # job token (normalised) → score (0=F, 1=M)

# Decision thresholds on a [0,1] score. Above MALE_T → Male, below FEMALE_T →
# Female, in between → abstain. The dead zone in the middle is what makes the
# pipeline abstain on weakly-gendered or near-balanced names/jobs.
MALE_T = 0.7
FEMALE_T = 0.3


def init_resources(names_csv: str, jobs_csv: str) -> None:
    """Load INSEE firstname ratios and the job→score table into module globals.

    Safe to call once per multiprocessing worker. Mirrors the loader in the
    main pipeline so the two stay consistent.
    """
    global _NAMES_DICT, _JOBS_DICT

    names_df = pd.read_csv(names_csv, delimiter=';')
    names_df = names_df.loc[names_df.n > 99]
    scores = (names_df['sexratio_prenom'].astype(str)
              .str.replace(',', '.', regex=False))
    scores = pd.to_numeric(scores, errors='coerce')
    _NAMES_DICT = {_norm(name): float(score)
                   for name, score in zip(names_df['preusuel'], scores)
                   if pd.notna(score)}

    jobs_df = pd.read_csv(jobs_csv, delimiter=';', index_col=0, header=None)
    _JOBS_DICT = {_norm(k): v for k, v in jobs_df.iloc[:, 0].to_dict().items()}


# ──────────────────────────────────────────────
# NORMALISATION
# ──────────────────────────────────────────────
_COMBINING_TABLE = {c: None for c in range(sys.maxunicode + 1)
                    if unicodedata.combining(chr(c))}
_UNK_RE = re.compile(r'<unk>')


def _norm(s) -> str:
    """Lowercase, strip <unk>, strip accents. Keeps internal punctuation for
    honorific detection ('m.' must survive); token splitting handles the rest.
    """
    if not isinstance(s, str):
        return ""
    s = _UNK_RE.sub('', s.lower())
    s = unicodedata.normalize('NFKD', s).translate(_COMBINING_TABLE)
    return re.sub(r'\s+', ' ', s).strip()


# ──────────────────────────────────────────────
# GUARD SETS
# ──────────────────────────────────────────────
# Professions historically epicene or only recently feminised: neither the
# masculine nor the feminine form licenses an inference. They must be carried
# by firstname / honorific / coref. This list is ALSO the neutralisation lever:
# a role title whose lexicon score is gender-skewed but which is epicene in
# usage (e.g. 'secretaire') is placed here so it cannot assert a gender on its
# own, nor clash with a name under the co-equal rule. (Accent-free, lowercase.)
_AMBIGUOUS_PROFESSIONS = {
    'ministre', 'premier', 'premiere', 'professeur', 'professeure',
    'auteur', 'auteure', 'autrice', 'docteur', 'docteure', 'doctoresse',
    'ecrivain', 'ecrivaine', 'juge', 'maire', 'medecin', 'magistrat',
    'magistrate', 'procureur', 'procureure', 'depute', 'deputee',
    'ingenieur', 'ingenieure', 'maitre', 'general', 'generale',
    'secretaire',   # added: epicene in usage ("le/la secrétaire"), skewed in
                    # the lexicon — neutralised so it neither labels nor clashes
}

# Institutional / collective nouns that must never be gendered as a person.
_INSTITUTIONAL_NOUNS = {
    'commission', 'prefecture', 'mairie', 'direction', 'redaction', 'police',
    'gendarmerie', 'assemblee', 'senat', 'parlement', 'gouvernement',
    'ministere', 'presidence', 'conseil', 'tribunal', 'cour',
    'administration', 'delegation', 'ambassade', 'prefet', 'justice', 'armee',
    'marine', 'municipalite', 'region', 'commune', 'collectivite', 'autorite',
    'agence', 'bureau', 'cellule', 'equipe', 'federation', 'association',
    'fondation', 'institution', 'organisation', 'syndicat', 'parti',
    'coalition', 'alliance', 'mouvement', 'groupe', 'sdj', 'journal',
    'presse', 'revue', 'hebdomadaire', 'quotidien', 'radio', 'television',
    'chaine', 'antenne', 'sofica', 'societe', 'entreprise', 'holding',
    'filiale', 'comptes', 'chambre', 'province', 'hierarchie', 'encadrement',
    'management',
}

# Tokens that match an INSEE firstname but are not people in this corpus and
# should never anchor a gender. Normalised (lowercase, accent-free):
#   france  — country
#   lula    — surname-as-name noise
#   sharon  — ambiguous / non-person usage
#   isra    — fragment of "Israël"; the ë splits the raw token, leaving "Isra"
#             which would otherwise resolve to a firstname
_BLOCKLIST_FIRSTNAMES = {'france', 'lula', 'sharon', 'isra','barbe','beria','sultan','péguy'}

# French articles/particles that are never firstnames (skipped in name lookup).
_SKIP_TOKENS = {
    'la', 'le', 'les', 'un', 'une', 'des', 'du', 'de', 'l', 'd', 'et', 'ou',
    'en', 'au', 'aux', 'sur', 'par', 'pour', 'avec', 'ce', 'cet', 'cette',
    'son', 'sa', 'mon', 'ma', 'ton', 'ta',
}


# ──────────────────────────────────────────────
# SIGNAL DETECTORS  (each returns 'Male' / 'Female' / None; None = abstain)
# ──────────────────────────────────────────────
# M./M (monsieur, either preprocessing) handled by `\bmm?\b\.?`. The bare-M
# alternative is deliberate — we don't know whether the corpus keeps the dot.
# `mme`/`mlle` are NOT caught by it (no word boundary after "mm"/"m" there).
# Accent-free after _norm: père→pere, abbé→abbe. The œ ligature is NOT
# decomposed by NFKD, so we match both 'sœur' and 'soeur' explicitly.
_HONORIFIC_M = re.compile(
    r"\bmm?\b(?!['\u2019])\.?|"
    r'\b(?:monsieur|mgr|cardinal|prince|sir|lord|comte|pere|frere|abbe)\b',
    re.I)
_HONORIFIC_F = re.compile(
    r'\b(?:mme|madame|mlle|princesse|lady|comtesse|mere|soeur|sœur)\b', re.I)

_PRONOUN_M = re.compile(r'\b(?:il|ils)\b', re.I)
_PRONOUN_F = re.compile(r'\b(?:elle|elles)\b', re.I)

# Leading determiner only (gender of the following noun). Weak signal.
_DET_M = re.compile(r'^(?:le|un)\s+\w', re.I)
_DET_F = re.compile(r'^(?:la|une)\s+\w', re.I)


def _score_to_gender(score: float) -> Optional[str]:
    if score > MALE_T:
        return 'Male'
    if score < FEMALE_T:
        return 'Female'
    return None


def _firstname_genders(source: str) -> set:
    """Set of genders implied by recognised INSEE firstnames in the source.

    Returns a subset of {'Male', 'Female'}. Empty if no firstname recognised,
    {one} for a clean signal, or {'Male','Female'} when the source contains two
    firstnames of opposing gender — the latter is what flags a coref chain that
    has probably conflated two different people.
    """
    if _NAMES_DICT is None:
        return set()
    s = _norm(source)
    s = re.sub(r'[^\w\s]', ' ', s)
    found = set()
    for tok in s.split():
        if tok in _SKIP_TOKENS:
            continue
        if tok in _BLOCKLIST_FIRSTNAMES:
            continue
        # A lowercase institutional homograph ('marine') is the institution;
        # capitalisation is lost after _norm, so we err toward NOT reading
        # 'marine'/'france' as a name by skipping institutional tokens.
        if tok in _INSTITUTIONAL_NOUNS:
            continue
        score = _NAMES_DICT.get(tok)
        if score is None:
            continue
        g = _score_to_gender(score)
        if g:
            found.add(g)
    return found


def _firstname_gender(source: str) -> Optional[str]:
    """Single firstname gender, or None on no-signal / conflict."""
    g = _firstname_genders(source)
    return next(iter(g)) if len(g) == 1 else None


def _honorific_genders(source: str) -> set:
    """Set of genders implied by honorifics ({} / {one} / {'Male','Female'})."""
    s = _norm(source)
    found = set()
    if _HONORIFIC_M.search(s):
        found.add('Male')
    if _HONORIFIC_F.search(s):
        found.add('Female')
    return found


def _honorific_gender(source: str) -> Optional[str]:
    g = _honorific_genders(source)
    return next(iter(g)) if len(g) == 1 else None


def _job_gender(source: str) -> Optional[str]:
    """Job-title cue from the lexicon, with epicene professions excluded."""
    if _JOBS_DICT is None:
        return None
    s = _norm(source)
    tokens = [t for t in re.findall(r'\w+', s)
              if t not in _AMBIGUOUS_PROFESSIONS]
    for tok in tokens:
        if tok in _JOBS_DICT:
            try:
                return _score_to_gender(float(_JOBS_DICT[tok]))
            except (TypeError, ValueError):
                continue
    return None


def _pronoun_gender(text: str) -> Optional[str]:
    """Explicit personal pronoun only (il/ils vs elle/elles). Abstain on both."""
    s = _norm(text)
    m, f = bool(_PRONOUN_M.search(s)), bool(_PRONOUN_F.search(s))
    if m and f:
        return None
    if m:
        return 'Male'
    if f:
        return 'Female'
    return None


def _pronoun_det_gender(source: str) -> Optional[str]:
    """Surface cue: explicit pronoun (strong) then leading determiner (weak).

    Determiners are suppressed when the source is institutional or contains an
    epicene profession, otherwise 'la France' / 'le ministre' would inject
    spurious gender. This keeps the signal usable as a validation reference.
    """
    g = _pronoun_gender(source)
    if g is not None:
        return g

    s = _norm(source)
    # Leading determiner — only if the source is not institutional/epicene.
    tokens = set(re.findall(r'\w+', s))
    if tokens & _INSTITUTIONAL_NOUNS or tokens & _AMBIGUOUS_PROFESSIONS:
        return None
    if _DET_M.search(s):
        return 'Male'
    if _DET_F.search(s):
        return 'Female'
    return None


# ──────────────────────────────────────────────
# CONTINUOUS SCORING  (P(male) ∈ [0, 1]; 1 = male, 0 = female)
# ──────────────────────────────────────────────
# The continuous path exists to avoid the argmax/threshold step, which is the
# operation that converts a majority-male context into female→male flips under
# coref over-merging. A linearly-propagated score conserves aggregate gender
# mass under any re-partition of mentions into chains.
#
# NOTE: the firm label treats the job title as a decisive, co-equal cue, but
# the continuous score keeps it SOFT — `_source_hard_score` pins to 0/1 only on
# pronoun / firstname / honorific (signals about the individual), never on a job
# title (a population base rate). So a job-only source can read firm=Female with
# a score equal to that occupation's raw ratio (e.g. 0.05), not a hard 0.0.
def _firstname_score(text: str) -> Optional[float]:
    """Mean INSEE sex ratio over recognised firstnames (raw, continuous).

    Uses the raw ratio rather than a thresholded label, so 'Dominique'≈0.5
    contributes 0.5 instead of abstaining. None if no firstname recognised.
    """
    if _NAMES_DICT is None:
        return None
    s = re.sub(r'[^\w\s]', ' ', _norm(text))
    vals = []
    for tok in s.split():
        if (tok in _SKIP_TOKENS or tok in _BLOCKLIST_FIRSTNAMES
                or tok in _INSTITUTIONAL_NOUNS):
            continue
        v = _NAMES_DICT.get(tok)
        if v is not None:
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def _honorific_score(text: str) -> Optional[float]:
    s = _norm(text)
    m, f = bool(_HONORIFIC_M.search(s)), bool(_HONORIFIC_F.search(s))
    if m and not f:
        return 1.0
    if f and not m:
        return 0.0
    return None  # both or neither → no signal


def _job_score(text: str) -> Optional[float]:
    """Continuous job score from the lexicon. Epicene professions excluded.

    Jobs always contribute here (no longer gated behind firstname absence) —
    they are pooled co-equally with the other cues in `_mention_score`.
    """
    if _JOBS_DICT is None:
        return None
    for tok in re.findall(r'\w+', _norm(text)):
        if tok in _AMBIGUOUS_PROFESSIONS:
            continue
        if tok in _JOBS_DICT:
            try:
                return float(_JOBS_DICT[tok])
            except (TypeError, ValueError):
                continue
    return None


def _mention_score(text: str) -> Optional[float]:
    """Continuous P(male) for ONE mention, or None if no signal.

    An explicit pronoun is a HARD pin: 'elle' returns 0.0 and 'il' returns 1.0
    outright, with nothing averaged in — it is never possible to score 'elle'
    as male. Otherwise the firstname, honorific and job scores present are
    pooled by simple mean.
    """
    if not isinstance(text, str):
        return None
    p = _pronoun_gender(text)
    if p == 'Male':
        return 1.0
    if p == 'Female':
        return 0.0
    subs = [v for v in (_firstname_score(text),
                        _honorific_score(text),
                        _job_score(text)) if v is not None]
    return sum(subs) / len(subs) if subs else None


def _source_hard_score(source: str) -> Optional[float]:
    """Hard 0.0 / 1.0 when the source is UNAMBIGUOUSLY gendered by its own
    strong cues (explicit pronoun, clear firstname, or honorific), else None.

    "Unambiguous" means the strong cues all point one way: a clear firstname
    (INSEE ratio past the 0.7/0.3 threshold), an honorific, or an il/elle
    pronoun. A borderline name ('Dominique'≈0.5) or any M/F clash returns None,
    deferring to the chain. Job titles are deliberately NOT included here (see
    the note above the scoring block). This is the "trust the name first → 1 or
    0" rule.
    """
    genders = set()
    p = _pronoun_gender(source)
    if p:
        genders.add(p)
    genders |= _firstname_genders(source)
    genders |= _honorific_genders(source)
    if genders == {'Male'}:
        return 1.0
    if genders == {'Female'}:
        return 0.0
    return None


# ──────────────────────────────────────────────
# MULTI-SPEAKER
# ──────────────────────────────────────────────
_CAP_TOKEN = re.compile(r'\b([A-ZÀ-Ý][\wÀ-ÿ\-]+)')
# Capitalised tokens that are NOT proper names (leading determiners, honorific
# abbreviations, demonstratives). Normalised (lowercase, dot stripped).
_NON_NAME_CAP = {
    'la', 'le', 'les', 'l', 'un', 'une', 'des', 'du', 'de', 'd',
    'm', 'mm', 'mme', 'mlle', 'mgr', 'dr', 'me', 'st',
    'aujourd', 'hui', 'cette', 'ce', 'ces', 'leur', 'leurs',
}


def _proper_names(seg: str) -> list:
    """Capitalised tokens in a segment that look like proper names — leading
    determiners and honorifics (capitalised only by position) are dropped."""
    return [t for t in _CAP_TOKEN.findall(seg)
            if t.lower().strip('.') not in _NON_NAME_CAP]


def _is_multispeaker(source: str) -> bool:
    """True only when two or more comma / 'et' / '&'-separated segments each
    carry a proper name.

    This distinguishes genuine multi-speaker sources ("Pierre et Marie",
    "Churchill, Roosevelt et Staline", "le général de Gaulle et M. Bidault")
    from APPOSITIONS, which describe a single referent and must NOT be 'Other':
    "La mère, Mme Duval-Lavallée" and "le ministre des affaires étrangères, M.
    Padilla" each have a proper name in only one segment, so they are single.
    """
    if not isinstance(source, str):
        return False
    parts = re.split(r'\s+et\s+|\s*&\s*|\s*,\s*', source)
    return sum(1 for p in parts if _proper_names(p)) >= 2


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────
def gender_firm(source: str, chain_gender: Optional[str] = None) -> tuple:
    """Conservative label. Returns (label, basis).

    SOURCE-FIRST, CUES CO-EQUAL — the source's own firstname, honorific and job
    cues are pooled with equal authority and trusted before the chain, so a
    quote keeps its own gender rather than inheriting a merged chain's majority
    (this is what makes it robust to coref over-merging):

      1. Collect the genders implied by the source's firstname, honorific and
         job-title cues.
           * all present cues agree            → that gender
           * any two disagree (Male vs Female) → Unknown ('conflict')
           * no cue present                    → fall through
      2. chain_gender, if supplied (Male/Female) — used only for cue-less
         sources.
      3. otherwise Unknown.

    `basis` is one of: 'multispeaker', 'firstname', 'honorific', 'job',
    'firstname+honorific', 'firstname+job', 'honorific+job',
    'firstname+honorific+job', 'conflict', 'chain', ''.

    Epicene/role titles in `_AMBIGUOUS_PROFESSIONS` (e.g. ministre, secrétaire)
    never produce a job cue, so they neither label a cue-less source nor clash
    with a name. (An explicit source pronoun is enforced separately as a hard
    pin in `genderize`, so 'elle' can never end up Male.)
    """
    if _is_multispeaker(source):
        return 'Other', 'multispeaker'

    fn  = _firstname_gender(source)
    hon = _honorific_gender(source)
    job = _job_gender(source)

    cues = {g for g in (fn, hon, job) if g}     # source's own gendered cues
    if len(cues) >= 2:                          # i.e. {'Male', 'Female'}
        return 'Unknown', 'conflict'
    if len(cues) == 1:
        g = next(iter(cues))
        basis = '+'.join(name for name, val in (('firstname', fn),
                                                ('honorific', hon),
                                                ('job', job)) if val)
        return g, basis

    if chain_gender in ('Male', 'Female'):
        return chain_gender, 'chain'

    return 'Unknown', ''


def genderize(source: str, chain_gender: Optional[str] = None,
              chain_score: Optional[float] = None,
              chain_conflict: bool = False) -> dict:
    """Full per-source result.

    Returns:
        gender_firm        : committed label (firstname/honorific/job co-equal,
                             then chain)
        gender_firm_basis  : which cue(s) decided it
        gender_inclusive   : firm, falling back to pronoun/determiner on abstain
        gender_pronoun_det : independent il/elle/le/la cue (validation signal)
        gender_conflict    : True if firstnames + honorifics together imply BOTH
                             Male and Female. Use this to flag coref chains that
                             have likely merged people of different gender.
                             (Job cues are NOT folded into this flag — it is
                             specifically a coref-merge diagnostic.)
        gender_score       : continuous P(male) in [0,1], or None. The source's
                             OWN strong signal (pronoun/firstname/honorific)
                             takes priority and pins to 0/1; the chain score is
                             used only to fill a gap. Job titles stay soft here
                             (raw ratio), never a hard pin.
    """
    if not isinstance(source, str) or not source.strip():
        return {'gender_firm': 'Unknown', 'gender_firm_basis': '',
                'gender_inclusive': 'Unknown', 'gender_pronoun_det': 'Unknown',
                'gender_conflict': False, 'gender_score': chain_score}

    firm, basis = gender_firm(source, chain_gender)

    pron = _pronoun_det_gender(source) or 'Unknown'

    # Common-sense pin: an explicit pronoun in the source can never be
    # contradicted ('elle' is never Male), even by a confident chain.
    src_pron = _pronoun_gender(source)
    if src_pron == 'Male' and firm == 'Female':
        firm, basis = 'Unknown', 'pronoun-chain-conflict'
    elif src_pron == 'Female' and firm == 'Male':
        firm, basis = 'Unknown', 'pronoun-chain-conflict' 
    # Cross-gender conflict over the strong cues (firstnames + honorifics),
    # independent of the firm decision and of chain_gender.
    strong = _firstname_genders(source) | _honorific_genders(source)
    conflict = ('Male' in strong) and ('Female' in strong)

    # Inclusive: extend coverage only where firm abstained. Firm Male/Female/
    # Other is never overridden by the weaker surface cue.
    if firm == 'Unknown' and pron in ('Male', 'Female'):
        inclusive = pron
    else:
        inclusive = firm

    # Continuous score. Trust the source's own name first: an unambiguously
    # gendered source pins to exactly 0.0 / 1.0. Otherwise borrow the chain's
    # soft score — but only if the chain is coherent; a conflicted (merged)
    # chain's blended score is meaningless for an ambiguous source, so we don't
    # donate it. Failing that, fall back to any weak source signal (job).
    hard = _source_hard_score(source)
    if hard is not None:
        score = hard
    elif chain_score is not None and not chain_conflict:
        score = chain_score
    else:
        score = _mention_score(source)   # weak source signal (e.g. job), or None

    return {'gender_firm': firm,
            'gender_firm_basis': basis,
            'gender_inclusive': inclusive,
            'gender_pronoun_det': pron,
            'gender_conflict': conflict,
            'gender_score': round(score, 4) if score is not None else None}


def genderize_chain(mentions) -> dict:
    """Gender one coreference chain from a flat list of its mention strings.

    This is the shared primitive behind both use cases:
      - quote gendering: pass the mention texts of the entity's COREF chain;
      - book re-gendering: pass a character's proper + common + pronoun
        mention texts (expand by count for the majority votes to be weighted).

    Detectors are type-agnostic, so the list can be unsorted: a name only
    trips the firstname detector, an honorific only the honorific detector, a
    pronoun only the pronoun vote.

    Firm philosophy: a single gender across firstnames + honorifics; if those
    are silent, a job-title majority vote. Conflicting firstnames/honorifics →
    abstain (the chain has likely merged people). NOTE: at the CHAIN level the
    job cue is a majority-vote fallback (not co-equal with name/honorific as it
    is for a single source), because a multi-mention chain aggregates better by
    voting than by the single-NP co-equal/conflict rule.
      inclusive  : firm, falling back to a pronoun (il/elle) majority when firm
                   abstained (but never on a detected conflict).
      conflict   : firstnames + honorifics together imply BOTH genders — the
                   flag for chains that probably conflated several people.
    """
    texts = [m for m in (mentions or []) if isinstance(m, str)]

    fn, hon = set(), set()
    for m in texts:
        fn |= _firstname_genders(m)
        hon |= _honorific_genders(m)
    strong = fn | hon
    conflict = ('Male' in strong) and ('Female' in strong)

    if len(strong) == 1:
        firm, basis = next(iter(strong)), 'firstname/honorific'
    elif conflict:
        firm, basis = 'Unknown', 'conflict'
    else:
        # No firstname/honorific anywhere → job-title majority vote.
        male = female = 0
        for m in texts:
            jg = _job_gender(m)
            if jg == 'Male':
                male += 1
            elif jg == 'Female':
                female += 1
        if male > female:
            firm, basis = 'Male', 'job'
        elif female > male:
            firm, basis = 'Female', 'job'
        else:
            firm, basis = 'Unknown', ''

    # Pronoun majority across mentions (il/elle only — no determiners here).
    male = female = 0
    for m in texts:
        pg = _pronoun_gender(m)
        if pg == 'Male':
            male += 1
        elif pg == 'Female':
            female += 1
    if male > female:
        pron = 'Male'
    elif female > male:
        pron = 'Female'
    else:
        pron = 'Unknown'

    if firm == 'Unknown' and not conflict and pron in ('Male', 'Female'):
        inclusive = pron
    else:
        inclusive = firm

    # Continuous score: mean per-mention P(male). Linear, so it conserves
    # aggregate gender mass under coref re-partitioning (no argmax flip).
    # `n_gendered` / `coverage` tell you how much evidence backs the score.
    mscores = [s for s in (_mention_score(m) for m in texts) if s is not None]
    n_gendered = len(mscores)
    score = round(sum(mscores) / n_gendered, 4) if n_gendered else None
    coverage = round(n_gendered / len(texts), 4) if texts else 0.0

    return {'gender_firm': firm,
            'gender_firm_basis': basis,
            'gender_inclusive': inclusive,
            'gender_pronoun': pron,
            'gender_conflict': conflict,
            'gender_score': score,        # P(male) in [0,1], None if no signal
            'n_gendered': n_gendered,     # mentions that produced a score
            'gender_coverage': coverage}  # n_gendered / total mentions
# ──────────────────────────────────────────────
if __name__ == '__main__':
    # Quick smoke test with the dict stubbed (no CSVs needed).
    _NAMES_DICT = {'marie': 0.02, 'pierre': 0.98, 'dominique': 0.5}
    _JOBS_DICT = {'boulanger': 0.95, 'infirmiere': 0.03, 'secretaire': 0.08}
    examples = [
        "Marie Dupont",
        "M. Pierre Durand",
        "Mme la directrice",          # honorific F + epicene job (job suppressed)
        "le boulanger",               # job → Male (co-equal cue)
        "le secrétaire de la rédaction",  # secrétaire epicene → Unknown
        "le secrétaire, M. Pierre Durand",# name wins; secrétaire neutralised
        "elle",                       # pronoun only
        "la France",                  # institutional → no determiner gender
        "le ministre",                # epicene → firm abstains, det suppressed
        "Pierre et Marie",            # multi-speaker → Other
        "La mère, Mme Duval-Lavallée",# apposition → Female (not Other)
        "Dominique",                  # near-balanced name → abstain
        "Mme Pierre",                 # honorific F vs firstname M → conflict
        "Mgr Lustiger",               # M honorific
        "la comtesse de Paris",       # F honorific
        "M Dupont",                   # bare M (no dot) → Male
        "l'abbé Pierre",              # M honorific (abbe) + firstname M → agree
    ]
    for ex in examples:
        print(f"{ex!r:32} → {genderize(ex)}")
