#!/usr/bin/env python3
# coding: utf-8
"""
regender_book.py
================
Recompute the `gender` field of every character in a propp `.book` file using
the unified logic in `gendering.py`, replacing propp's per-mention gender
aggregation (which mis-handles unassigned mentions and hides coref merges).

Each character in a .book is a coreference chain whose mentions are already
grouped into proper / common / pronoun. We flatten them (expanded by their
count `c` so the majority votes are weighted) and feed them to
`gendering.genderize_chain`.

The original gender block is preserved under `gender_propp` for comparison;
the new block is written to `gender`.

Usage:
    python regender_book.py path/to/file.book [--out out.book]
    python regender_book.py path/to/books_dir/        # in-place on *.book
"""

import sys
import json
import argparse
from pathlib import Path

from . import gendering

# Point these at the same CSVs the quote pipeline uses.
#NAMES_CSV = '/Users/b.de-courson/code/genderednews-tools/citations_masc_computing/postprocessing/utils/insee_sexratio_firstnames_2017.csv'
NAMES_CSV = "https://raw.githubusercontent.com/getalp/genderednews/cc64c5d5c448ceb6172905a719ee8c5b3d95753c/gn_modules/processing/processings/data/prenoms_clean.csv"
#JOBS_CSV  = '/Users/b.de-courson/code/genderednews-tools/citations_masc_computing/postprocessing/utils/occupations_clean.csv'
JOBS_CSV = "https://raw.githubusercontent.com/getalp/genderednews/refs/heads/main/gn_modules/processing/processings/data/occupations_clean.csv"

def character_mentions(character: dict) -> list:
    """Flatten a character's mentions into a flat list of strings, repeated by
    their count so job/pronoun majority votes are weighted by frequency."""
    out = []
    mentions = character.get('mentions', {}) or {}
    for key in ('proper', 'common', 'pronoun'):
        for item in mentions.get(key, []) or []:
            name = item.get('n')
            if isinstance(name, str):
                out.extend([name] * max(1, int(item.get('c', 1))))
    return out


def regender_character(character: dict) -> dict:
    """Return a new gender block from the character's mentions."""
    res = gendering.genderize_chain(character_mentions(character))
    return {
        'firm':      res['gender_firm'],
        'inclusive': res['gender_inclusive'],
        'pronoun':   res['gender_pronoun'],
        'score':     res['gender_score'],      # continuous P(male) in [0,1]
        'coverage':  res['gender_coverage'],
        'n_gendered': res['n_gendered'],
        'basis':     res['gender_firm_basis'],
        'conflict':  res['gender_conflict'],
    }


def regender_book(characters: list) -> list:
    """Recompute gender for every character in a parsed .book (list)."""
    for ch in characters:
        if 'gender' in ch and 'gender_propp' not in ch:
            ch['gender_propp'] = ch['gender']      # keep the old block
        ch['gender'] = regender_character(ch)
    return characters


def process_file(path: Path, out: Path) -> dict:
    characters = json.loads(path.read_text(encoding='utf-8'))
    characters = regender_book(characters)
    out.write_text(json.dumps(characters, ensure_ascii=False, indent=4),
                   encoding='utf-8')
    counts = {'Male': 0, 'Female': 0, 'Unknown': 0, 'Other': 0, 'conflict': 0}
    for ch in characters:
        g = ch['gender']
        counts[g['firm']] = counts.get(g['firm'], 0) + 1
        if g['conflict']:
            counts['conflict'] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description="Re-gender characters in .book files.")
    ap.add_argument('path', type=Path, help="A .book file or a directory of them.")
    ap.add_argument('--out', type=Path, default=None,
                    help="Output file (single-file mode). Defaults to in-place.")
    args = ap.parse_args()

    gendering.init_resources(NAMES_CSV, JOBS_CSV)

    if args.path.is_dir():
        files = sorted(args.path.glob('*.book'))
        print(f"Re-gendering {len(files)} .book files in {args.path} ...")
        for f in files:
            c = process_file(f, f)         # in-place
            print(f"  {f.name}: {c}")
    else:
        out = args.out or args.path
        c = process_file(args.path, out)
        print(f"Re-gendered {args.path.name} → {out}")
        print(f"  firm counts: {c}")


if __name__ == '__main__':
    main()
