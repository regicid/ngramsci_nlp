---
language: fr
pretty_name: Portraits — people in the French press, by month, newspaper and section
tags: [press, france, named-entities, sqlite]
---

# Portraits database

How often, and how, the French press portrays people: one SQLite file (`portraits.sqlite`) of aggregates per
person × month × newspaper × section, built from the character releases of
[regicid/propp-press](https://huggingface.co/datasets/regicid/propp-press) (Le Monde 1944–, the political articles of
the web press, and the full web press of Le Figaro, Libération, JDD, La Croix, Le Parisien, Les Échos, Mediapart,
Paris Match, Télérama, Valeurs Actuelles). It is the database behind the Portraits pages of Gallicagram.

It holds counts and sums only, no article text. `version.json` gives the build date, the sha256, the row counts, the
corpora and the code commit; earlier builds are in the repo history.

## Tables

- `names`: `key` (normalised name), `name` (display name), `n_articles`, `n_mentions`, `first_month`, `last_month`.
  Names in at least 100 articles.
- `cells`: one row per (`key`, `month`, `newspaper`, `section`): `n_articles` (articles naming the person),
  `n_mentions`, `n_segments` (scored passages), `W` (sum of P(pertinent) over passages) and, for each portrayal score
  s in valence / extremisme / dynamique / journaliste, the sums `S_s` (and second moments `WW`, `SS_s`, `WS_s`, `W3`
  over articles, for article-clustered variances); `_nm` columns weight passages by P(pertinent) × (1 − P(dynamique ≠ 0))
  (momentum-free valence). Agency counts from the syntactic slots: `agent`, `agent_speech`, `patient`, `obl`.
- `cells_year`: the same sums by year, plus `E_s` / `EW_s`, each cell's contribution to a person's expected score given
  everyone else in the same year × newspaper × section (expected score = ΣE_s / ΣEW_s).
- `totals_month`: everyone's sums per month × newspaper × section.

Cells keep sums, not means, so any set of cells merges exactly: mean of s = ΣS_s / ΣW.

## Use

```python
from huggingface_hub import hf_hub_download
import sqlite3
db = sqlite3.connect(hf_hub_download("regicid/portraits-db", "portraits.sqlite", repo_type="dataset"))
db.execute("select month, sum(n_articles) from cells where key = 'charles de gaulle' group by month").fetchall()
```
