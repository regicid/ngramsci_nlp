---
language: fr
license: mit
base_model: almanach/camembert-large
pipeline_tag: text-classification
tags: [press, politics, stance, portrayal, french]
---

# portrayal-camembert-large

How a French press passage portrays a **target person**: is the passage about them, is it negative or positive about them,
does it frame them as extreme or moderate, does it show them rising or falling, and is the judgement the journalist's own.
Three seeds of the same fine-tune; average their probabilities. Two seeds give almost the same results as three.
The previous version (5 seeds, base training data only) is kept at revision `v1-5seeds-base`.

Part of [ngramsci_nlp](https://github.com/regicid/ngramsci_nlp), which builds the inputs (people found by propp NER +
Qwen coreference) and runs this model on whole corpora (stage `portrayal`).

## Outputs

| head | classes | meaning |
|---|---|---|
| `pertinent` | 0 / 1 | the passage says something about the target (not a passing mention) |
| `valence` | -1 / 0 / 1 | negative / neutral / positive judgement of the target |
| `extremisme` | -1 / 0 / 1 | framed as moderate / no framing / framed as extreme (label or substance) |
| `dynamique` | -1 / 0 / 1 | declining / no information / rising in the race |
| `journaliste` | 0 / 1 | the judgement is the author's own (not a quoted person's) |

Full definitions: the annotation codebook (v4) in the ngramsci_nlp repo.

## Input

`"Titre : <title>\n<passage>"`, max 384 tokens. The passage is the sentence mentioning the target plus its neighbouring
sentences; every mention of the target is wrapped in `[[ ]]` and mentions of the target's name are replaced by the
placeholder `[[Dominique Martin]]` (also in the title), so the model judges the portrayal, not the person.
The vector fed to the heads is the mean hidden state over the marked target spans concatenated with the mean over the
whole input.

```python
from ngramsci.portrayal import Scorer          # pip install "ngramsci[portrayal] @ git+https://github.com/regicid/ngramsci_nlp"
s = Scorer.from_hub("n-gramsci/portrayal-camembert-large")
s.score([{"title": "...", "passage": "Pour [[Dominique Martin]], la candidate d'extrême droite, ..."}])
```

## Training and evaluation

* Teacher labels: Claude Sonnet following the codebook (v4). Training data (12,410 passages from French online press, 2008–2026):
  * 6,250 mention passages about the 2027 presidential candidates, 2,000 of them picked by active learning (rare classes);
  * 3,750 more candidate passages (v2);
  * 2,410 passages about **other people** (politicians and non-politicians), many found only through coreference (v2).
* 11,203 train rows after dropping rows where the mention designates someone else; 12 epochs, last epoch kept, fp16,
  CamemBERT-large, LR 1.5e-5, seeds 0–2.
* Evaluation sets, all kept out of training and labelled by the same teacher. Mean Cohen's κ over the five heads, ensemble of seeds:

| set | v1 (5 seeds) | v2 (3 seeds) |
|---|---|---|
| gold: 596 candidate mentions, 2026 | 0.691 | 0.729 |
| A: 600 passages where the candidate is found by coreference | 0.531 | 0.616 |
| B: 652 passages rich in rare classes | 0.723 | 0.743 |
| C: 800 passages about 500 other people | 0.491 | 0.613 |

Per head on gold (v2 ensemble): valence 0.69, extremisme 0.66, dynamique 0.68, pertinent 0.73, journaliste 0.89.
On other people who are not politicians: valence 0.75, extremisme 0.86, journaliste 0.78, pertinent 0.61
(Claude Opus against the teacher on the same rows: 0.84 / 0.90 / 0.84 / 0.47).

For reference, Claude Opus against the Sonnet teacher on 200 gold mentions: valence 0.74, extremisme 0.77, dynamique 0.80,
pertinent 0.73, journaliste 0.67. Rare classes are few in the gold set, so their κ is noisy: judge the model on aggregates
(person × period means). Candidate-level correlation of mean valence with the teacher on gold: 0.79 (single seeds 0.72–0.79;
it varies from 0.66 to 0.86 between seeds of the same recipe).

Reliability of single passages vs averages per person (bias, how many mentions are needed): see [EVALUATION.md](EVALUATION.md).

## Limitations

* Trained mostly on politicians. `valence`, `extremisme` and `journaliste` work on other people too; `dynamique` (the electoral race)
  is almost always neutral for non-politicians, and what counts as `pertinent` for them is ambiguous even between teachers.
* Labels come from an LLM teacher, not from human coders; the model reproduces the teacher's reading of the codebook.
* French online press only. The training passages are not released (copyrighted articles).
