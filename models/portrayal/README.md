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

**In one sentence:** the score of a single passage is noisy, but averages over a few hundred mentions per person are reliable.
Per-passage errors are mostly random and cancel out. What remains is a small bias per person, about 0.01–0.03 on a −1 to +1
scale (details in [Evaluation](#evaluation)).

Part of [ngramsci_nlp](https://github.com/regicid/ngramsci_nlp), which builds the inputs (people found by propp NER +
Qwen coreference) and runs this model on whole corpora (stage `portrayal`). This is v2. The previous version (5 seeds,
candidate passages only) is kept at revision `v1-5seeds-base`.

## Outputs

| head | classes | meaning |
|---|---|---|
| `pertinent` | 0 / 1 | the passage says something about the target (not a passing mention) |
| `valence` | -1 / 0 / 1 | negative / neutral / positive judgement of the target |
| `extremisme` | -1 / 0 / 1 | framed as moderate / no framing / framed as extreme (label or substance) |
| `dynamique` | -1 / 0 / 1 | declining / no information / rising in the race |
| `journaliste` | 0 / 1 | the judgement is the author's own (not a quoted person's) |

Full definitions: the annotation codebook (v4) in the ngramsci_nlp repo.

## Input and usage

`"Titre : <title>\n<passage>"`, max 384 tokens. The passage is the sentence mentioning the target plus its neighbouring
sentences. Every mention of the target is wrapped in `[[ ]]`. Mentions of the target's name are replaced by the
placeholder `[[Dominique Martin]]` (also in the title), so the model judges the portrayal, not the person.
The vector fed to the heads is the mean hidden state over the marked target spans, concatenated with the mean over the
whole input.

```python
from ngramsci.portrayal import Scorer          # pip install "ngramsci[portrayal] @ git+https://github.com/regicid/ngramsci_nlp"
s = Scorer.from_hub("n-gramsci/portrayal-camembert-large", seeds=[0])    # seed 0 alone: recommended for large corpora
s.score([{"title": "...", "passage": "Pour [[Dominique Martin]], la candidate d'extrême droite, ...", "full_name": "Dominique Martin"}])
```

The repository holds three seeds (`seed0/`–`seed2/`) of the same fine-tune. See [Seeds](#seeds) for when to average them.

## Training

* Teacher labels: **Claude Sonnet** following the codebook (v4). 12,410 passages from French online press, 2008–2026:
  * 6,250 mention passages about the 2027 presidential candidates, 2,000 of them picked by active learning (rare classes);
  * 3,750 more candidate passages;
  * 2,410 passages about **other people** (politicians and not), many found only through coreference.
* 11,203 train / 941 validation rows after dropping rows where the mention designates someone else. CamemBERT-large,
  LR 1.5e-5, 12 epochs, last epoch kept, fp16.

## Evaluation

"Accuracy" here means agreement with the Sonnet teacher on held-out passages, not with human coders. Claude Opus gives an
idea of how much the choice of teacher matters: on 200 gold mentions, Opus agrees with Sonnet at κ = 0.74 on valence,
0.77 on extremisme, 0.80 on dynamique, 0.73 on pertinent and 0.67 on journaliste.

Held-out sets (none of these passages, or their source articles, are in the training data):

| set | passages | what it contains |
|---|---|---|
| gold | 596 | ordinary mentions of 2027 candidates in 2026 articles, sampled like the training data |
| A | 600 | passages where the candidate is found by coreference ("il", "le maire de Meaux"); 450 without the name |
| B | 652 | passages chosen to over-represent rare labels (praise, moderate framing, decline) |
| C | 800 | passages about 500 other people, politicians and not |

Numbers below are for **seed 0 alone** unless stated.

### 1. One passage: noisy

Cohen's κ against the teacher, gold set:

| valence | extremisme | dynamique | pertinent | journaliste |
|---|---|---|---|---|
| 0.67 | 0.69 | 0.68 | 0.72 | 0.88 |

Mean κ on the other sets: A 0.59, B 0.75, C 0.63.

κ ≈ 0.7 means the model and the teacher disagree on a fair share of single passages, mostly neutral vs mildly negative
or positive. Opus disagrees with the same teacher about as often. **Do not use the model to classify individual
passages for close reading, or to count rare cases one by one.**

### 2. Averages per person: reliable

For each candidate, compare the model's mean score with the teacher's mean label on the same passages. Model score for a
passage = P(positive) − P(negative). Teacher label = −1 / 0 / +1.

Pooled over gold + A + B (19 candidates, median 72 mentions each):

| head | r (model mean vs teacher mean) | overall offset | random error per candidate | **bias per candidate** |
|---|---|---|---|---|
| valence | 0.90 | +0.016 | 0.047 | **0.029** |
| extremisme | 0.96 | +0.002 | 0.020 | **0.013** |
| dynamique | 0.96 | +0.008 | 0.026 | **0.021** |

The gap between model and teacher for a candidate has two parts:

* **random error** comes from the per-passage mistakes. It shrinks as 1/√n with the number of mentions n.
* **bias** is a lean for or against a particular person. It does not shrink with more mentions.

The split compares the observed spread of the gaps across candidates with the spread expected from per-passage errors
alone. Any excess is bias.

**How reliability grows with the number of mentions.** Expected r between the model's and the teacher's candidate means,
projected from the measured random error and bias, with the spread of candidates as in the evaluation sets:

| mentions per candidate | 30 | 100 | 300 | 1,000 | ceiling |
|---|---|---|---|---|---|
| valence | 0.87 | 0.94 | 0.96 | 0.97 | 0.98 |
| extremisme | 0.92 | 0.97 | 0.98 | 0.99 | 0.99 |
| dynamique | 0.91 | 0.96 | 0.97 | 0.98 | 0.98 |

Measured, not projected: with ~28 mentions per candidate (gold alone), valence r = 0.77; with 72 (gold + A + B), 0.90.
**The curve flattens around 300 mentions.** Beyond that, random error is smaller than the bias, and more data no longer
helps.

### 3. What this means in practice

* **≥ 300 mentions per person and period:** averages track the teacher closely (r ≈ 0.96–0.98). In a week of political
  press, that holds for the main candidates. Pool minor candidates over a month or more.
* **Gaps between two people smaller than about 0.06 on valence (0.03 on extremisme, 0.04 on dynamique) are within the
  model's error.** That is roughly twice the bias.
* **Changes over time for the same person are measured better than gaps between people.** A person's bias is largely
  constant, so it cancels out in that person's trend.
* **Report the bias, not only r.** r depends on how different the people compared really are: people portrayed alike
  give a modest r even with a perfect model. The bias is in the same units as the scores.

### v1 vs v2

Mean κ over the five heads, ensemble of seeds:

| set | v1 (5 seeds) | v2 (3 seeds) |
|---|---|---|
| gold | 0.691 | 0.729 |
| A | 0.531 | 0.616 |
| B | 0.723 | 0.743 |
| C | 0.491 | 0.613 |

## Limitations

* **One teacher.** The model reproduces Sonnet's reading of the codebook, including any slant it has towards particular
  people. Whether Sonnet's per-person means differ from Opus's has not been measured: there are too few Opus labels per
  person.
* **The projections assume independent errors across mentions.** Mentions of the same story reprinted across outlets are
  not independent, so 1,000 mentions may count as a few hundred. Read the n = 1,000 column as an upper bound.
* **The bias is estimated from 19 candidates.** Treat 0.01–0.03 as an order of magnitude. B over-represents rare, strong
  judgements, so the pooled valence bias (0.029) may be higher than on ordinary coverage. Gold + A alone gives 0.016.
* **Other people.** Trained mostly on politicians. On non-politicians (3-seed ensemble), valence (κ 0.75), extremisme
  (0.86) and journaliste (0.78) work well. `dynamique` is almost always neutral, and `pertinent` is ambiguous even between
  teachers (Opus vs Sonnet κ 0.47; model vs Sonnet 0.61).
* **Dates and sources.** Evaluation passages are from 2026, French online press. Older periods are not evaluated. The
  training passages are not released (copyrighted articles).

## Seeds

Averaging seeds barely changes the scores (gold mean κ: single seeds 0.71–0.73, 2 or 3 seeds 0.73). It removes the risk
of an unlucky seed, but that risk mostly affects single passages. On candidate averages it is small (valence r 0.90 for
seed 0 alone and for 3 seeds, pooled sets). **Seed 0 alone is recommended for large corpora.** Seed 0 is never the worst
of the three on any evaluation set. Average several seeds if you need scores for single passages.
