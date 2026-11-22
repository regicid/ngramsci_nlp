# How reliable is the portrayal model?

Short answer: **one mention is noisy, but averages over a few hundred mentions per person are reliable.**
The model's score for a single passage often differs from the teacher's label. These errors are mostly random,
so they cancel out when you average. What remains is a small bias per person, about 0.01–0.03 on a −1 to +1 scale.

Model: `n-gramsci/portrayal-camembert-large` v2 (base + G + H training data). All numbers below use **seed 0 alone**,
the recommended setting for large runs (see "Seeds" at the end).

## What the model is compared against

The model is a student of **Claude Sonnet**, which labelled every passage following the codebook (v4). "Accuracy" here
means agreement with Sonnet on held-out passages, not with human coders. Claude Opus gives an idea of how much the
choice of teacher matters: on 200 gold mentions, Opus agrees with Sonnet at κ = 0.74 on valence, 0.77 on extreme
framing and 0.80 on dynamique.

Held-out evaluation sets (none of these passages, or their source articles, are in the training data):

| set | passages | what it contains |
|---|---|---|
| gold | 596 | ordinary mentions of 2027 candidates in 2026 articles, sampled like the training data |
| A | 600 | passages where the candidate is found by coreference ("il", "le maire de Meaux"); 450 without the name |
| B | 652 | passages chosen to over-represent rare labels (praise, moderate framing, decline) |
| C | 800 | passages about 500 other people, politicians and not |

## 1. One mention: noisy

Cohen's κ against the teacher, gold set, seed 0:

| valence | extremisme | dynamique | pertinent | journaliste |
|---|---|---|---|---|
| 0.67 | 0.69 | 0.68 | 0.72 | 0.88 |

Mean κ on the other sets: A 0.59, B 0.75, C 0.63.

κ ≈ 0.7 means the model and the teacher disagree on a fair share of single passages, mostly neutral vs mildly
negative or positive. A second strong model (Opus) disagrees with the same teacher about as often. **Do not use the
model to classify individual passages for close reading or to count rare cases one by one.**

## 2. Averages per person: reliable

For each candidate, compare the model's mean score with the teacher's mean label on the same passages. Model score
for a passage = P(positive) − P(negative). Teacher label = −1 / 0 / +1.

Pooled over gold + A + B (19 candidates, median 72 mentions each):

| head | r (model mean vs teacher mean) | overall offset | random error per candidate | **bias per candidate** |
|---|---|---|---|---|
| valence | 0.90 | +0.016 | 0.047 | **0.029** |
| extremisme | 0.96 | +0.002 | 0.020 | **0.013** |
| dynamique | 0.96 | +0.008 | 0.026 | **0.021** |

The gap between model and teacher for a candidate has two parts:

* **random error** comes from the per-passage mistakes. It shrinks as 1/√n with the number of mentions n.
* **bias** is a lean for or against a particular person. It does not shrink with more mentions.

The split comes from comparing the observed spread of the gaps across candidates with the spread expected from
per-passage errors alone. Any excess is bias.

### How reliability grows with the number of mentions

Expected correlation between the model's and the teacher's candidate means, as a function of mentions per
candidate. Projected from the measured random error and bias; the spread of candidates is as in the evaluation sets:

| mentions per candidate | 30 | 100 | 300 | 1,000 | ceiling |
|---|---|---|---|---|---|
| valence | 0.87 | 0.94 | 0.96 | 0.97 | 0.98 |
| extremisme | 0.92 | 0.97 | 0.98 | 0.99 | 0.99 |
| dynamique | 0.91 | 0.96 | 0.97 | 0.98 | 0.98 |

Measured, not projected: with ~28 mentions per candidate (gold alone), valence r = 0.77; with 72 (gold + A + B), 0.90.

**The curve flattens around 300 mentions.** Beyond that, random error is smaller than the bias, and more data no
longer helps. The ceiling is set by the bias, not by volume.

## 3. What this means in practice

* **≥ 300 mentions per person and period:** averages track the teacher closely (r ≈ 0.96–0.98). In a week of
  political press, that holds for the main candidates. Pool minor candidates over a month or more.
* **Gaps between two people smaller than about 0.06 on valence (0.03 on extremisme, 0.04 on dynamique) are within
  the model's error.** That is roughly twice the bias.
* **Changes over time for the same person are measured better than gaps between people.** A person's bias is
  largely constant, so it cancels out in that person's trend.
* **Report the bias, not only r.** r depends on how different the people compared really are: candidates portrayed
  alike give a modest r even with a perfect model. The bias is in the same units as the scores.

## Caveats

* **One teacher.** The model reproduces Sonnet's reading of the codebook, including any slant it has towards
  particular people. We have not measured whether Sonnet's per-person means differ from Opus's: there are too few
  Opus labels per person.
* **The projections assume independent errors across mentions.** Mentions of the same story reprinted across
  outlets are not independent, so 1,000 mentions may count as a few hundred. Read the n = 1,000 column as an upper
  bound.
* **The bias is estimated from 19 candidates.** Treat 0.01–0.03 as an order of magnitude. B over-represents rare,
  strong judgements, so the pooled valence bias (0.029) may be higher than on ordinary coverage. Gold + A alone
  gives 0.016.
* **Other people.** On non-politicians (3-seed ensemble), valence (κ 0.75) and extremisme (0.86) work well. `dynamique` is almost
  always neutral, and `pertinent` is ambiguous even between teachers (Opus vs Sonnet κ 0.47; model vs Sonnet 0.61).
* **Dates.** The evaluation passages are from 2026 and from French online press. Older periods are not evaluated.

## Seeds

The repository holds three seeds trained the same way. Averaging seeds barely changes the scores (gold mean κ:
single seeds 0.71–0.73, 2 seeds 0.73, 3 seeds 0.73). It removes the risk of an unlucky seed, but that risk mostly
affects single passages. On candidate averages it is small (valence r 0.90 for seed 0 alone vs 0.90 for 3 seeds,
pooled sets). **Seed 0 alone is recommended for large corpora.** Seed 0 is never the worst of the three on any
evaluation set.
