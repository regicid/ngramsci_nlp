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
Five seeds of the same fine-tune; average their probabilities. Two seeds are enough: on the gold set the mean of any
two seeds matches the five-seed ensemble (one seed alone loses 0.015–0.045 κ); production uses seeds 0 and 1.

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

* Teacher labels: Claude Sonnet following the codebook, on 6,250 mention passages from French online press (2008–2026)
  about the 2027 presidential candidates; 2,000 of them picked by active learning (rare classes: moderate framing,
  decline, praise).
* Training: 5,220 train / 941 validation rows (rows where the mention designates someone else were dropped), 12 epochs,
  last epoch kept, fp16, CamemBERT-large, seeds 0–4.
* Gold set: 596 mentions from 2026 articles kept out of training, labelled by the same teacher. Cohen's κ against the teacher
  (single seeds, last epoch):

| | pertinent | valence | extremisme | dynamique | journaliste |
|---|---|---|---|---|---|
| κ (5 seeds) | 0.65–0.71 | 0.62–0.67 | 0.56–0.65 | 0.61–0.64 | 0.78–0.85 |

For reference, Claude Opus against the Sonnet teacher on 200 gold mentions: valence 0.74, extremisme 0.77, dynamique 0.80,
pertinent 0.73, journaliste 0.67. The gold set has few rare-class cases (5 moderate, 16 declining), so the κ of those
classes is noisy (95 % CI ±0.06–0.12): judge the model on aggregates (candidate × period means), where it tracks the
teacher closely.

## Limitations

* Trained on politicians (2027 candidates). `valence`, `pertinent` and `journaliste` transfer reasonably to other people;
  `extremisme` and `dynamique` are about political framing and the electoral race and mean little for non-politicians.
* Labels come from an LLM teacher, not from human coders; the model reproduces the teacher's reading of the codebook.
* French online press only. The training passages are not released (copyrighted articles).
