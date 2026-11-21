# ngramsci_nlp

French press articles in, people out: sections, mentions, coreference chains, gender, quotes and how each person is
portrayed. Runs on one GPU for a day's articles, or across several clusters for millions.

## Stages

| stage | does | model |
|---|---|---|
| `rubriques` | section (6 classes) | [regicid/bert-rubriques](https://huggingface.co/regicid/bert-rubriques) |
| `parse_ner` | parse, mentions (people, places…), coreference windows | [fr_dep_news_trf](https://huggingface.co/spacy/fr_dep_news_trf), [propp-fr NER](https://huggingface.co/AntoineBourgois/propp-fr_NER_camembert-large_FAC_GPE_LOC_PER_TIME_VEH) on [camembert-large](https://huggingface.co/almanach/camembert-large) |
| `qwen` | coreference | [lattice-nlp/Qwen3.5-4B_inline_coreference_resolution](https://huggingface.co/lattice-nlp/Qwen3.5-4B_inline_coreference_resolution) (LoRA on [unsloth/Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B)) |
| `assemble` | characters, their attributes (agent, patient, modifiers…), gender | [propp-fr](https://github.com/lattice-8094/propp), `ngramsci.gender` |
| `quotes` | quotes, speaker, cue | GenderedNews REBEL ([Zenodo 15228575](https://zenodo.org/records/15228575), on [BARThez](https://huggingface.co/moussaKam/barthez)), CTranslate2 |
| `portrayal` | per mention: relevance, valence, extreme/moderate framing, momentum, author's own judgement | [n-gramsci/portrayal-camembert-large](https://huggingface.co/n-gramsci/portrayal-camembert-large) |

`text → rubriques, quotes, parse_ner → qwen → assemble → portrayal`. Sections of the existing corpus:
[n-gramsci/rubriques-presse](https://huggingface.co/datasets/n-gramsci/rubriques-presse).

## One machine

```bash
pip install -e ".[ner,assemble]"            # one env per stage kind, see envs/build_env.sh
ngramsci run --corpus daily --stages parse_ner,qwen,assemble,quotes,portrayal --root runs/2026-11-22 --input new.jsonl.gz
```

Input: `jsonl.gz`, one article per line: `{doc_id, text, media, date}`, with `text` = `title///body`.

## Several clusters

Pull workers share a **board** (a private HF dataset repo): each takes a shard whose inputs are ready, claims it, runs it
and publishes the outputs, so the machines never talk to each other. Article text and token tables are *private*
outputs: they stay on the site that made them and never go to the hub.

```bash
cp sites.example.yaml sites.local.yaml                   # ssh aliases, paths, envs, partitions
ngramsci ship-text --corpus politique --to cca           # text: saclay -> cca, over ssh
ngramsci launch --corpus politique --stages parse_ner --site cca --jobs 14
ngramsci launch --corpus politique --stages qwen --site ruche --jobs 8
ngramsci status --corpus politique
```

Corpora are YAML files in `corpora/`. Partitions marked `ask: true` (quota-limited hours) need `--yes`. On network
filesystems, pack envs as squashfs (`envs/pack_squashfs.sh`) so that vLLM starts in minutes rather than half an hour.

## Licence

MIT. Portrayal annotation codebook: `docs/codebook_v4.md`.
