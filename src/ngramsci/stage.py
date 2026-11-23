"""The stage contract.

A stage turns the inputs of one shard into outputs. It declares:
  * `inputs`  : names of outputs of other stages it reads (e.g. "text", "windows"),
  * `outputs` : what it writes, each with a path pattern and a `private` flag,
  * `env`     : which environment kind it needs (propp, vllm, quotes, ...),
  * `gpu`     : whether it needs a GPU.
Public outputs go to the store shared by all machines (the HF hub board, or a shared disk). Private outputs (article
text, token tables: anything that contains the articles themselves) stay in the site's private store, so a stage
with private inputs only runs on a site that holds them.

A stage is done for a shard when its `done` output exists (written last, or in the same hub commit as the others).
`load()` runs once per process (models); `run()` once per shard."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Output:
    name: str
    pattern: str                 # relative path, formatted with k (shard number), e.g. "mentions/shard_{k:05d}.ner.parquet"
    private: bool = False

    def path(self, k: int, layout: dict | None = None) -> str:
        return (layout or {}).get(self.name, self.pattern).format(k=k)


@dataclass
class Ctx:
    """What a stage sees at run time."""
    corpus: Any                  # config.Corpus
    site: Any                    # config.Site (or None in tests)
    scratch: str                 # per-(stage, shard) dir that survives restarts: partial outputs can be resumed
    log: Callable[[str], None] = print
    deadline: float | None = None  # unix time after which the stage should stop at a safe point (raise Deadline)

    def p(self, stage: str, key: str, default=None):
        return self.corpus.p(stage, key, default)


class Deadline(Exception):
    """Raised by a stage that stops early (time limit) leaving resumable partial work in ctx.scratch."""


@dataclass
class Stage:
    name: str
    inputs: list[str]
    outputs: list[Output]
    env: str
    gpu: bool = True
    done: str | None = None      # name of the output that marks completion (default: the last one)
    ttl_hours: float = 6.0       # claim lifetime
    load: Callable[[Ctx], Any] = lambda ctx: None
    run: Callable[..., dict] = None   # run(k, inp: {name: path}, out: {name: path}, state, ctx) -> info dict
    doc: str = ""

    @property
    def done_output(self) -> Output:
        n = self.done or self.outputs[-1].name
        return next(o for o in self.outputs if o.name == n)

    def out(self, name: str) -> Output:
        return next(o for o in self.outputs if o.name == name)

    @property
    def local(self) -> bool:
        """Only private inputs and outputs: the stage never touches the board, so its claims stay on the site."""
        return all(o.private for o in self.outputs) and all(producer(n)[1].private for n in self.inputs)


REGISTRY: dict[str, Stage] = {}


def register(stage: Stage) -> Stage:
    REGISTRY[stage.name] = stage
    return stage


def producer(output_name: str) -> tuple[Stage | None, Output]:
    """The stage that writes an output, and its Output spec. 'text' is produced outside the pipeline (ship-text)."""
    load_all()
    for s in REGISTRY.values():
        for o in s.outputs:
            if o.name == output_name:
                return s, o
    if output_name == "text":
        return None, TEXT
    raise KeyError(f"no stage produces {output_name!r}")


TEXT = Output("text", "text/shard_{k:05d}.jsonl.gz", private=True)


def load_all():
    """Import the built-in stages (they register themselves). Imports are light: models load in Stage.load."""
    from .stages import rubriques, parse_ner, parse, qwen, assemble, quotes, quote_links, portrayal  # noqa: F401


def get(name: str) -> Stage:
    load_all()
    return REGISTRY[name]


def shard_name(k: int) -> str:
    return f"shard_{k:05d}"


def tmp_path(path: str) -> str:
    d, b = os.path.split(path)
    return os.path.join(d, f".{b}.tmp")
