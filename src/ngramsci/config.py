"""Corpus and site configuration (YAML).

A *corpus* says what to process: shard count, the hub repo used as the shared board, where the text comes from, and
per-stage parameters. A *site* says where things run: ssh alias, checkout and work dirs, environments, partitions.
Both are plain YAML so the same code runs everywhere; site files hold paths and logins and stay out of git
(`sites.local.yaml`, see `sites.example.yaml`)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Corpus:
    name: str
    n_shards: int
    hub_repo: str | None = None            # private HF dataset repo used as the board (None -> disk-only corpus)
    source: dict = field(default_factory=dict)   # where shipped text comes from: {host, dir, pattern}
    params: dict = field(default_factory=dict)   # per-stage parameters: {stage_name: {...}}
    layout: dict = field(default_factory=dict)   # per-output path overrides: {output_name: "dir/shard_{k:05d}.ext"}
    ttl_hours: dict = field(default_factory=dict)  # claim lifetime per stage

    def p(self, stage: str, key: str, default: Any = None) -> Any:
        return (self.params.get(stage) or {}).get(key, default)


@dataclass
class Partition:
    name: str
    sbatch: list[str]                      # raw #SBATCH options
    gpu: str | None = None                 # v100 / a100 / h100 / l40s / ada / None (cpu)
    workers: int = 1                       # worker processes per job (2 per V100 for NER)
    max_hours: float = 23.0                # workers stop cleanly before the Slurm limit
    ask: bool = False                      # needs explicit --yes (quota-limited hours)


@dataclass
class Site:
    name: str
    ssh: str | None                        # ssh alias (None = local machine)
    code: str                              # git checkout of this repo on the site
    work: str                              # per-site work dir: private store, caches, logs
    envs: dict = field(default_factory=dict)       # env kind -> activation (str) or {squashfs, python_squashfs}
    partitions: dict = field(default_factory=dict) # name -> Partition
    stages: dict = field(default_factory=dict)     # stage -> preferred partition names, best first
    pre: str = ""                          # shell lines run first in every job (HF_HOME, proxies, modules...)
    models: dict = field(default_factory=dict)     # local model dirs: {qwen: merged model dir, rebel: rebel pack dir}
    scheduler: str = "slurm"

    @property
    def private(self) -> str:              # site-local store for private outputs (text, token tables)
        return os.path.join(self.work, "private")

    @property
    def logs(self) -> str:
        return os.path.join(self.work, "logs")


def _expand(v):
    if isinstance(v, str):
        return os.path.expandvars(os.path.expanduser(v))
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_corpus(path_or_name: str) -> Corpus:
    p = Path(path_or_name)
    if not p.exists():
        p = Path(__file__).resolve().parents[2] / "corpora" / f"{path_or_name}.yaml"
    d = yaml.safe_load(p.read_text())
    return Corpus(**d)


def find_sites_file(explicit: str | None = None) -> Path:
    for c in [explicit, os.environ.get("NGRAMSCI_SITES"), "sites.local.yaml",
              Path(__file__).resolve().parents[2] / "sites.local.yaml", Path.home() / ".config/ngramsci/sites.yaml"]:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("no sites file: copy sites.example.yaml to sites.local.yaml (or set NGRAMSCI_SITES)")


def load_sites(path: str | None = None) -> dict[str, Site]:
    raw = yaml.safe_load(find_sites_file(path).read_text())
    out = {}
    for name, d in (raw.get("sites") or raw).items():
        d = dict(d)
        parts = {pn: Partition(name=pn, **pd_) for pn, pd_ in (d.pop("partitions", {}) or {}).items()}
        out[name] = Site(name=name, partitions=parts, **_expand(d))
    return out


def local_site(sites: dict[str, Site], name: str | None = None) -> Site:
    name = name or os.environ.get("NGRAMSCI_SITE")
    if not name:
        raise SystemExit("which site is this? pass --site or set NGRAMSCI_SITE")
    return sites[name]
