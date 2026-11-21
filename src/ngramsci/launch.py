"""Launch workers on remote sites over ssh, and report status.

`launch` renders one Slurm script per job from the site file (partition options, environment activation, the site's
`pre` lines), pulls the site's git checkout, copies the corpus and site files next to it, and runs sbatch. Each job
runs `partition.workers` worker processes, each with its own starting shard, spread over all jobs so workers start far
apart. Partitions marked `ask: true` (quota-limited hours) need --yes.
`status` prints, per stage, shards done on the board, and the ngramsci jobs in each site's queue.
ssh runs with BatchMode: it never prompts; open the ssh control sockets first (`ssh <alias> true`)."""
from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from . import stage as S


def ssh(site, cmd: str, stdin: str | None = None, timeout=120) -> str:
    if site.ssh is None:
        args = ["bash", "-lc", cmd]
    else:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", site.ssh, f"bash -lc {shlex.quote(cmd)}"]
    r = subprocess.run(args, input=stdin, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{site.name}: `{cmd[:80]}` failed: {r.stderr.strip()[:300]}")
    return r.stdout


def activation(site, kind: str) -> str:
    e = site.envs.get(kind)
    if e is None:
        raise SystemExit(f"site {site.name} has no env {kind!r} (sites file: envs.{kind})")
    if isinstance(e, dict) and "squashfs" in e:
        return f"source {site.code}/envs/activate_sqfs.sh {e['squashfs']} {e.get('python_squashfs', '')}"
    if isinstance(e, dict):
        return e["activate"]
    return e


def pick_partition(site, stage_names, name=None):
    if name:
        return site.partitions[name]
    for pn in site.stages.get(stage_names[0], []):
        if pn in site.partitions:
            return site.partitions[pn]
    raise SystemExit(f"site {site.name} lists no partition for {stage_names[0]} (sites file: stages.{stage_names[0]})")


def render(site, part, corpus_file, sites_file, stage_names, starts, job_name, extra_env=None) -> str:
    stages = [S.get(n) for n in stage_names]
    kinds = {s.env for s in stages}
    if len(kinds) > 1:
        raise SystemExit(f"stages {stage_names} need different envs {kinds}: launch them separately")
    env = {"PYTHONUNBUFFERED": "1", "NGRAMSCI_SITE": site.name, "NGRAMSCI_SITES": sites_file, **(extra_env or {})}
    lines = ["#!/bin/bash", f"#SBATCH --job-name={job_name}", f"#SBATCH --output={site.logs}/%x_%j.out"]
    lines += [f"#SBATCH {o}" for o in part.sbatch]
    lines += [site.pre or "", activation(site, kinds.pop()), f"export PYTHONPATH={site.code}/src:${{PYTHONPATH:-}}"]
    lines += [f"export {k}={shlex.quote(str(v))}" for k, v in env.items()]
    lines += [f"cd {site.code}", 'echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"']
    for j, (k, rev) in enumerate(starts):
        lines.append(f"python -m ngramsci.cli work --corpus {corpus_file} --stages {','.join(stage_names)} --start {k}"
                     f"{' --reverse' if rev else ''} --max-hours {part.max_hours} &")
        if j + 1 < len(starts):
            lines.append("sleep 90")                # staggered: workers do not pick the same shard at the same instant
    lines += ["wait", ""]
    return "\n".join(l for l in lines if l is not None)


def launch(corpus, corpus_file, sites, sites_file, site_name, stage_names, jobs=1, partition=None, yes=False,
           pull=True, dry=False, offset=0, total_jobs=None, extra_env=None):
    site = sites[site_name]
    part = pick_partition(site, stage_names, partition)
    if part.ask and not yes:
        raise SystemExit(f"{site.name}:{part.name} is quota-limited: add --yes to really launch there")
    n_workers = (total_jobs or jobs) * part.workers
    remote_dir = f"{site.work}/launch"
    rc, rs = f"{remote_dir}/{Path(corpus_file).name}", f"{remote_dir}/sites.yaml"
    scripts = []
    for j in range(jobs):
        starts = []
        for w in range(part.workers):
            i = (offset + j) * part.workers + w
            starts.append((i * corpus.n_shards // max(1, n_workers), i % 2 == 1))
        name = f"ng-{corpus.name}-{'+'.join(stage_names)}-{site.name}-{offset + j}"
        scripts.append((name, render(site, part, rc, rs, stage_names, starts, name, extra_env)))
    if dry:
        for n, s in scripts:
            print(f"----- {n}\n{s}")
        return []
    if pull:
        print(ssh(site, f"cd {site.code} && git pull --ff-only -q && git log -1 --oneline").strip())
    ssh(site, f"mkdir -p {remote_dir} {site.logs}")
    ssh(site, f"cat > {rc}", stdin=Path(corpus_file).read_text())
    ssh(site, f"cat > {rs}", stdin=Path(sites_file).read_text())
    ids = []
    for name, s in scripts:
        path = f"{remote_dir}/{name}.sh"
        ssh(site, f"cat > {path}", stdin=s)
        out = ssh(site, f"sbatch {path}")
        ids.append(out.strip().split()[-1] if out.strip() else "?")
        print(f"{site.name}: {name} -> job {ids[-1]}")
    return ids


def queue(site) -> list[str]:
    try:
        out = ssh(site, "squeue -u $USER -h -o '%i|%j|%T|%M|%R'", timeout=60)
    except Exception as e:
        return [f"(unreachable: {str(e)[:80]})"]
    return [l for l in out.splitlines() if "|ng-" in l]


def status(corpus, sites, board, stage_names=None):
    S.load_all()
    names = stage_names or [n for n in ("rubriques", "parse_ner", "qwen", "assemble", "quotes", "portrayal") if n in S.REGISTRY]
    print(f"{corpus.name}: {corpus.n_shards} shards  ({time.strftime('%H:%M')})")
    for n in names:
        st = S.get(n); o = st.done_output
        done = "private" if o.private else len(board.done(o, refresh=True))
        claimed = len(board.claims(n)) if not o.private else "-"
        print(f"  {n:<10} done {done!s:>5}  claimed {claimed!s:>4}")
    for s in sites.values():
        q = queue(s)
        print(f"  {s.name}: {len(q)} jobs" + "".join(f"\n      {l}" for l in q))
