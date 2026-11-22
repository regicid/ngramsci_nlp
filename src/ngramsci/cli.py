"""ngramsci command line.

  On a cluster (inside jobs):
    work       pull worker: run stages on whatever shards are ready (see worker.py)
  From the laptop (over ssh):
    launch     submit worker jobs to a site          status     board counts + queues of all sites
    ship-text  copy text shards from the source host (saclay) to a site's private store
  Anywhere:
    run        fused mode: stages one after another on a local dir (daily batches, tests)
    publish    push outputs from a disk store to the hub board in batched commits
    extract    (saclay) build a corpus from the scrape CSVs     shard: number the chunks as shards
    merge-qwen merge the coreference LoRA into its base model   export-portrayal: training checkpoints -> hub format"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from . import config as C


def _sites(a):
    return C.load_sites(a.sites)


def cmd_work(a):
    from .worker import Worker
    corpus = C.load_corpus(a.corpus); site = C.local_site(_sites(a), a.site)
    Worker(corpus, site, a.stages.split(","), start=a.start, reverse=a.reverse, max_hours=a.max_hours, poll=a.poll,
           max_shards=a.max_shards, batch_minutes=a.batch_minutes).loop()


def cmd_run(a):
    from .worker import run_local
    corpus = C.load_corpus(a.corpus)
    site = None
    if a.site or os.environ.get("NGRAMSCI_SITE"):
        site = C.local_site(_sites(a), a.site)
    shards = [int(x) for x in a.shards.split(",")]
    if a.input:                                         # one input file -> shard 0 of the run dir
        os.makedirs(os.path.join(a.root, "text"), exist_ok=True)
        shutil.copy(a.input, os.path.join(a.root, "text", "shard_00000.jsonl.gz")); shards = [0]
    run_local(corpus, a.stages.split(","), a.root, shards, site=site)


def cmd_launch(a):
    from .launch import launch
    corpus = C.load_corpus(a.corpus); sites = _sites(a)
    extra = dict(kv.split("=", 1) for kv in a.env) if a.env else None
    launch(corpus, a.corpus, sites, str(C.find_sites_file(a.sites)), a.site, a.stages.split(","), jobs=a.jobs,
           partition=a.partition, yes=a.yes, pull=not a.no_pull, dry=a.dry, offset=a.offset, total_jobs=a.total_jobs, extra_env=extra)


def cmd_status(a):
    from .launch import status
    from .store import board
    corpus = C.load_corpus(a.corpus); sites = _sites(a)
    status(corpus, {n: s for n, s in sites.items() if s.partitions}, board(corpus, None, outbox_root=os.path.expanduser("~/.cache/ngramsci/outbox")),
           a.stages.split(",") if a.stages else None)


def cmd_ship_text(a):
    """source host -> laptop -> site, as a tar stream (the text never touches the hub)."""
    corpus = C.load_corpus(a.corpus); sites = _sites(a)
    src = corpus.source; dst = sites[a.to]
    pat = src.get("pattern", "shard_{k:05d}.jsonl.gz")
    lo, hi = (int(x) for x in a.shards.split("-")) if a.shards else (0, corpus.n_shards - 1)
    files = " ".join(pat.format(k=k) for k in range(lo, hi + 1))
    dest = os.path.join(dst.private, corpus.name, "text")
    src_ssh = sites[src["host"]].ssh if src.get("host") in sites else src.get("host")
    pull = ["ssh", "-o", "BatchMode=yes", src_ssh, f"cd {src['dir']} && tar cf - {files}"]
    push = ["ssh", "-o", "BatchMode=yes", dst.ssh, f"mkdir -p {dest} && tar xf - -C {dest} && ls {dest} | wc -l"]
    p1 = subprocess.Popen(pull, stdout=subprocess.PIPE)
    out = subprocess.run(push, stdin=p1.stdout, capture_output=True, text=True)
    p1.wait()
    if p1.returncode or out.returncode:
        sys.exit(f"ship-text failed: {out.stderr.strip()[:300]}")
    print(f"{dst.name}: {out.stdout.strip()} text shards in {dest}")


def cmd_publish(a):
    """Disk store -> hub board, batched: up to --max shards per commit, every --period s, until nothing is left."""
    import time
    from huggingface_hub import CommitOperationAdd, HfApi
    from . import stage as S
    from .store import DiskStore, HubStore
    corpus = C.load_corpus(a.corpus); disk = DiskStore(a.root, corpus.layout)
    hub = HubStore(corpus.hub_repo, os.path.join(a.root, ".outbox"), corpus.layout); api = HfApi()
    stages = [S.get(n) for n in (a.stages.split(",") if a.stages else ["parse_ner", "qwen", "assemble", "quotes", "rubriques", "portrayal"])]
    while True:
        pending = []
        for st in stages:
            on_hub = hub.done(st.done_output, refresh=True)
            pending += [(st, k) for k in sorted(disk.done(st.done_output) - on_hub)]
        if not pending:
            print("nothing left to publish"); return
        batch = pending[:a.max]; ops = []
        for st, k in batch:
            ops += [CommitOperationAdd(o.path(k, corpus.layout), disk.path(o, k))
                    for o in sorted((o for o in st.outputs if not o.private), key=lambda o: o.name == st.done_output.name)]
        try:
            api.create_commit(corpus.hub_repo, repo_type="dataset", operations=ops, commit_message=f"publish {len(batch)} shard outputs")
            print(f"published {len(batch)} ({len(pending) - len(batch)} left)", flush=True)
            if len(pending) > len(batch):
                continue
            if not a.watch:
                return
        except Exception as e:
            print(f"commit failed: {type(e).__name__} {str(e)[:150]}", flush=True)
        time.sleep(a.period)


def cmd_extract(a):
    from .extract import extract
    out = os.path.realpath(a.out)
    if a.require_prefix and not out.startswith(a.require_prefix):
        sys.exit(f"refusing to write outside {a.require_prefix}: {out}")
    extract(a.csv_dir, out, medias=a.media.split(",") if a.media else None, exclude=a.exclude.split(",") if a.exclude else (),
            procs=a.procs, ids=a.ids, start=a.start, end=a.end, regex=a.regex, chunk=a.chunk)


def cmd_shard(a):
    from .extract import shard
    shard(a.chunks, a.shards)


def cmd_merge_qwen(a):
    from .tools import merge_qwen
    merge_qwen(a.adapter, a.out)


def cmd_export_portrayal(a):
    from .portrayal import export
    export(a.runs, a.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ngramsci", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", help="sites file (default: $NGRAMSCI_SITES, ./sites.local.yaml)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("work"); p.set_defaults(f=cmd_work)
    p.add_argument("--corpus", required=True); p.add_argument("--stages", required=True); p.add_argument("--site")
    p.add_argument("--start", type=int, default=0); p.add_argument("--reverse", action="store_true")
    p.add_argument("--max-hours", type=float, default=23.0); p.add_argument("--poll", type=int, default=300)
    p.add_argument("--max-shards", type=int)
    p.add_argument("--batch-minutes", type=float, default=30.0, help="claim about this much work per hub commit")

    p = sp.add_parser("run"); p.set_defaults(f=cmd_run)
    p.add_argument("--corpus", required=True); p.add_argument("--stages", required=True); p.add_argument("--root", required=True)
    p.add_argument("--shards", default="0"); p.add_argument("--input", help="a text jsonl.gz to process as shard 0"); p.add_argument("--site")

    p = sp.add_parser("launch"); p.set_defaults(f=cmd_launch)
    p.add_argument("--corpus", required=True); p.add_argument("--stages", required=True); p.add_argument("--site", required=True)
    p.add_argument("--jobs", type=int, default=1); p.add_argument("--partition"); p.add_argument("--yes", action="store_true")
    p.add_argument("--dry", action="store_true"); p.add_argument("--no-pull", action="store_true")
    p.add_argument("--offset", type=int, default=0, help="index of the first job among all jobs of this launch wave")
    p.add_argument("--total-jobs", type=int, help="jobs in the whole wave (all sites), to spread start shards")
    p.add_argument("--env", nargs="*", help="extra KEY=VALUE for the job")

    p = sp.add_parser("status"); p.set_defaults(f=cmd_status)
    p.add_argument("--corpus", required=True); p.add_argument("--stages")

    p = sp.add_parser("ship-text"); p.set_defaults(f=cmd_ship_text)
    p.add_argument("--corpus", required=True); p.add_argument("--to", required=True); p.add_argument("--shards", help="a-b")

    p = sp.add_parser("publish"); p.set_defaults(f=cmd_publish)
    p.add_argument("--corpus", required=True); p.add_argument("--root", required=True); p.add_argument("--stages")
    p.add_argument("--max", type=int, default=60); p.add_argument("--period", type=int, default=600); p.add_argument("--watch", action="store_true")

    p = sp.add_parser("extract"); p.set_defaults(f=cmd_extract)
    p.add_argument("--csv-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--media"); p.add_argument("--exclude")
    p.add_argument("--ids"); p.add_argument("--start", default=""); p.add_argument("--end", default="9999"); p.add_argument("--regex")
    p.add_argument("--chunk", type=int, default=5000); p.add_argument("--procs", type=int, default=12)
    p.add_argument("--require-prefix", default="/data/", help="refuse outputs outside this prefix (saclay: the root disk is small)")

    p = sp.add_parser("shard"); p.set_defaults(f=cmd_shard); p.add_argument("chunks"); p.add_argument("shards")
    p = sp.add_parser("merge-qwen"); p.set_defaults(f=cmd_merge_qwen)
    p.add_argument("--adapter", default="lattice-nlp/Qwen3.5-4B_inline_coreference_resolution"); p.add_argument("--out", required=True)
    p = sp.add_parser("export-portrayal"); p.set_defaults(f=cmd_export_portrayal)
    p.add_argument("runs", nargs="+"); p.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
