"""The generic pull worker, identical on every machine.

Loop: pick a (stage, shard) whose done-output is missing, whose inputs exist (public ones on the board, private ones
in this site's private store) and that nobody claimed recently -> claim -> fetch inputs -> run -> publish -> repeat.
Shards are claimed in batches of about --batch-minutes of work (sized from this worker's measured shard times; one
shard until a first one is timed), so a worker commits to the hub about once per batch.
Workers start at different shards (--start) and wrap around, so they spread over the shards and then steal what is
left. A worker exits when every shard of its stages is done, or before --max-hours so a Slurm limit never kills it
half-way (partial work stays in its scratch dir and is resumed by the next worker on this site).

`run_local` runs stages one after another in one process on a local directory: the fused mode used for daily
batches and tests (models loaded once, no claims, no hub)."""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback

from . import stage as S
from .store import DiskStore, board as make_board, private_store


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Worker:
    def __init__(self, corpus, site, stage_names, start=0, reverse=False, max_hours=23.0, poll=300,
                 max_shards=None, board=None, private=None, shards=None, batch_minutes=30.0, max_batch=20):
        self.c, self.site = corpus, site
        self.stages = [S.get(n) for n in stage_names]
        self.board = board or make_board(corpus, site)
        self.priv = private or private_store(corpus, site)
        n = corpus.n_shards
        ks = list(shards) if shards is not None else list(range(n))
        i0 = ks.index(start) if start in ks else 0
        self.order = ks[i0:] + ks[:i0]
        if reverse:
            self.order = self.order[:1] + self.order[1:][::-1]
        self.t0, self.max_s, self.poll = time.time(), max_hours * 3600, poll
        self.max_shards, self.n_done = max_shards, 0
        self.states, self.failed = {}, set()
        self.batch_s, self.max_batch = batch_minutes * 60, max_batch
        self.queue: list = []                                 # (stage, k) claimed by us, not started yet
        self.secs: dict[str, list[float]] = {}               # stage -> seconds per shard, measured here
        self.work = site.work if site else "."

    # where things are --------------------------------------------------------------------------------------
    def store_of(self, o: S.Output):
        return self.priv if o.private else self.board

    def claimer(self, st):                                   # site-local stages: disk locks, no hub commits
        return self.priv if st.local else self.board

    def available(self, name: str) -> set[int]:
        _, o = S.producer(name)
        return self.store_of(o).done(o)

    def scratch(self, st, k):
        return os.path.join(self.work, "scratch", self.c.name, st.name, S.shard_name(k))

    def mine(self, st, k):                                   # partial work of an earlier worker on this site: resume it
        d = self.scratch(st, k)
        return os.path.isdir(d) and any(os.scandir(d))

    # picking -----------------------------------------------------------------------------------------------
    def batch_size(self, st) -> int:
        t = self.secs.get(st.name)
        if not t:
            return 1
        return max(1, min(self.max_batch, round(self.batch_s / (sum(t) / len(t)))))

    def release_queue(self):
        for st, k in self.queue:
            self.claimer(st).release(st.name, k)
        self.queue = []

    def pick(self):
        """(stage, k, waiting): the next shard of our queue, else a newly claimed batch. stage None and waiting True:
        inputs are missing, or the board could not be reached (retry later)."""
        while self.queue:
            st, k = self.queue.pop(0)
            if k not in self.store_of(st.done_output).done(st.done_output):
                return st, k, False
            self.claimer(st).release(st.name, k)             # finished elsewhere meanwhile
        waiting = False
        for st in self.stages:
            done = self.store_of(st.done_output).done(st.done_output)
            todo = [k for k in self.order if k not in done and (st.name, k) not in self.failed]
            if not todo:
                continue
            ready = set(todo)
            for name in st.inputs:
                ready &= self.available(name)
            if len(ready) < len(todo):
                waiting = True
            if not ready:
                continue
            ttl = self.c.ttl_hours.get(st.name, st.ttl_hours)
            cl = self.claimer(st)
            claims = cl.claims(st.name)
            # Partial work left on this site goes first, but it is claimed like any other shard: a non-empty scratch
            # dir may belong to a worker still running it (two workers on one shard -> the first to finish deletes
            # the dir under the other). A worker stopping at its deadline releases its claim, so resuming is immediate.
            free = [k for k in sorted(todo, key=lambda k: not self.mine(st, k))
                    if k in ready and not (claims.get(k) and time.time() - claims[k][0] < ttl * 3600)]
            if not free:
                continue
            won = cl.claim_many(st.name, free[:self.batch_size(st)], ttl)
            if won is None:
                return None, None, True                       # board unreachable (e.g. HTTP 429): retry later
            done = self.store_of(st.done_output).done(st.done_output, refresh=True)
            for k in won:
                if k in done:
                    cl.release(st.name, k)                    # finished meanwhile (our listing was cached)
                else:
                    self.queue.append((st, k))
            if self.queue:
                _log(f"{st.name}: claimed {[k for _, k in self.queue]}")
                st, k = self.queue.pop(0)
                return st, k, waiting
            return None, None, True                           # lost every race: look again
        return None, None, waiting

    # running -----------------------------------------------------------------------------------------------
    def state(self, st, ctx):
        if st.name not in self.states:
            _log(f"loading {st.name}")
            self.states[st.name] = st.load(ctx)
        return self.states[st.name]

    def run_one(self, st, k):
        scratch = self.scratch(st, k); os.makedirs(scratch, exist_ok=True)
        ctx = S.Ctx(corpus=self.c, site=self.site, scratch=scratch, log=_log, deadline=self.t0 + self.max_s)
        cache = os.path.join(self.work, "cache", self.c.name)
        inp = {}
        for name in st.inputs:
            _, o = S.producer(name)
            inp[name] = self.store_of(o).fetch(o, k, cache)
        outdir = os.path.join(scratch, "out"); os.makedirs(outdir, exist_ok=True)
        out = {o.name: os.path.join(outdir, f"{o.name}__{os.path.basename(o.path(k, self.c.layout))}") for o in st.outputs}
        t = time.time()
        info = st.run(k, inp, out, self.state(st, ctx), ctx) or {}
        self.secs.setdefault(st.name, []).append(time.time() - t)
        info.update(seconds=round(time.time() - t), who=getattr(self.board, "who", ""), stage=st.name, shard=k)
        missing = [n for n, p in out.items() if not os.path.exists(p)]
        if missing:
            raise RuntimeError(f"{st.name} shard {k} did not write {missing}")
        pub = [(o, out[o.name]) for o in st.outputs if not o.private]
        prv = [(o, out[o.name]) for o in st.outputs if o.private]
        if prv:
            self.priv.publish(st.name, k, prv, st.done_output if st.done_output.private else prv[-1][0])
        if pub:
            self.board.publish(st.name, k, pub, st.done_output if not st.done_output.private else pub[-1][0])
        elif not st.local:
            self.board.release(st.name, k)
        for name, p in inp.items():                              # downloaded inputs are not kept
            if p.startswith(cache) and os.path.exists(p):
                os.remove(p)
        shutil.rmtree(scratch, ignore_errors=True)
        _log(f"{st.name} shard {k}: {json.dumps(info)}")
        return info

    def loop(self):
        try:
            self._loop()
        finally:                                                 # also on a crash: free our batch, push outputs
            self.release_queue()
            for _ in range(12):                                  # last outputs must reach the board
                if self.board.flush():
                    break
                time.sleep(max(60, getattr(self.board, "wait_s", lambda: 0)()))

    def _loop(self):
        fails = 0
        while True:
            if time.time() - self.t0 > self.max_s:
                _log("max hours reached"); break
            if self.max_shards is not None and self.n_done >= self.max_shards:
                break
            try:
                st, k, waiting = self.pick()
            except RuntimeError as e:                             # listing failed after retries (hub rate limit)
                _log(f"pick failed ({e}), retrying in {self.poll}s"); time.sleep(self.poll); continue
            if st is None:
                if not waiting:
                    _log("ALL SHARDS DONE"); break
                self.board.flush(force=False)
                wait = max(self.poll, getattr(self.board, "wait_s", lambda: 0)())
                _log(f"waiting for inputs or the board ({wait:.0f}s)"); time.sleep(wait); continue
            try:
                self.run_one(st, k); self.n_done += 1; fails = 0
                self.board.flush(force=False)                     # timed: outputs of a long batch still go out
            except S.Deadline:
                _log(f"{st.name} shard {k}: stopped at the deadline, partial work kept")
                self.claimer(st).release(st.name, k); break
            except Exception:
                traceback.print_exc(); fails += 1
                self.failed.add((st.name, k)); self.claimer(st).release(st.name, k)
                if fails >= 3:
                    _log("3 failures in a row, exiting"); break


def run_local(corpus, stage_names, root, shards=(0,), site=None):
    """Fused mode: all stages, in order, in this process, on <root> (private and public outputs side by side)."""
    st = DiskStore(root, corpus.layout)
    w = Worker(corpus, site, stage_names, board=st, private=st, shards=list(shards), poll=0, max_hours=1e6)
    w.work = root
    for name in stage_names:
        w.stages = [S.get(name)]
        for k in shards:
            if k not in st.done(w.stages[0].done_output):
                w.run_one(w.stages[0], k)
    return st
