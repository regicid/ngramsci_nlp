"""The generic pull worker, identical on every machine.

Loop: pick a (stage, shard) whose done-output is missing, whose inputs exist (public ones on the board, private ones
in this site's private store) and that nobody claimed recently -> claim -> fetch inputs -> run -> publish -> repeat.
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
                 max_shards=None, board=None, private=None, shards=None):
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
    def pick(self):
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
            for k in todo:
                if k not in ready:
                    continue
                if self.mine(st, k):
                    return st, k, waiting
                c = claims.get(k)
                if c and time.time() - c[0] < ttl * 3600:
                    continue
                if cl.claim(st.name, k, ttl):
                    if k in self.store_of(st.done_output).done(st.done_output, refresh=True):
                        cl.release(st.name, k); continue      # finished meanwhile (our listing was cached)
                    return st, k, waiting
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
        fails = 0
        while True:
            if time.time() - self.t0 > self.max_s:
                _log("max hours reached"); break
            if self.max_shards is not None and self.n_done >= self.max_shards:
                break
            st, k, waiting = self.pick()
            if st is None:
                if not waiting:
                    _log("ALL SHARDS DONE"); break
                self.board.flush(); _log(f"waiting for inputs ({self.poll}s)"); time.sleep(self.poll); continue
            try:
                self.run_one(st, k); self.n_done += 1; fails = 0
            except S.Deadline:
                _log(f"{st.name} shard {k}: stopped at the deadline, partial work kept"); break
            except Exception:
                traceback.print_exc(); fails += 1
                self.failed.add((st.name, k)); self.claimer(st).release(st.name, k)
                if fails >= 3:
                    _log("3 failures in a row, exiting"); break
        for _ in range(12):                                      # last outputs must reach the board
            if self.board.flush():
                break
            time.sleep(300)


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
