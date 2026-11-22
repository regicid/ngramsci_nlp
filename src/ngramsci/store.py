"""Stores: where shard outputs live and how workers claim shards.

* DiskStore — a directory (a shared disk on one cluster, or a site's private store). Claims are mkdir locks.
* HubStore  — a private HF dataset repo, the board shared by all machines. Outputs wait in an on-disk outbox and
  are committed in batches: the hub allows 128 commits/hour per repo, so a worker makes one commit per shard
  (outputs of shard k + release of its claim + claim of the next shard). Claims are files
  claims/<stage>/shard_K__<unix time>__<who>: one listing gives who claimed what and when, without per-file calls.
  Claims are not atomic: after claiming, a worker re-lists and backs off if an older live claim exists (the only
  cost of a race is one duplicated shard).
Both expose the same interface, so the worker does not care which one it talks to."""
from __future__ import annotations

import os
import re
import shutil
import socket
import time
from pathlib import Path

from .stage import Output, shard_name


def pattern_regex(pattern: str) -> re.Pattern:
    """'mentions/shard_{k:05d}.ner.parquet' -> regex on the file name capturing k."""
    base = os.path.basename(pattern)
    parts = re.split(r"\{k(?::0?(\d+)d)?\}", base)
    rx = "".join(re.escape(p) if i % 2 == 0 else r"(\d+)" for i, p in enumerate(parts) if p is not None)
    return re.compile("^" + rx + "$")


def _shards_in(names, pattern: str) -> set[int]:
    rx = pattern_regex(pattern)
    out = set()
    for n in names:
        m = rx.match(n)
        if m:
            out.add(int(m.group(1)))
    return out


def who_am_i() -> str:
    return f"{socket.gethostname().split('.')[0]}-{os.getpid()}"


class Store:
    layout: dict = {}

    def rel(self, o: Output, k: int) -> str:
        return o.path(k, self.layout)

    # interface ---------------------------------------------------------------------------------------------
    def done(self, o: Output, refresh: bool = False) -> set[int]: ...
    def fetch(self, o: Output, k: int, dest: str) -> str: ...
    def publish(self, stage: str, k: int, files: list[tuple[Output, str]], done: Output): ...
    def flush(self, claim: tuple[str, int] | None = None) -> bool: ...
    def claims(self, stage: str) -> dict[int, tuple[float, str]]: ...
    def claim(self, stage: str, k: int, ttl_h: float) -> bool: ...
    def release(self, stage: str, k: int): ...


class DiskStore(Store):
    def __init__(self, root: str, layout: dict | None = None):
        self.root, self.layout, self.who = root, layout or {}, who_am_i()
        os.makedirs(root, exist_ok=True)

    def path(self, o: Output, k: int) -> str:
        return os.path.join(self.root, self.rel(o, k))

    def done(self, o, refresh=False):
        d = os.path.join(self.root, os.path.dirname(o.path(0, self.layout)))
        try:
            return _shards_in(os.listdir(d), self.layout.get(o.name, o.pattern))
        except FileNotFoundError:
            return set()

    def fetch(self, o, k, dest=None):
        p = self.path(o, k)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p

    def publish(self, stage, k, files, done):
        """Move files into place atomically, the done output last."""
        for o, src in sorted(files, key=lambda f: f[0].name == done.name):
            dst = self.path(o, k)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        self.release(stage, k)

    def flush(self, claim=None):
        if claim:
            return self.claim(claim[0], claim[1], 1e9)
        return True

    def _lock(self, stage, k):
        return os.path.join(self.root, "claims", stage, shard_name(k))

    def claims(self, stage):
        d = os.path.join(self.root, "claims", stage); out = {}
        for n in (os.listdir(d) if os.path.isdir(d) else []):
            p = os.path.join(d, n)
            try:
                out[int(n.split("_")[1])] = (os.path.getmtime(p), open(os.path.join(p, "who")).read().strip())
            except (OSError, ValueError, IndexError):
                pass
        return out

    def claim(self, stage, k, ttl_h):
        lock = self._lock(stage, k)
        os.makedirs(os.path.dirname(lock), exist_ok=True)
        try:
            os.mkdir(lock)                                         # atomic on NFS/Lustre/GPFS/Ceph
        except FileExistsError:
            if time.time() - os.path.getmtime(lock) < ttl_h * 3600:
                return False
            os.utime(lock)                                         # stale: take it over
        Path(lock, "who").write_text(self.who)
        return True

    def touch(self, stage, k):
        try:
            os.utime(self._lock(stage, k))
        except OSError:
            pass

    def release(self, stage, k):
        """Remove the lock if it is ours (a stale lock taken over by another worker is theirs now)."""
        lock = self._lock(stage, k)
        try:
            if open(os.path.join(lock, "who")).read().strip() != self.who:
                return
        except OSError:
            pass                                                   # no who file: half-made lock, remove it
        shutil.rmtree(lock, ignore_errors=True)


class HubStore(Store):
    """Private HF dataset repo as the board. outbox: local dir (persistent) holding files not yet committed."""
    LIST_EVERY = 120                      # s: listings are cached this long (1000 API calls / 5 min per account)

    def __init__(self, repo: str, outbox: str, layout: dict | None = None, api=None):
        from huggingface_hub import HfApi
        self.repo, self.layout, self.api, self.who = repo, layout or {}, api or HfApi(), who_am_i()
        self.outbox = os.path.join(outbox, self.who); os.makedirs(self.outbox, exist_ok=True)
        self._cache: dict[str, tuple[float, list[str]]] = {}
        self._pending_ops: list = []      # claim deletions/additions to send with the next commit
        self._released: set = set()
        self.adopt_orphans(outbox)

    # listings ----------------------------------------------------------------------------------------------
    def _list(self, d: str, refresh=False) -> list[str]:
        t, names = self._cache.get(d, (0, None))
        if names is not None and not refresh and time.time() - t < self.LIST_EVERY:
            return names
        for i in range(6):
            try:
                names = [f.path.split("/")[-1] for f in self.api.list_repo_tree(self.repo, repo_type="dataset", path_in_repo=d)]
                break
            except Exception as e:                                 # 404 = folder not created yet
                if "404" in str(e) or "not found" in str(e).lower():
                    names = []; break
                time.sleep(30 * (i + 1))
        else:
            raise RuntimeError(f"cannot list {self.repo}/{d}")
        self._cache[d] = (time.time(), names)
        return names

    def done(self, o, refresh=False):
        pat = self.layout.get(o.name, o.pattern)
        got = _shards_in(self._list(os.path.dirname(pat), refresh), pat)
        local = os.path.join(self.outbox, os.path.dirname(pat))            # waiting in our outbox counts as done
        if os.path.isdir(local):
            got |= _shards_in(os.listdir(local), pat)
        return got

    def fetch(self, o, k, dest):
        from huggingface_hub import hf_hub_download
        rel = self.rel(o, k); loc = os.path.join(self.outbox, rel)
        if os.path.exists(loc):
            return loc
        for i in range(6):
            try:
                return hf_hub_download(self.repo, rel, repo_type="dataset", local_dir=dest)
            except Exception as e:
                if i == 5:
                    raise
                print(f"download {rel} failed ({type(e).__name__}), retrying", flush=True); time.sleep(30 * (i + 1))

    # publishing --------------------------------------------------------------------------------------------
    def publish(self, stage, k, files, done):
        for o, src in files:
            dst = os.path.join(self.outbox, self.rel(o, k))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        self.release(stage, k)

    def adopt_orphans(self, root, idle_s=3600):
        """Outboxes of dead workers (no change for an hour) are merged into ours, so their outputs still get published."""
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if p == self.outbox or not os.path.isdir(p):
                continue
            newest = max((os.path.getmtime(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs), default=0)
            if time.time() - newest > idle_s:
                for r, _, fs in os.walk(p):
                    for f in fs:
                        src = os.path.join(r, f); dst = os.path.join(self.outbox, os.path.relpath(src, p))
                        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(src, dst)
                shutil.rmtree(p, ignore_errors=True)

    def _outbox_files(self):
        return [os.path.join(r, f) for r, _, fs in os.walk(self.outbox) for f in fs]

    def flush(self, claim=None):
        """One commit: everything in the outbox + pending claim releases + (optionally) a new claim.
        On failure (e.g. HTTP 429) the outbox is kept and the call returns False."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        files = self._outbox_files()
        ops = [CommitOperationAdd(os.path.relpath(f, self.outbox), f) for f in files]
        ops += self._pending_ops
        name = None
        if claim:
            name = f"claims/{claim[0]}/{shard_name(claim[1])}__{int(time.time())}__{self.who}"
            ops.append(CommitOperationAdd(name, self.who.encode()))
        if not ops:
            return True
        try:
            self.api.create_commit(self.repo, repo_type="dataset", operations=ops,
                                   commit_message=f"{self.who}: {len(files)} files" + (f", claim {claim[0]} {claim[1]}" if claim else ""))
        except Exception as e:
            print(f"commit failed ({type(e).__name__}: {str(e)[:120]}); outbox kept", flush=True)
            return False
        for f in files:
            os.remove(f)
        self._pending_ops = []
        self._cache.clear()
        return True

    # claims ------------------------------------------------------------------------------------------------
    def claims(self, stage):
        """{k: (time, who)} keeping the oldest live claim per shard (that one wins a race)."""
        out = {}
        for n in self._list(f"claims/{stage}", refresh=True):
            try:
                s, t, who = n.split("__", 2); k, t = int(s[6:]), float(t)
            except ValueError:
                continue
            if k not in out or t < out[k][0]:
                out[k] = (t, who)
        return out

    def claim(self, stage, k, ttl_h):
        c = self.claims(stage).get(k)
        if c and time.time() - c[0] < ttl_h * 3600 and c[1] != self.who:
            return False
        if not self.flush(claim=(stage, k)):
            return False
        c = self.claims(stage).get(k)                               # race check: oldest live claim wins
        live = c and time.time() - c[0] < ttl_h * 3600
        if live and c[1] != self.who:
            self.release(stage, k)
            return False
        return True

    def release(self, stage, k):
        """Delete our claim files for (stage, k) in the next commit."""
        from huggingface_hub import CommitOperationDelete
        for n in self._list(f"claims/{stage}"):
            if n.startswith(shard_name(k) + "__") and n.endswith("__" + self.who):
                self._pending_ops.append(CommitOperationDelete(f"claims/{stage}/{n}"))

    def touch(self, stage, k):
        pass                                                        # TTL covers the shard; no heartbeat commits


def board(corpus, site=None, outbox_root: str | None = None) -> Store:
    """The shared store for a corpus: its hub repo, or a disk dir when the corpus has none."""
    layout = corpus.layout or {}
    if corpus.hub_repo:
        root = outbox_root or os.path.join(site.work if site else ".", "outbox", corpus.name)
        os.makedirs(root, exist_ok=True)
        return HubStore(corpus.hub_repo, root, layout)
    return DiskStore(os.path.join(site.work if site else ".", "board", corpus.name), layout)


def private_store(corpus, site) -> DiskStore:
    return DiskStore(os.path.join(site.private, corpus.name), corpus.layout or {})
