"""Core machinery without models: stores, claims, the worker loop, fused mode, launch rendering."""
import gzip
import json
import os
import time
from types import SimpleNamespace

import pytest

from ngramsci import stage as S
from ngramsci.config import Corpus, Partition, Site
from ngramsci.store import DiskStore, HubStore, pattern_regex
from ngramsci.worker import Worker, run_local


def test_pattern_regex():
    rx = pattern_regex("mentions/shard_{k:05d}.ner.parquet")
    assert rx.match("shard_00042.ner.parquet").group(1) == "00042"
    assert not rx.match("shard_00042.ner.parquet.tmp")
    assert not rx.match(".shard_00042.ner.parquet")


# two toy stages: "upper" (text -> private + public outputs), "count" (public input -> public output)
def _upper(k, inp, out, state, ctx):
    rows = [json.loads(l) for l in gzip.open(inp["text"], "rt")]
    with gzip.open(out["up_private"], "wt") as f:
        f.write("\n".join(r["text"].upper() for r in rows))
    json.dump({"n": len(rows)}, open(out["up_public"], "w"))
    return {"n": len(rows)}


def _count(k, inp, out, state, ctx):
    n = json.load(open(inp["up_public"]))["n"]
    json.dump({"n2": n * 2}, open(out["count"], "w"))
    return {}


S.register(S.Stage("t_upper", inputs=["text"], env="x", gpu=False, run=_upper,
                   outputs=[S.Output("up_private", "up/shard_{k:05d}.txt.gz", private=True), S.Output("up_public", "up_pub/shard_{k:05d}.json")],
                   done="up_public"))
S.register(S.Stage("t_count", inputs=["up_public"], env="x", gpu=False, run=_count,
                   outputs=[S.Output("count", "count/shard_{k:05d}.json")]))


def _text(root, k, n=3):
    p = os.path.join(root, "text"); os.makedirs(p, exist_ok=True)
    with gzip.open(os.path.join(p, f"shard_{k:05d}.jsonl.gz"), "wt") as f:
        for i in range(n):
            f.write(json.dumps({"doc_id": f"d{k}_{i}", "text": f"Bonjour {i}"}) + "\n")


def test_disk_claims(tmp_path):
    a, b = DiskStore(str(tmp_path)), DiskStore(str(tmp_path))
    b.who = "other"
    assert a.claim("st", 3, ttl_h=1)
    assert not b.claim("st", 3, ttl_h=1)          # live claim
    os.utime(a._lock("st", 3), (time.time() - 7200, time.time() - 7200))
    assert b.claim("st", 3, ttl_h=1)              # stale -> taken over
    a.release("st", 3)                            # not ours any more: kept
    assert a.claims("st")[3][1] == "other"
    b.release("st", 3)
    assert 3 not in a.claims("st")


def test_partial_work_of_a_running_worker_is_not_taken(tmp_path):
    """Two workers of one site: a non-empty scratch dir under a live claim belongs to the worker running it."""
    corpus = Corpus(name="toy", n_shards=2)
    board, priv = DiskStore(str(tmp_path / "board")), DiskStore(str(tmp_path / "priv"))
    for k in range(2):
        _text(str(tmp_path / "priv"), k)
    site = Site(name="A", ssh=None, code=".", work=str(tmp_path / "w"))
    a = Worker(corpus, site, ["t_upper"], board=board, private=priv, poll=0)
    b = Worker(corpus, site, ["t_upper"], board=board, private=priv, poll=0)
    board.who = "a"
    st, k, _ = a.pick()
    assert k == 0
    os.makedirs(os.path.join(a.scratch(st, 0), "out"))       # a is running shard 0
    board.who = "b"
    assert b.pick()[1] == 1                                   # b leaves it alone
    board.release(st.name, 1)
    board.who = "a"; board.release(st.name, 0)                # a stopped at its deadline: claim released
    board.who = "b"
    assert b.pick()[1] == 0                                   # now b resumes the partial work first


def test_worker_two_sites(tmp_path):
    """Site A holds the text; the board is a shared disk. A runs t_upper; B (no text) can only run t_count."""
    corpus = Corpus(name="toy", n_shards=4)
    board = DiskStore(str(tmp_path / "board"))
    pa, pb = DiskStore(str(tmp_path / "privA")), DiskStore(str(tmp_path / "privB"))
    for k in range(4):
        _text(str(tmp_path / "privA"), k)
    site = lambda n: Site(name=n, ssh=None, code=".", work=str(tmp_path / n))
    wb = Worker(corpus, site("B"), ["t_upper", "t_count"], board=board, private=pb, poll=0, max_shards=10)
    st, k, waiting = wb.pick()
    assert st is None and waiting                   # B has no text: nothing to do yet
    wa = Worker(corpus, site("A"), ["t_upper"], board=board, private=pa, poll=0, start=2)
    wa.loop()
    assert board.done(S.get("t_upper").done_output) == {0, 1, 2, 3}
    assert pa.done(S.get("t_upper").out("up_private")) == {0, 1, 2, 3}   # private outputs stayed on A
    assert not os.path.exists(tmp_path / "board" / "up")
    wb.loop()
    assert board.done(S.get("t_count").done_output) == {0, 1, 2, 3}
    assert json.load(open(tmp_path / "board" / "count" / "shard_00001.json")) == {"n2": 6}
    assert not board.claims("t_count")             # claims released


def test_run_local(tmp_path):
    _text(str(tmp_path), 0)
    st = run_local(Corpus(name="toy", n_shards=1), ["t_upper", "t_count"], str(tmp_path))
    assert st.done(S.get("t_count").done_output) == {0}


class FakeApi:
    """In-memory stand-in for HfApi: list_repo_tree + create_commit."""
    def __init__(self):
        self.files = {}; self.commits = 0; self.fail = 0         # fail: next n commits raise (HTTP 429)

    def list_repo_tree(self, repo, repo_type=None, path_in_repo=""):
        p = path_in_repo.rstrip("/") + "/"
        return [SimpleNamespace(path=f) for f in self.files if f.startswith(p) and "/" not in f[len(p):]]

    def create_commit(self, repo, repo_type=None, operations=(), commit_message=""):
        if self.fail:
            self.fail -= 1; raise RuntimeError("429 Client Error: Too Many Requests")
        self.commits += 1
        for op in operations:
            if type(op).__name__ == "CommitOperationDelete":
                self.files.pop(op.path_in_repo, None)
            else:
                self.files[op.path_in_repo] = op.path_or_fileobj


def test_hub_claim_race_and_batching(tmp_path):
    api = FakeApi()
    a = HubStore("r", str(tmp_path / "ob"), api=api); a.who = "a"; a.LIST_EVERY = 0
    b = HubStore("r", str(tmp_path / "ob2"), api=api); b.who = "b"; b.LIST_EVERY = 0
    assert a.claim("qwen", 5, 6)
    assert not b.claim("qwen", 5, 6)
    # a race: b's claim lands but a's older one wins
    api.files[f"claims/qwen/shard_00007__{int(time.time()) - 10}__a"] = b"a"
    assert not b.claim("qwen", 7, 6)
    assert all(not f.endswith("__b") for f in api.files)          # b withdrew its claim
    # publishing shard 5 + claiming 6 = one commit
    o = S.Output("q", "qwen_k10/shard_{k:05d}.ner.jsonl.gz")
    src = tmp_path / "x.gz"; src.write_bytes(b"x")
    n0 = api.commits
    a.publish("qwen", 5, [(o, str(src))], o)
    assert 5 in a.done(o)                                             # waiting in the outbox counts as done
    assert a.claim("qwen", 6, 6)
    assert api.commits == n0 + 1
    assert "qwen_k10/shard_00005.ner.jsonl.gz" in api.files
    assert not any(f.startswith("claims/qwen/shard_00005") for f in api.files)


def test_hub_batch_claims_timed_flush_and_backoff(tmp_path):
    api = FakeApi()
    a = HubStore("r", str(tmp_path / "ob"), api=api); a.who = "a"; a.LIST_EVERY = 0
    b = HubStore("r", str(tmp_path / "ob2"), api=api); b.who = "b"; b.LIST_EVERY = 0
    assert a.claim_many("p", [1, 2, 3], 6) == [1, 2, 3] and api.commits == 1        # one commit, three claims
    assert b.claim_many("p", [3, 4], 6) == [4]                                        # 3 is a's
    o = S.Output("q", "p/shard_{k:05d}.parquet")
    src = tmp_path / "x"; src.write_bytes(b"x")
    a.publish("p", 1, [(o, str(src))], o)
    n0 = api.commits
    assert not a.flush(force=False) and api.commits == n0                           # unforced: too soon
    a._last -= a.MIN_INTERVAL + 1
    assert a.flush(force=False) and "p/shard_00001.parquet" in api.files
    assert not any(f.startswith("claims/p/shard_00001") for f in api.files)          # released with its output
    api.fail = 1
    assert a.claim_many("p", [5], 6) is None and a.wait_s() > 50
    assert not a.flush() and api.commits == n0 + 1                                   # backing off: no call at all
    a._next_ok = 0
    assert a.claim_many("p", [5], 6) == [5]


def test_worker_batches_on_a_rate_limited_hub(tmp_path):
    """Workers on a hub board: shards are claimed in batches, a 429 makes them wait instead of hammering."""
    api = FakeApi()
    corpus = Corpus(name="toy", n_shards=6)
    pa = DiskStore(str(tmp_path / "privA"))
    for k in range(6):
        _text(str(tmp_path / "privA"), k)
    hub = HubStore("r", str(tmp_path / "ob"), api=api); hub.LIST_EVERY = 0
    w = Worker(corpus, Site(name="A", ssh=None, code=".", work=str(tmp_path / "A")), ["t_upper"], board=hub, private=pa,
               poll=0)
    w.secs["t_upper"] = [1.0]; w.batch_s = 3.0                                     # batches of 3 shards
    api.fail = 1; hub.MAX_BACKOFF = 0
    w.loop()
    assert hub.done(S.get("t_upper").done_output, refresh=True) == {0, 1, 2, 3, 4, 5}
    assert api.commits == 3                      # claim 0-2, claim 3-5 (+ outputs 0-2), final push of 3-5
    assert not hub.claims("t_upper")


def test_adopting_orphans_once(tmp_path, monkeypatch):
    """A dead worker's outbox goes to exactly one new worker, and the adopter's outbox does not look idle."""
    import ngramsci.store as store
    names = iter(["a-1", "b-2"]); monkeypatch.setattr(store, "who_am_i", lambda: next(names))
    root = tmp_path / "ob"; dead = root / "dead-1" / "p"; dead.mkdir(parents=True)
    f = dead / "shard_00001.parquet"; f.write_bytes(b"x"); os.utime(f, (0, 0))
    a = HubStore("r", str(root), api=FakeApi())
    assert os.path.exists(os.path.join(a.outbox, "p", "shard_00001.parquet")) and not (root / "dead-1").exists()
    b = HubStore("r", str(root), api=FakeApi())
    assert not os.listdir(b.outbox) and os.path.exists(os.path.join(a.outbox, "p", "shard_00001.parquet"))


def test_launch_render():
    from ngramsci.launch import render
    site = Site(name="c", ssh="c", code="/code", work="/w", envs={"vllm": {"squashfs": "/e.sqfs", "python_squashfs": "/p.sqfs"}},
                pre="export HF_HOME=/hf")
    part = Partition(name="h100", sbatch=["--partition=gpu_h100", "--gres=gpu:1"], workers=2, max_hours=70)
    s = render(site, part, "/w/launch/politique.yaml", "/w/launch/sites.yaml", ["qwen"], [(0, False), (150, True)], "ng-x")
    assert "#SBATCH --partition=gpu_h100" in s and "activate_sqfs.sh /e.sqfs /p.sqfs" in s
    assert s.count("python -m ngramsci.cli work") == 2 and "--start 150 --reverse" in s and "sleep 90" in s
    with pytest.raises(SystemExit):
        render(site, part, "c", "s", ["qwen", "parse_ner"], [(0, False)], "x")    # two env kinds in one job


def _two(k, inp, out, state, ctx):
    for n in ("a", "b"):
        open(out[n], "w").write(n)
    return {}


S.register(S.Stage("t_two", inputs=["text"], env="x", gpu=False, run=_two,
                   outputs=[S.Output("a", "a/shard_{k:05d}.txt"), S.Output("b", "b/shard_{k:05d}.txt")]))


def test_outputs_with_same_basename(tmp_path):
    _text(str(tmp_path), 0)
    run_local(Corpus(name="toy", n_shards=1), ["t_two"], str(tmp_path))
    assert open(tmp_path / "a" / "shard_00000.txt").read() == "a" and open(tmp_path / "b" / "shard_00000.txt").read() == "b"


def _priv(k, inp, out, state, ctx):
    open(out["p"], "w").write("x")
    return {}


S.register(S.Stage("t_priv", inputs=["text"], env="x", gpu=False, run=_priv,
                   outputs=[S.Output("p", "p/shard_{k:05d}.txt", private=True)]))


def test_local_stage_claims_stay_on_site(tmp_path):
    """A stage with only private inputs/outputs (parse) claims with disk locks: no hub commit at all."""
    api = FakeApi()
    hub = HubStore("r", str(tmp_path / "ob"), api=api); hub.LIST_EVERY = 0
    priv = DiskStore(str(tmp_path / "priv"))
    for k in range(3):
        _text(str(tmp_path / "priv"), k)
    assert S.get("t_priv").local and not S.get("t_upper").local
    w = Worker(Corpus(name="toy", n_shards=3), Site(name="A", ssh=None, code=".", work=str(tmp_path / "w")), ["t_priv"],
               board=hub, private=priv, poll=0)
    w.loop()
    assert priv.done(S.get("t_priv").done_output) == {0, 1, 2}
    assert api.commits == 0 and not priv.claims("t_priv")
