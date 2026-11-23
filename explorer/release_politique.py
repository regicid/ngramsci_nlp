"""Build the propp-press release files of politique or presse3 (characters + portrayal, entities), in two steps (cca htc):
  [RELEASE_CORPUS=presse3] python explorer/release_politique.py fetch   -> <W>/cache/release[_presse3]/ (one job)
  [RELEASE_CORPUS=presse3] python explorer/release_politique.py I N     -> <W>/release[_presse3]/data/{characters,entities}_qwen4b/<corpus>/
                                                                         (task I of an array of N, after the fetch: afterok)
Reads the board (characters/, entities/, mentions/, portrayal/ + portrayal_surnames/ or portrayal_all/ where they exist)
and the private text (dates, and the text the names are recomputed from for name_source).
The fetch is the only step that talks to the hub: one listing, then one file at a time, throttled and waiting out HTTP 429
(HF allows 1000 API requests / 5 min per account and each download is 2-3 of them: 32 build tasks downloading for
themselves hit it within minutes, 2026-11-23). The build tasks read the cache only (no hub call), so any N is safe."""
import functools, gzip, json, os, sys, time
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ngramsci import release as Rl
from ngramsci.stages.portrayal import Surnames, load_inputs
SN = Surnames.from_json(os.path.join(os.path.dirname(__file__), "..", "data", "surnames.json"))
CORPUS = os.environ.get("RELEASE_CORPUS", "politique")
N_SHARDS = {"politique": 311, "presse3": 642}[CORPUS]
R = f"regicid/qcoref-{CORPUS}-work"; W = os.environ.get("NGRAMSCI_WORK", "/sps/humanum/home/bdecourson/ngramsci_work")
SUF = "" if CORPUS == "politique" else f"_{CORPUS}"
TXT, OUT, CACHE = f"{W}/private/{CORPUS}/text", f"{W}/release{SUF}", f"{W}/cache/release{SUF}"

DIRS = ("entities", "mentions", "characters", "portrayal", "portrayal_surnames", "portrayal_all")
MANIFEST = f"{CACHE}/manifest.json"
FETCH_INTERVAL = 1.0                       # s between downloads: ~3 API requests/s at most, under the hub's 1000 / 5 min


def fetch():
    """Everything the build reads, into CACHE (resumable: files already there are skipped, without any hub call)."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    os.makedirs(CACHE, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(R, repo_type="dataset") if f.split("/")[0] in DIRS)
    json.dump(files, open(MANIFEST, "w"))
    todo = [f for f in files if not os.path.exists(f"{CACHE}/{f}")]
    print(f"{len(files)} board files, {len(todo)} to fetch", flush=True)
    for i, f in enumerate(todo):
        for wait in [60, 120, 300, 300, 300, 300, 300, 600, 600, 600, None]:
            try:
                hf_hub_download(R, f, repo_type="dataset", local_dir=CACHE); break
            except HfHubHTTPError as e:
                if wait is None or getattr(e.response, "status_code", None) not in (429, 500, 502, 503, 504):
                    raise
                ra = (e.response.headers or {}).get("Retry-After")
                print(f"{f}: HTTP {e.response.status_code}, waiting {ra or wait}s", flush=True)
                time.sleep(int(ra) if ra and ra.isdigit() else wait)
        time.sleep(FETCH_INTERVAL)
        if i % 200 == 0:
            print(f"{i}/{len(todo)}", flush=True)
    missing = [f for f in files if not os.path.exists(f"{CACHE}/{f}")]
    if missing:
        raise SystemExit(f"{len(missing)} files still missing, e.g. {missing[:3]}")
    print("FETCH DONE", flush=True)


@functools.lru_cache(1)
def on_board():
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"no {MANIFEST}: run `release_politique.py fetch` first")
    return set(json.load(open(MANIFEST)))


def get(p):
    """Local path of a board file, None if the board has no such file (portrayal_all/ before score_all runs...)."""
    if p not in on_board():
        return None
    f = f"{CACHE}/{p}"
    if not os.path.exists(f):
        raise SystemExit(f"{p} is on the board but not in {CACHE}: re-run the fetch")
    return f

def one(k):
    txt, fe_in = f"{TXT}/shard_{k:05d}.jsonl.gz", get(f"entities/shard_{k:05d}.parquet")
    docs, ents = load_inputs(txt, get(f"mentions/shard_{k:05d}.ner.parquet"), fe_in, exclude_possessive=True)
    ds = docs["date"]
    c, e = pd.read_parquet(get(f"characters/shard_{k:05d}.parquet")), pd.read_parquet(fe_in)
    allp = get(f"portrayal_all/shard_{k:05d}.parquet")           # score_all run: every singular chain, alone
    ps = [pd.read_parquet(allp)] if allp else \
        [pd.read_parquet(f) for f in (get(f"portrayal/shard_{k:05d}.parquet"), get(f"portrayal_surnames/shard_{k:05d}.parquet")) if f]
    p = pd.concat(ps, ignore_index=True)
    p = Rl.locate(p, docs, ents, mode="chain" if allp else "name")          # which mentions each passage scores
    # scores per mention in entities (presse3 on); politique keeps the 2026-11-22 layout (passage lists) until re-released
    per_mention = CORPUS != "politique" or os.environ.get("PER_MENTION") == "1"
    co = Rl.characters(c, p, ds, sources=Rl.name_sources(docs, ents, SN), passages=not per_mention)
    eo = Rl.entities(e, ds, port=p if per_mention else None)
    fc, fe = f"{OUT}/data/characters_qwen4b/{CORPUS}/part-{k:05d}.parquet", f"{OUT}/data/entities_qwen4b/{CORPUS}/part-{k:05d}.parquet"
    co.to_parquet(fc, index=False); eo.to_parquet(fe, index=False)
    rc, re_ = pd.read_parquet(fc, columns=["portrayal_n", "date"]), pd.read_parquet(fe, columns=["date"])
    ok = len(rc) == len(c) and len(re_) == len(e) and int(rc.portrayal_n.sum()) == len(p) and rc.date.notna().all() and re_.date.notna().all()
    located = int(p["sent"].notna().sum())
    return [k, bool(ok), len(c), int((rc.portrayal_n > 0).sum()), len(p), len(e), len(ps), located]

if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(); sys.exit()
    i, n = int(sys.argv[1]), int(sys.argv[2])
    for d in ("characters_qwen4b", "entities_qwen4b"):
        os.makedirs(f"{OUT}/data/{d}/{CORPUS}", exist_ok=True)
    res = []
    for k in range(i, N_SHARDS, n):
        res.append(one(k)); print(res[-1], flush=True)
    json.dump(res, open(f"{OUT}/slice_{i:03d}.json", "w")); print("DONE", flush=True)
