"""Build the propp-press release files of politique (characters + portrayal, entities), one task of an array.
usage (cca htc array): python explorer/release_politique.py I N   -> <W>/release/data/{characters,entities}_qwen4b/politique/
Reads the board (characters/, entities/, portrayal/ and, when present, portrayal_surnames/) and dates from the private text."""
import gzip, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from ngramsci import release as Rl
R = "regicid/qcoref-politique-work"; W = os.environ.get("NGRAMSCI_WORK", "/sps/humanum/home/bdecourson/ngramsci_work")
TXT, OUT, CACHE = f"{W}/private/politique/text", f"{W}/release", f"{W}/cache/release"

def get(p):
    try:
        return hf_hub_download(R, p, repo_type="dataset", local_dir=CACHE)
    except EntryNotFoundError:
        return None

def one(k):
    with gzip.open(f"{TXT}/shard_{k:05d}.jsonl.gz", "rt") as f:
        ds = pd.Series({(d := json.loads(l))["doc_id"]: d.get("date") for l in f})
    c, e = pd.read_parquet(get(f"characters/shard_{k:05d}.parquet")), pd.read_parquet(get(f"entities/shard_{k:05d}.parquet"))
    ps = [pd.read_parquet(p) for p in (get(f"portrayal/shard_{k:05d}.parquet"), get(f"portrayal_surnames/shard_{k:05d}.parquet")) if p]
    p = pd.concat(ps, ignore_index=True)
    co, eo = Rl.characters(c, p, ds), Rl.entities(e, ds)
    fc, fe = f"{OUT}/data/characters_qwen4b/politique/part-{k:05d}.parquet", f"{OUT}/data/entities_qwen4b/politique/part-{k:05d}.parquet"
    co.to_parquet(fc, index=False); eo.to_parquet(fe, index=False)
    rc, re_ = pd.read_parquet(fc, columns=["portrayal_n", "date"]), pd.read_parquet(fe, columns=["date"])
    ok = len(rc) == len(c) and len(re_) == len(e) and int(rc.portrayal_n.sum()) == len(p) and rc.date.notna().all() and re_.date.notna().all()
    return [k, bool(ok), len(c), int((rc.portrayal_n > 0).sum()), len(p), len(e), len(ps)]

if __name__ == "__main__":
    i, n = int(sys.argv[1]), int(sys.argv[2])
    for d in ("characters_qwen4b", "entities_qwen4b"):
        os.makedirs(f"{OUT}/data/{d}/politique", exist_ok=True)
    res = []
    for k in range(i, 305, n):
        res.append(one(k)); print(res[-1], flush=True)
        for f in os.listdir(CACHE) if False else []:
            pass
    json.dump(res, open(f"{OUT}/slice_{i:03d}.json", "w")); print("DONE", flush=True)
