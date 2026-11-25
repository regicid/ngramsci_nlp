"""Name counts for the family-name table (data/surnames.json), on cca: full names AND bare names (« Lula », « Madonna »,
« Mélenchon ») per article and year, from the coref outputs and the private text (portrayal.name_rows).
  python explorer/name_counts.py count CORPUS I N   task I of N over the shards of CORPUS (lemonde | politique | presse3)
                                                    -> <W>/name_counts/CORPUS/part_I.parquet
  python explorer/name_counts.py build              every part -> data/surnames.json (+ data/aliases.json applied)
lemonde reads the board symlinks (lemonde_setup.py links); politique / presse3 read the release caches
(release_politique.py fetch), so no hub call here."""
import glob, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ngramsci.stages.portrayal import Surnames, load_aliases, load_inputs, name_counts, name_rows

W = os.environ.get("NGRAMSCI_WORK", "/sps/humanum/home/bdecourson/ngramsci_work")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def inputs(corpus):
    """[(text, mentions, entities)] paths per shard."""
    if corpus == "lemonde":
        B, P = f"{W}/board/lemonde", f"{W}/private/lemonde"
        ks = pd.read_csv(f"{B}/mapping.tsv", sep="\t").k
        return [(f"{P}/text/shard_{k:05d}.parquet", f"{B}/mentions/shard_{k:05d}.ner.parquet",
                 f"{B}/entities/shard_{k:05d}.parquet") for k in ks]
    C = f"{W}/cache/release" + ("" if corpus == "politique" else f"_{corpus}")
    out = []
    for e in sorted(glob.glob(f"{C}/entities/shard_*.parquet")):
        k = e.rsplit("_", 1)[1].split(".")[0]
        t = f"{W}/private/{corpus}/text/shard_{k}.jsonl.gz"
        if os.path.exists(t) and os.path.exists(m := f"{C}/mentions/shard_{k}.ner.parquet"):
            out.append((t, m, e))
    return out


def count(corpus, i, n):
    """Task i of n. Parts of an earlier run with more tasks (part_K, K >= n) are moved to name_counts_stale/ by task 0:
    the build reads every part (2026-11-25: 20 leftover parts of a 40-task run counted half of Le Monde twice)."""
    if i == 0:
        stale = f"{W}/name_counts_stale/{corpus}"
        for f in glob.glob(f"{W}/name_counts/{corpus}/part_*.parquet"):
            if int(os.path.basename(f)[5:8]) >= n:
                os.makedirs(stale, exist_ok=True); os.replace(f, f"{stale}/{os.path.basename(f)}")
    al = load_aliases(f"{DATA}/aliases.json")
    compounds = Surnames.from_json(f"{DATA}/surnames.json").compounds
    rows = []
    for j, (t, m, e) in enumerate(inputs(corpus)):
        if j % n != i:
            continue
        docs, ents = load_inputs(t, m, e, exclude_possessive=True)
        for d, g in ents.groupby("doc_id", sort=False):
            if d not in docs.index:
                continue
            y = str(docs.at[d, "date"])[:4]
            if y.isdigit():
                rows += [(d, al.get(_k(nm), nm) if kind == "full" else nm, int(y), kind)
                         for nm, kind in name_rows(docs.at[d, "text"], g, compounds)]
        print(j, len(rows), flush=True)
    os.makedirs(f"{W}/name_counts/{corpus}", exist_ok=True)
    name_counts(pd.DataFrame(rows, columns=["doc_id", "full_name", "year", "kind"])).to_parquet(
        f"{W}/name_counts/{corpus}/part_{i:03d}.parquet")
    print("DONE", flush=True)


def _k(s):
    from ngramsci.explorer import name_key
    return name_key(s)


def build():
    c = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{W}/name_counts/*/part_*.parquet"))])
    c = c.groupby(["year", "full_name", "kind"])["n"].sum().reset_index()
    sn = Surnames.build(c, aliases=load_aliases(f"{DATA}/aliases.json"))
    sn.to_json(f"{DATA}/surnames.json")
    print(len(sn.t), "names resolve in some year;", len(sn.compounds), "compounds;", len(sn.firstnames), "first names")


if __name__ == "__main__":
    if sys.argv[1] == "count":
        count(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        build()
