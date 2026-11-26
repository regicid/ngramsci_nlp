"""Publish the API database (portraits.sqlite, built by build_monthly.py sqlite) to the public hub repo DB_REPO, so the
API machine fetches it from there (explorer/fetch_db.py) instead of through the laptop. On cca, after the sqlite step:
  python explorer/publish_db.py [path]        default <W>/explorer/monthly/portraits.sqlite
The db holds aggregates only (sums per name x month x newspaper x section, name totals, the precomputed cells_year /
totals_month): no article text. The check below refuses a db with any other table or a long text value, so a later
change to build_monthly.py cannot push passages to a public repo by accident.
Each publish is one commit: portraits.sqlite + version.json (build time, sha256, row counts, corpora, code commit).
Earlier versions stay in the repo history (fetch_db.py --revision <commit> to roll back)."""
import datetime, hashlib, json, os, sqlite3, subprocess, sys

DB_REPO = "regicid/portraits-db"
W = os.environ.get("NGRAMSCI_WORK", "/sps/humanum/home/bdecourson/ngramsci_work")
TABLES = {"names", "cells", "cells_year", "totals_month"}
MAX_TEXT = 200                       # longest text value allowed (names, keys, newspaper/section codes)
CORPORA = ["lemonde", "politique", "presse3", "presse4"]


def check(db):
    """Row counts per table; SystemExit if a table is unknown or a text column holds anything longer than MAX_TEXT."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tables = {t for (t,) in con.execute("select name from sqlite_master where type='table'")}
    if tables - TABLES:
        sys.exit(f"unexpected tables {sorted(tables - TABLES)}: not published (could hold text)")
    rows = {}
    for t in sorted(tables):
        rows[t] = con.execute(f"select count(*) from {t}").fetchone()[0]
        for _, col, typ, *_ in con.execute(f"pragma table_info({t})"):
            if typ.upper() in ("TEXT", ""):
                m = con.execute(f"select max(length({col})) from {t}").fetchone()[0] or 0
                if m > MAX_TEXT:
                    sys.exit(f"{t}.{col} has a {m}-character value: not published (could be article text)")
    con.close()
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(db):
    from huggingface_hub import HfApi
    rows = check(db)
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    version = {"built_at": datetime.datetime.fromtimestamp(os.path.getmtime(db), datetime.timezone.utc).isoformat(timespec="seconds"),
               "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "size": os.path.getsize(db), "sha256": sha256(db), "rows": rows, "corpora": CORPORA, "code": commit}
    vf = f"{os.path.dirname(os.path.abspath(db))}/version.json"
    json.dump(version, open(vf, "w"), indent=1)
    print(json.dumps(version), flush=True)
    a = HfApi()
    a.create_repo(DB_REPO, repo_type="dataset", private=False, exist_ok=True)
    from huggingface_hub import CommitOperationAdd
    ops = [CommitOperationAdd("portraits.sqlite", db), CommitOperationAdd("version.json", vf)]
    if "README.md" not in a.list_repo_files(DB_REPO, repo_type="dataset"):
        ops.append(CommitOperationAdd("README.md", os.path.join(os.path.dirname(os.path.abspath(__file__)), "portraits_db_README.md")))
    c = a.create_commit(DB_REPO, ops, repo_type="dataset",
                        commit_message=f"portraits.sqlite built {version['built_at']} ({rows.get('names')} names, code {commit})")
    print("PUBLISHED", c.commit_url, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{W}/explorer/monthly/portraits.sqlite")
