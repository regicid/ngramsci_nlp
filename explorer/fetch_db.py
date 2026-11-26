"""On the API machine: fetch the latest portraits.sqlite from the public hub repo (explorer/publish_db.py puts it there) and
swap it in atomically.
  python fetch_db.py [--dest /opt/bazoulay/portraits.sqlite] [--revision COMMIT] [--force]
Does nothing when the local version.json (next to the db) already has the hub's sha256. Otherwise downloads into the
destination's folder, checks the sha256 and that sqlite opens it, then os.replace(): readers already connected keep
the old file until they reconnect, new connections get the new one. No token needed (public repo).
Needs huggingface_hub (pip install huggingface_hub)."""
import argparse, hashlib, json, os, shutil, sqlite3, sys

DB_REPO = "regicid/portraits-db"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    from huggingface_hub import hf_hub_download
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="/opt/bazoulay/portraits.sqlite")
    p.add_argument("--revision", default=None, help="a commit of the hub repo (roll back); default: latest")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    folder = os.path.dirname(os.path.abspath(a.dest))
    local_v = os.path.join(folder, "version.json")
    remote = json.load(open(hf_hub_download(DB_REPO, "version.json", repo_type="dataset", revision=a.revision)))
    if not a.force and os.path.exists(a.dest) and os.path.exists(local_v) and json.load(open(local_v)).get("sha256") == remote["sha256"]:
        print("up to date:", remote["built_at"]); return
    if shutil.disk_usage(folder).free < 1.2 * remote["size"]:
        sys.exit(f"not enough space in {folder} for {remote['size'] / 1e9:.1f} GB")
    tmp = os.path.join(folder, ".portraits_download")
    f = hf_hub_download(DB_REPO, "portraits.sqlite", repo_type="dataset", revision=a.revision, local_dir=tmp)
    if sha256(f) != remote["sha256"]:
        sys.exit("sha256 mismatch: download kept in " + tmp + ", db not replaced")
    con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
    n = con.execute("select count(*) from names").fetchone()[0]; con.close()
    if n != remote["rows"]["names"]:
        sys.exit(f"names has {n} rows, version.json says {remote['rows']['names']}: db not replaced")
    os.replace(f, a.dest)
    json.dump(remote, open(local_v, "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print("replaced", a.dest, "with the db built", remote["built_at"], f"({n} names)")


if __name__ == "__main__":
    main()
