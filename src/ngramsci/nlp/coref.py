"""From Qwen window outputs to document-level coreference chains.

Each window gives local cluster ids for its PER mentions. Windows are linked greedily left to right: a local cluster
joins the global cluster most of its already-seen mentions belong to (windows overlap by one segment), otherwise it
starts a new one; no transitive chaining. Then clusters sharing an identical proper-name mention are merged.
PER mentions never seen by a valid window are singletons."""
from collections import Counter, defaultdict


def coref_ids(per_mids, wins, outs, text_of, prop_of, extract):
    """per_mids: PER mention ids in text order. wins: window dicts {w, input, mention_ids}; outs: {w: qwen output};
    extract(input, output) -> local ids. Returns ({mid: cluster_id renumbered from 0}, n_bad_windows)."""
    gid, n_global, bad = {}, 0, 0
    for w in sorted(wins, key=lambda w: w["w"]):
        o = outs.get(w["w"])
        if o is None:
            bad += 1; continue
        local = extract(w["input"], o)
        if len(local) != len(w["mention_ids"]):
            bad += 1; continue
        by_local = defaultdict(list)
        for mid, lid in zip(w["mention_ids"], local):
            by_local[lid].append(mid)
        for members in by_local.values():
            votes = Counter(gid[m] for m in members if m in gid)
            target = votes.most_common(1)[0][0] if votes else n_global
            if not votes:
                n_global += 1
            for m in members:
                gid.setdefault(m, target)
    for m in per_mids:
        if m not in gid:
            gid[m] = n_global; n_global += 1
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    first = {}
    for m in per_mids:                                   # proper-name merge
        if prop_of[m] == "PROP":
            t, c = text_of[m], find(gid[m])
            if t in first and find(first[t]) != c:
                parent[c] = find(first[t])
            else:
                first.setdefault(t, c)
    renum, res = {}, {}
    for m in per_mids:
        res[m] = renum.setdefault(find(gid[m]), len(renum))
    return res, bad
