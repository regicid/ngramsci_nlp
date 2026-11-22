"""Portrayal explorer data: per-name scores by month and outlet, for a static search page.

cells(): one portrayal shard + its articles' dates -> sums per (name, month, outlet). Names are keyed loosely
(case, accents, hyphens): « Jean-Luc Mélenchon » and « Jean-luc Melenchon » are one key; homonyms stay merged.
Each segment counts with weight w = p_pertinent (long portraits weigh more). Score x of a segment: P(+) - P(-) for
valence, extremisme, dynamique (in [-1, 1]); p_journaliste for journaliste. Per article d: W_d = sum w, S_d = sum w x.
A cell keeps n_docs, n_seg, W = sum W_d, S = sum S_d, and SS = sum S_d^2, WS = sum W_d S_d, WW = sum W_d^2, so any
union of cells gives the weighted mean R = S / W and its standard error with articles as the resampling unit:
var(R) = (SS - 2 R WS + R^2 WW) / W^2 (ratio estimator, clustered by article).

site(): merged cells -> index.json (names for autocomplete) + data/b_XXX.json (cells of the names hashed there)."""
from __future__ import annotations

import json
import os
import re
import unicodedata
import zlib

import numpy as np
import pandas as pd

DIMS = ["valence", "extremisme", "dynamique", "journaliste"]
SUMS = ["S", "SS", "WS"]


def name_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def outlet(doc_id: pd.Series) -> pd.Series:
    return doc_id.str.rsplit("_", n=1).str[0]


def cells(port: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """port: portrayal rows of one shard; dates: doc_id -> 'YYYY-MM-DD'."""
    d = port[["doc_id", "full_name", "p_pertinent"]].copy()
    w = port["p_pertinent"].to_numpy()
    for k in DIMS[:3]:
        d[f"x_{k}"] = w * (port[f"p_{k}_pos"].to_numpy() - port[f"p_{k}_neg"].to_numpy())
    d["x_journaliste"] = w * port["p_journaliste"].to_numpy()
    d["key"] = d["full_name"].map(name_key)
    d = d[d["key"] != ""]
    g = d.groupby(["key", "doc_id"], sort=False)
    per = g[["p_pertinent"] + [f"x_{k}" for k in DIMS]].sum().rename(columns={"p_pertinent": "W"})
    per["n_seg"] = g.size()
    per = per.reset_index()
    per["month"] = per["doc_id"].map(dates).str[:7]
    per = per.dropna(subset=["month"])
    per["outlet"] = outlet(per["doc_id"])
    per["WW"] = per["W"] ** 2
    for k in DIMS:
        s = per.pop(f"x_{k}")
        per[f"S_{k}"], per[f"SS_{k}"], per[f"WS_{k}"] = s, s ** 2, per["W"] * s
    per["n_docs"] = 1
    out = per.drop(columns="doc_id").groupby(["key", "month", "outlet"], sort=False).sum().reset_index()
    names = d.groupby(["key", "full_name"]).size().rename("n").reset_index()   # spellings, for display
    return out, names


def merge(parts) -> tuple[pd.DataFrame, pd.DataFrame]:
    cs, ns = zip(*parts)
    c = pd.concat(cs).groupby(["key", "month", "outlet"], sort=False).sum().reset_index()
    n = pd.concat(ns).groupby(["key", "full_name"])["n"].sum().reset_index()
    return c, n


def bucket(key: str, n: int) -> int:
    return zlib.crc32(key.encode()) % n


def summarise(c: pd.DataFrame, by: list[str], dims=DIMS[:3]) -> pd.DataFrame:
    """Sums -> per-cell n_docs, W, WW and, per dim, the mean R and its article-clustered SE (rounded for the page).
    The page merges cells with R = sum W_i R_i / W and var = sum (W_i^2 se_i^2 + WW_i (R_i - R)^2) / W^2: exact but for
    one cross term, sum_i 2 (R_i - R)(WS_i - R_i WW_i), small when weights are even within a cell."""
    g = c.groupby(by, sort=True)[["n_docs", "W", "WW"] + [f"{s}_{k}" for k in dims for s in SUMS]].sum().reset_index()
    out = g[by + ["n_docs"]].copy()
    out["W"], out["WW"] = g["W"].round(2), g["WW"].round(2)
    for k in dims:
        R = g[f"S_{k}"] / g["W"]
        var = (g[f"SS_{k}"] - 2 * R * g[f"WS_{k}"] + R ** 2 * g["WW"]).clip(lower=0) / g["W"] ** 2
        out[k], out[f"se_{k}"] = R.round(4), np.sqrt(var).round(4)
    return out


def site(c: pd.DataFrame, names: pd.DataFrame, out: str, min_docs=20, monthly_docs=300, n_buckets=250, top_spellings=4):
    """index.json: {dims, outlets, n_buckets, names: [[key, display, n_docs, n_seg, other spellings], ...]} by n_docs.
    data/b_XXX.json: {key: {"y": columns of (year, outlet) cells, "m": columns of monthly cells (big names only)}}."""
    dims = DIMS[:3]
    tot = c.groupby("key")[["n_docs", "n_seg"]].sum()
    keep = tot[tot["n_docs"] >= min_docs].sort_values("n_docs", ascending=False)
    names = names[names["key"].isin(keep.index)].sort_values("n", ascending=False)
    sp = names.groupby("key")["full_name"].apply(lambda s: list(s[:top_spellings]))
    c = c[c["key"].isin(keep.index)].copy()
    outlets = sorted(c["outlet"].unique())
    c["year"] = c["month"].str[:4].astype(int)
    c["o"] = c["outlet"].map({o: i for i, o in enumerate(outlets)})
    y = summarise(c, ["key", "year", "o"])
    big = keep.index[keep["n_docs"] >= monthly_docs]
    m = summarise(c[c["key"].isin(big)], ["key", "month"])
    index = [[k, sp[k][0], int(r.n_docs), int(r.n_seg), sp[k][1:]] for k, r in keep.iterrows()]
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    json.dump({"n_buckets": n_buckets, "dims": dims, "outlets": outlets, "names": index},
              open(os.path.join(out, "index.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    cols = lambda t, first: {col: t[col].tolist() for col in first + ["n_docs", "W", "WW"] + [x for k in dims for x in (k, f"se_{k}")]}
    buckets: dict[int, dict] = {}
    for k, g in y.groupby("key", sort=False):
        buckets.setdefault(bucket(k, n_buckets), {})[k] = {"y": cols(g, ["year", "o"])}
    for k, g in m.groupby("key", sort=False):
        buckets[bucket(k, n_buckets)][k]["m"] = cols(g, ["month"])
    for b, rows in buckets.items():
        json.dump(rows, open(os.path.join(out, "data", f"b_{b:03d}.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    return len(index), len(y), len(m)


def merged(cells: dict, dim: str):
    """Python twin of the page's merge (tests): cells = columns with n_docs, W, WW, dim, se_dim."""
    W, WW = np.asarray(cells["W"]), np.asarray(cells["WW"])
    R_i, se = np.asarray(cells[dim]), np.asarray(cells[f"se_{dim}"])
    R = (W * R_i).sum() / W.sum()
    return R, np.sqrt((W ** 2 * se ** 2 + WW * (R_i - R) ** 2).sum()) / W.sum()


def stats(rows: dict, dim: str):
    """Exact weighted mean and SE from summed cells."""
    W, S, SS, WS, WW = (np.sum(rows[k]) for k in ["W", f"S_{dim}", f"SS_{dim}", f"WS_{dim}", "WW"])
    R = S / W
    return R, np.sqrt(max(SS - 2 * R * WS + R * R * WW, 0)) / W
