# Shared read-only access to GDGB runs + reference caches, and the temporal
# metric primitives used by tasks A-D. The data root is never written to.
# The metric primitives are also exposed, data-free, as the `r2o` package.
# Conventions copied from the archived reviewer-response temporal_fidelity.py
# (evaluator-matched reference slots; seed rows excluded; RUN_OVERRIDES for the
# two contaminated weibo L2 cells) and from the G0.6 lineage (dedup/jitter arms).
import os
import numpy as np
import pandas as pd

REPO = os.environ.get("R2O_GDGB_ROOT")  # root containing GDGB-ALGO/ and GDGB_dataset/ (see scoring/README.md)
if not REPO:
    raise SystemExit("set R2O_GDGB_ROOT to the directory that contains GDGB-ALGO/ and GDGB_dataset/")
CONFIGS = os.path.join(REPO, "GDGB-ALGO/TDGG/LLMGraph/tasks/general/configs")
CACHE = os.path.join(REPO, "GDGB_dataset/cache")

INFO_DS = ["dianping", "wikirevision", "weibo_daily", "weibo_tech"]
LEVELS = [("L0", "re_qwen3"), ("L1", "react_v2"), ("L2", "memory_v1")]
# GPT-4o-Mini backbone cells that actually have completed runs (verified).
BACKBONE_CELLS = {
    "dianping_l0_gpt4omini_api": ("dianping", "L0-api"),
    "wikirevision_l0_gpt4omini_api": ("wikirevision", "L0-api"),
    "wikirevision_l1_gpt4omini_api": ("wikirevision", "L1-api"),
    "weibo_tech_l0_gpt4omini_api": ("weibo_tech", "L0-api"),
}
# From the archive script: latest-run is contaminated/dangling for these cells.
RUN_OVERRIDES = {
    "weibo_tech_memory_v1": "offline-run-20260724_113841-3azdhedc",
    "weibo_daily_memory_v1": "offline-run-20260731_112556-pgm0hq8v",
}
CELL_FLAGS = {
    "weibo_tech_react_v2": "SHORT_RUN_8000",
    "weibo_tech_memory_v1": "SHORT_RUN_8650_pre_contamination",
    "weibo_daily_memory_v1": "SHORT_RUN_8500_newer_latest_CONTAMINATED",
}


def cells_for(ds):
    out = [("%s_%s" % (ds, sfx), lvl) for lvl, sfx in LEVELS]
    for cfg, (d, lvl) in BACKBONE_CELLS.items():
        if d == ds:
            out.append((cfg, lvl))
    return out


def load_ref(ds, usecols=("actor_id", "timestamp", "original_edge_idx")):
    p = os.path.join(CACHE, "%s_edge.csv" % ds)
    df = pd.read_csv(p, usecols=list(usecols), low_memory=False)
    df["original_edge_idx"] = pd.to_numeric(df["original_edge_idx"], errors="coerce")
    df = df.dropna(subset=["original_edge_idx"])
    df["original_edge_idx"] = df["original_edge_idx"].astype(np.int64)
    df["actor_id"] = df["actor_id"].astype(str)
    return df.sort_values("original_edge_idx").reset_index(drop=True)


def resolve_run(cfg):
    wandb = os.path.join(CONFIGS, cfg, "generated_data/wandb")
    if cfg in RUN_OVERRIDES:
        p = os.path.join(wandb, RUN_OVERRIDES[cfg], "files/media/edges.csv")
        return (p, RUN_OVERRIDES[cfg]) if os.path.exists(p) else (None, "override_missing")
    p = os.path.join(wandb, "latest-run/files/media/edges.csv")
    if not os.path.exists(p):
        return None, "missing_latest_run"
    return p, os.path.basename(os.path.realpath(os.path.join(wandb, "latest-run")))


def load_gen(cfg):
    """Non-seed generated rows, or (None, reason)."""
    path, run = resolve_run(cfg)
    if path is None:
        return None, run
    g_all = pd.read_csv(path, index_col=0, low_memory=False)
    seed = g_all["seed"].astype(str).str.lower().isin(["true", "1"])
    g = g_all[~seed].copy()
    g["actor_id"] = g["actor_id"].astype(str)
    return g, run


def match_ref(ref, gen):
    idx = pd.to_numeric(gen["original_edge_idx"], errors="coerce").dropna().astype(np.int64)
    return ref[ref["original_edge_idx"].isin(set(idx))]


def secs(series):
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    s = dt.astype("int64") / 1e9
    return s.where(dt.notna()).to_numpy(dtype=float)


def iets_by_actor(t, a):
    """dict actor -> np.diff(sorted finite times), actors with >=2 events."""
    df = pd.DataFrame({"t": t, "a": a})
    df = df[np.isfinite(df["t"])]
    out = {}
    for k, g in df.groupby("a", sort=False):
        s = np.sort(g["t"].to_numpy())
        if s.size > 1:
            out[k] = np.diff(s)
    return out


def pooled(by):
    return np.concatenate(list(by.values())) if by else np.array([])


def burstiness(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    mu, sd = x.mean(), x.std(ddof=1)
    return np.nan if mu + sd == 0 else (sd - mu) / (sd + mu)


def memory_coeff(by, min_pairs=10):
    """Goh-Barabasi memory coefficient: Pearson corr of consecutive IETs,
    pairs pooled across actors (pairs never cross actors)."""
    a, b = [], []
    for d in by.values():
        if d.size >= 2:
            a.append(d[:-1]); b.append(d[1:])
    if not a:
        return np.nan
    a, b = np.concatenate(a), np.concatenate(b)
    if a.size < min_pairs or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rho1_global(t, min_n=11):
    """Lag-1 autocorrelation of the global-stream IET sequence."""
    s = np.sort(np.asarray(t, float)[np.isfinite(t)])
    d = np.diff(s)
    if d.size < min_n or d[:-1].std() == 0 or d[1:].std() == 0:
        return np.nan
    return float(np.corrcoef(d[:-1], d[1:])[0, 1])


def jitter(t, seed=0):
    """G0.6 jitter null: spread ties uniformly within the round quantum
    (modal positive gap = median of positive gaps between unique sorted times)."""
    t = np.asarray(t, float)
    rng = np.random.default_rng(seed)
    allt = np.sort(np.unique(t[np.isfinite(t)]))
    gaps = np.diff(allt)
    q = np.median(gaps[gaps > 0]) if (gaps > 0).any() else 1.0
    out = t.copy()
    m = np.isfinite(t)
    out[m] = t[m] + rng.uniform(0, q, size=int(m.sum()))
    return out


def dedup(t, a):
    """G0.6 dedup arm: collapse duplicate (actor, timestamp) events."""
    m = pd.DataFrame({"t": t, "a": a}).dropna().drop_duplicates()
    return m["t"].to_numpy(), m["a"].to_numpy()


def rhythm_divergence(g_by, r_by, min_iet=5):
    """Per-actor P(dt|u) divergence: KS between gen and ref IETs of the SAME
    actor, over shared actors with >= min_iet IETs on both sides.
    Returns (median_ks, mean_ks, n_actors)."""
    from scipy.stats import ks_2samp
    shared = [k for k in g_by if k in r_by
              and g_by[k].size >= min_iet and r_by[k].size >= min_iet]
    if not shared:
        return np.nan, np.nan, 0
    ks = [ks_2samp(g_by[k], r_by[k]).statistic for k in shared]
    return float(np.median(ks)), float(np.mean(ks)), len(shared)
