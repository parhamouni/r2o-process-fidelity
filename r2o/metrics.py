"""R2O temporal-fidelity statistics (Rate / Rhythm / Order), data-free.

All functions operate on arrays of event times (seconds, float) and actor ids, or on the
dict actor -> inter-event-time array returned by `iets_by_actor`. They are the exact
functions used to produce the score tables in `scores/` (copied from scoring/gdgb_common.py
and scoring/taskB_count_conditional.py; behaviour is identical).

Axes (see the paper, Sec. 3):
  Rate    : log duration ratio log(T_g / T_r); per-actor event counts
  Rhythm  : pooled inter-event KS (`pooled` + scipy.stats.ks_2samp), W1 on log IETs,
            burstiness gap (`burstiness`), node rhythm (`rhythm_divergence`),
            rate-normalised shape KS (`shape_pool`), count-conditioned KS (`count_conditioned_ks`)
  Order   : memory coefficient (`memory_coeff`), lag-1 autocorrelation (`rho1_global`)
Measurement arms: `jitter` (spread timestamp ties within the round quantum), `dedup`.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

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


def shape_pool(by):
    out = []
    for d in by.values():
        if d.size >= 3 and d.mean() > 0:
            out.append(d / d.mean())
    return np.concatenate(out) if out else np.array([])


def count_conditioned_ks(g_by, r_by, reps=20):
    shared = [k for k in g_by if k in r_by and g_by[k].size >= 2 and r_by[k].size >= 2]
    if not shared:
        return np.nan, np.nan, 0
    vals = []
    for _ in range(reps):
        gs, rs = [], []
        for k in shared:
            n = min(g_by[k].size, r_by[k].size)
            gs.append(RNG.choice(g_by[k], n, replace=False))
            rs.append(RNG.choice(r_by[k], n, replace=False))
        vals.append(ks_2samp(np.concatenate(gs), np.concatenate(rs)).statistic)
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), len(shared)
