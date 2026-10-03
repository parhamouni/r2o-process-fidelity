#!/usr/bin/env python
# Task C -- PILOT real-vs-real band (prereg 2.2), v0.
# Deviation from prereg spec, declared: pairs are covariate-matched disjoint
# 10k-edge windows (the prereg's block bootstrap preserving actor identity is
# the confirmatory construction; this pilot answers feasibility + first Z's).
# Per dataset: slide non-overlapping 10k-edge windows over the full reference
# stream; covariates = duration, n_actors, edges_per_actor, median positive
# global IET (N3's matching variables, log-ratio distance). Select up to 60
# closest pairs (window reuse capped at 3). Metric panel per pair -> mu/sigma.
# Then Z-scores for every existing cell against the band.
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import gdgb_common as C

W = 10000
MAX_PAIRS = 60
USE_CAP = 3


def win_stats(t, a):
    f = np.isfinite(t)
    tt = t[f]
    if tt.size < 2:
        return None
    dup = 1.0 - pd.Series(tt).nunique() / tt.size
    d = np.diff(np.sort(tt))
    pos = d[d > 0]
    if pos.size == 0 or dup > 0.9:
        return None  # degenerate window
    return dict(duration=float(tt.max() - tt.min()),
                n_actors=int(pd.Series(a[f]).nunique()),
                edges_per_actor=float(tt.size / max(pd.Series(a[f]).nunique(), 1)),
                med_pos_iet=float(np.median(pos)), dup_frac=float(dup))


def cov_dist(x, y, keys=("duration", "n_actors", "edges_per_actor", "med_pos_iet")):
    return float(sum(abs(np.log(x[k] / y[k])) for k in keys))


def pair_metrics(tw_a, aw_a, tw_b, aw_b):
    by_a = C.iets_by_actor(tw_a, aw_a); by_b = C.iets_by_actor(tw_b, aw_b)
    pa, pb = C.pooled(by_a), C.pooled(by_b)
    out = {}
    if pa.size > 1 and pb.size > 1:
        out["ks"] = float(ks_2samp(pa, pb).statistic)
        out["w1"] = float(wasserstein_distance(pa, pb))
        out["abs_burst_gap"] = abs(C.burstiness(pa) - C.burstiness(pb))
    else:
        out["ks"] = out["w1"] = out["abs_burst_gap"] = np.nan
    ma, mb = C.memory_coeff(by_a), C.memory_coeff(by_b)
    out["abs_M_gap"] = abs(ma - mb)
    med_ks, _, nsh = C.rhythm_divergence(by_a, by_b, min_iet=5)
    out["rhythm_median_ks"] = med_ks; out["rhythm_n_shared"] = nsh
    return out


win_rows, pair_rows, band_rows, z_rows = [], [], [], []
for ds in C.INFO_DS:
    print("[ref] %s" % ds, flush=True)
    ref = C.load_ref(ds)
    t_all = C.secs(ref["timestamp"]); a_all = ref["actor_id"].to_numpy()
    n_win = len(ref) // W
    stats, keep = [], []
    for i in range(n_win):
        sl = slice(i * W, (i + 1) * W)
        s = win_stats(t_all[sl], a_all[sl])
        win_rows.append(dict(dataset=ds, window=i, degenerate=s is None,
                             **(s or {})))
        if s is not None:
            s["window"] = i
            stats.append(s); keep.append(i)
    print("  %d/%d usable windows" % (len(keep), n_win), flush=True)
    if len(keep) < 2:
        band_rows.append(dict(dataset=ds, n_windows=n_win, n_usable=len(keep),
                              n_pairs=0, feasible_30=False))
        continue
    # rank all pairs by covariate distance, greedy with reuse cap
    cand = []
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            cand.append((cov_dist(stats[i], stats[j]), stats[i]["window"], stats[j]["window"]))
    cand.sort()
    use = {}
    sel = []
    for d, wi, wj in cand:
        if use.get(wi, 0) >= USE_CAP or use.get(wj, 0) >= USE_CAP:
            continue
        sel.append((d, wi, wj))
        use[wi] = use.get(wi, 0) + 1; use[wj] = use.get(wj, 0) + 1
        if len(sel) >= MAX_PAIRS:
            break
    print("  %d pairs selected (best dist %.3f, worst %.3f)"
          % (len(sel), sel[0][0] if sel else np.nan, sel[-1][0] if sel else np.nan), flush=True)
    pm = []
    for d, wi, wj in sel:
        m = pair_metrics(t_all[wi * W:(wi + 1) * W], a_all[wi * W:(wi + 1) * W],
                         t_all[wj * W:(wj + 1) * W], a_all[wj * W:(wj + 1) * W])
        m.update(dataset=ds, win_a=wi, win_b=wj, cov_dist=d)
        pm.append(m); pair_rows.append(m)
    pmdf = pd.DataFrame(pm)
    band = dict(dataset=ds, n_windows=n_win, n_usable=len(keep),
                n_pairs=len(sel), feasible_30=len(sel) >= 30)
    for k in ("ks", "w1", "abs_burst_gap", "abs_M_gap", "rhythm_median_ks"):
        band["mu_" + k] = float(pmdf[k].mean())
        band["sigma_" + k] = float(pmdf[k].std(ddof=1))
    band_rows.append(band)

    # ---- Z-scores for existing cells against this band ----
    for cfg, level in C.cells_for(ds):
        gen, run = C.load_gen(cfg)
        if gen is None:
            continue
        ref_gen = C.match_ref(ref, gen)
        m = pair_metrics(C.secs(gen["timestamp"]), gen["actor_id"].to_numpy(),
                         C.secs(ref_gen["timestamp"]), ref_gen["actor_id"].to_numpy())
        zr = dict(config=cfg, dataset=ds, level=level)
        for k in ("ks", "w1", "abs_burst_gap", "abs_M_gap", "rhythm_median_ks"):
            zr["D_" + k] = m[k]
            sig = band["sigma_" + k]
            zr["Z_" + k] = ((m[k] - band["mu_" + k]) / sig
                            if sig and np.isfinite(sig) and sig > 0 else np.nan)
        z_rows.append(zr)
        print("  [z] %s" % cfg, flush=True)

pd.DataFrame(win_rows).to_csv("taskC_rr_windows.csv", index=False)
pd.DataFrame(pair_rows).to_csv("taskC_rr_pairs.csv", index=False)
pd.DataFrame(band_rows).to_csv("taskC_rr_band_summary.csv", index=False)
pd.DataFrame(z_rows).to_csv("taskC_cell_zscores.csv", index=False)
print("wrote taskC_rr_windows/pairs/band_summary/cell_zscores")
