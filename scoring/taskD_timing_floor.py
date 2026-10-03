#!/usr/bin/env python
# Task D -- CPU half of G2 (PILOT): non-LLM timing floor on the incumbent
# window. Protocol mirrors the agent runs: condition on the 1000-edge seed
# prefix (original_edge_idx 0..999), emit 9000 event times, score against the
# reference horizon (idx 1000..9999) on the GLOBAL stream. Baselines:
#   poisson    homogeneous rate fitted on seed
#   bootstrap  i.i.d. resample of seed global IETs
#   hawkes     univariate exponential-kernel Hawkes, MLE on seed, Ogata sim
# These are actor-less generators -> global-stream metrics only (declared).
# 20 simulation seeds per baseline -> mean +/- sd per metric.
import os
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.optimize import minimize
import gdgb_common as C

N_SEED, N_HOR, N_SIMS = 1000, 9000, 20


def hawkes_nll(params, t, T):
    # params in log space: mu, alpha (branching in (0,1) via sigmoid), beta
    lmu, la, lb = params
    mu, beta = np.exp(lmu), np.exp(lb)
    alpha = 1.0 / (1.0 + np.exp(-la))  # branching ratio < 1
    n = t.size
    S = np.zeros(n)
    for i in range(1, n):
        S[i] = np.exp(-beta * (t[i] - t[i - 1])) * (S[i - 1] + 1.0)
    lam = mu + alpha * beta * S
    if np.any(lam <= 0):
        return 1e12
    ll = np.log(lam).sum() - mu * T - alpha * np.sum(1.0 - np.exp(-beta * (T - t)))
    return -ll


def fit_hawkes(t):
    t = np.sort(t - t.min())
    T = t.max() if t.max() > 0 else 1.0
    scale = T / t.size  # rescale so mean IET ~ 1
    ts = t / scale
    Ts = T / scale
    best = None
    for la0 in (-1.0, 0.5):
        for lb0 in (-1.0, 0.5):
            r = minimize(hawkes_nll, x0=np.array([np.log(0.5), la0, lb0]),
                         args=(ts, Ts), method="Nelder-Mead",
                         options=dict(maxiter=2000, xatol=1e-4, fatol=1e-4))
            if best is None or r.fun < best.fun:
                best = r
    lmu, la, lb = best.x
    return dict(mu=np.exp(lmu), alpha=1.0 / (1.0 + np.exp(-la)),
                beta=np.exp(lb), scale=scale, nll=float(best.fun))


def sim_hawkes(p, n_events, rng):
    mu, alpha, beta = p["mu"], p["alpha"], p["beta"]
    t, events = 0.0, []
    S = 0.0  # sum of exp(-beta(t - t_j)) at current t
    lam = mu
    while len(events) < n_events:
        lam_bar = mu + alpha * beta * (S + (1 if events else 0))
        lam_bar = max(lam_bar, mu + alpha * beta * S, 1e-12)
        w = rng.exponential(1.0 / lam_bar)
        S = S * np.exp(-beta * w)
        t += w
        lam = mu + alpha * beta * S
        if rng.uniform() * lam_bar <= lam:
            events.append(t)
            S += 1.0
    return np.array(events) * p["scale"]


def metrics(sim_iets, ref_iets, ref_rho_by=None):
    out = dict(ks=float(ks_2samp(sim_iets, ref_iets).statistic),
               w1=float(wasserstein_distance(sim_iets, ref_iets)),
               burst_gap=C.burstiness(sim_iets) - C.burstiness(ref_iets))
    d = sim_iets
    if d.size > 11 and d[:-1].std() > 0 and d[1:].std() > 0:
        out["rho1_sim"] = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    else:
        out["rho1_sim"] = np.nan
    return out


rows = []
for ds in C.INFO_DS:
    print("[ds] %s" % ds, flush=True)
    ref = C.load_ref(ds)
    win = ref[ref["original_edge_idx"] < N_SEED + N_HOR]
    t = C.secs(win["timestamp"])
    idx = win["original_edge_idx"].to_numpy()
    seed_t = np.sort(t[(idx < N_SEED) & np.isfinite(t)])
    hor_t = np.sort(t[(idx >= N_SEED) & np.isfinite(t)])
    ref_iets = np.diff(hor_t)
    seed_iets = np.diff(seed_t)
    dur = seed_t.max() - seed_t.min()
    rate = (seed_t.size - 1) / dur if dur > 0 else np.nan
    rr = ref_iets
    ref_rho1 = (float(np.corrcoef(rr[:-1], rr[1:])[0, 1])
                if rr.size > 11 and rr[:-1].std() > 0 and rr[1:].std() > 0 else np.nan)
    hp = fit_hawkes(seed_t)
    print("  hawkes mu=%.3g alpha=%.3f beta=%.3g nll=%.1f"
          % (hp["mu"], hp["alpha"], hp["beta"], hp["nll"]), flush=True)
    sims = {
        "poisson": lambda rng: np.cumsum(rng.exponential(1.0 / rate, N_HOR)),
        "bootstrap": lambda rng: np.cumsum(rng.choice(seed_iets, N_HOR, replace=True)),
        "hawkes": lambda rng: sim_hawkes(hp, N_HOR, rng),
    }
    for name, f in sims.items():
        vals = []
        for s in range(N_SIMS):
            rng = np.random.default_rng(1000 + s)
            st = f(rng)
            vals.append(metrics(np.diff(np.sort(st)), ref_iets))
        vdf = pd.DataFrame(vals)
        row = dict(dataset=ds, baseline=name, n_sims=N_SIMS,
                   ref_burst=C.burstiness(ref_iets), ref_rho1=ref_rho1,
                   seed_rate_per_s=rate)
        if name == "hawkes":
            row.update(hawkes_mu=hp["mu"], hawkes_alpha=hp["alpha"], hawkes_beta=hp["beta"])
        for k in ("ks", "w1", "burst_gap", "rho1_sim"):
            row[k + "_mean"] = float(vdf[k].mean())
            row[k + "_sd"] = float(vdf[k].std(ddof=1))
        rows.append(row)
        print("  [done] %s ks=%.3f" % (name, row["ks_mean"]), flush=True)

df = pd.DataFrame(rows)
# context rows: agent global-stream KS/W1/burstiness for the same window. The paper's
# archived table is read if present; otherwise only the baseline rows are written.
ARCH = os.path.join(C.REPO, "GDGB-ALGO/session_2026-08-12_reviewer_response_ARCHIVE/analysis/reviewer_response/temporal_edge_fidelity.csv")
if os.path.exists(ARCH):
    arch = pd.read_csv(ARCH)
    ag = (arch[arch["dataset"].isin(C.INFO_DS)]
          [["dataset", "level", "ks_iet_global", "w1_iet_global", "burst_gap_global"]]
          .rename(columns={"level": "baseline", "ks_iet_global": "ks_mean",
                           "w1_iet_global": "w1_mean", "burst_gap_global": "burst_gap_mean"}))
    ag["baseline"] = "agent_" + ag["baseline"].astype(str)
    ag["n_sims"] = 1
    out = pd.concat([df, ag], ignore_index=True)
else:
    out = df
out.to_csv("taskD_timing_floor.csv", index=False)
print("wrote taskD_timing_floor.csv (%d rows)" % len(out))
