#!/usr/bin/env python
"""Matched-horizon control for the exploratory Qwen3-8B L0/L1 pairs.

Reviewer objection this answers: the four 8B L1 lanes were stopped at round
boundaries (4,900-5,696 events) while all four L0 lanes ran the full 9,000-event
horizon, so the within-dataset L0->L1 contrasts (notably wikirevision M
0.162 -> 0.008) compare arms at different n.

Control: truncate each L0 generated stream to its paired L1 event count, in
generation order, and re-score with the paper's own primitives and arms:
  - M          memory coefficient, dedup arm      (moment stat; prereg 2.5)
  - KS         pooled per-actor IET KS, as-generated arm (matches taskC D_ks)
  - KS_cc      count-conditioned KS, jitter arm    (matches taskB ks_countcond_jit)
  - Z_|dM|,Z_KS   against the dataset's existing RR band (taskC_rr_band_summary.csv)
The reference side is re-matched to the truncated stream via gdgb_common.match_ref,
so the comparison stays length-matched on both sides.

Run from the analysis directory that holds gdgb_common.py and
taskC_rr_band_summary.csv. Repo is read-only. Writes:
  matched_horizon_8b.csv, tab_matched_horizon.tex

Not covered here: degree MMD. Structural scoring lives in the separate
evaluator, and a 5k-edge structural value is not comparable to the grid's
10k-edge cells -- re-run the evaluator on the truncated stream if you want it.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scoring"))
import gdgb_common as C   # needs R2O_GDGB_ROOT (reads the archived 8B runs)

BAND_CSV = os.environ.get("BAND_CSV", "taskC_rr_band_summary.csv")
REPS_CC = 20
RNG = np.random.default_rng(0)

# Archived runs pinned to the ones scored in the paper (slm_taskA_metrics.csv).
CELLS = {
    "wikirevision": {
        "L0": ("wikirevision_l0_qwen3_8b_api", "offline-run-20260831_161404-xlj8ob77"),
        "L1": ("wikirevision_l1_qwen3_8b_api", "offline-run-20260831_161528-nsyz9nw8"),
    },
    "dianping": {
        "L0": ("dianping_l0_qwen3_8b_api", "offline-run-20260901_102437-e7oz9rsf"),
        "L1": ("dianping_l1_qwen3_8b_api", "offline-run-20260901_102841-esroswp5"),
    },
    "weibo_daily": {
        "L0": ("weibo_daily_l0_qwen3_8b_api", "offline-run-20260901_102529-6sn3ijje"),
        "L1": ("weibo_daily_l1_qwen3_8b_api", "offline-run-20260901_102958-rme99r27"),
    },
    "weibo_tech": {
        "L0": ("weibo_tech_l0_qwen3_8b_api", "offline-run-20260901_102658-ipf8cwd3"),
        "L1": ("weibo_tech_l1_qwen3_8b_api", "offline-run-20260901_103124-0s4w39kn"),
    },
}


def load_gen_pinned(cfg, run):
    """Non-seed generated rows from a PINNED archived run (not latest-run)."""
    p = os.path.join(C.CONFIGS, cfg, "generated_data/wandb", run, "files/media/edges.csv")
    if not os.path.exists(p):
        return None, "missing:%s" % p
    g_all = pd.read_csv(p, index_col=0, low_memory=False)
    seed = g_all["seed"].astype(str).str.lower().isin(["true", "1"])
    g = g_all[~seed].copy()
    g["actor_id"] = g["actor_id"].astype(str)
    return g, run


def count_conditioned_ks(g_by, r_by, reps=REPS_CC):
    shared = [k for k in g_by if k in r_by and g_by[k].size >= 2 and r_by[k].size >= 2]
    if not shared:
        return np.nan, 0
    vals = []
    for _ in range(reps):
        gs, rs = [], []
        for k in shared:
            n = min(g_by[k].size, r_by[k].size)
            gs.append(RNG.choice(g_by[k], n, replace=False))
            rs.append(RNG.choice(r_by[k], n, replace=False))
        vals.append(ks_2samp(np.concatenate(gs), np.concatenate(rs)).statistic)
    return float(np.mean(vals)), len(shared)


def score(ref, gen):
    """Paper-matched panel for one generated stream."""
    ref_gen = C.match_ref(ref, gen)
    t_g = C.secs(gen["timestamp"]); a_g = gen["actor_id"].to_numpy()
    t_r = C.secs(ref_gen["timestamp"]); a_r = ref_gen["actor_id"].to_numpy()
    r_by = C.iets_by_actor(t_r, a_r)

    # KS: as-generated arm, pooled per-actor IETs (taskC pair_metrics convention)
    g_by_raw = C.iets_by_actor(t_g, a_g)
    pg, pr = C.pooled(g_by_raw), C.pooled(r_by)
    ks = float(ks_2samp(pg, pr).statistic) if (pg.size > 1 and pr.size > 1) else np.nan

    # M: dedup arm (moment statistic)
    t_d, a_d = C.dedup(t_g, a_g)
    M_gen = C.memory_coeff(C.iets_by_actor(t_d, a_d))
    M_ref = C.memory_coeff(r_by)

    # count-conditioned KS: jitter arm (taskB convention)
    g_by_jit = C.iets_by_actor(C.jitter(t_g), a_g)
    ks_cc, n_cc = count_conditioned_ks(g_by_jit, r_by)

    return dict(n=len(gen), ks=ks, M_gen=M_gen, M_ref=M_ref,
                M_gap=M_gen - M_ref, abs_M_gap=abs(M_gen - M_ref),
                ks_countcond_jit=ks_cc, n_countcond_actors=n_cc,
                n_ref_matched=len(ref_gen))


band = pd.read_csv(BAND_CSV).set_index("dataset")
rows = []
for ds, arms in CELLS.items():
    print("[ref] %s" % ds, flush=True)
    ref = C.load_ref(ds)
    l1_cfg, l1_run = arms["L1"]
    l0_cfg, l0_run = arms["L0"]
    g1, s1 = load_gen_pinned(l1_cfg, l1_run)
    g0, s0 = load_gen_pinned(l0_cfg, l0_run)
    if g1 is None or g0 is None:
        rows.append(dict(dataset=ds, arm="ERROR", status="%s | %s" % (s0, s1)))
        continue
    n1 = len(g1)
    variants = [
        ("L0-full", g0, "full 9,000-event horizon as scored in the paper"),
        ("L0-trunc", g0.iloc[:n1].copy(), "L0 truncated to the L1 event count"),
        ("L1", g1, "L1 as scored in the paper (stopped at round boundary)"),
    ]
    for arm, g, note in variants:
        r = score(ref, g)
        r.update(dataset=ds, arm=arm, note=note, status="ok",
                 config=l0_cfg if arm.startswith("L0") else l1_cfg,
                 run_name=l0_run if arm.startswith("L0") else l1_run,
                 n_target_L1=n1)
        for k, col in (("ks", "ks"), ("abs_M_gap", "abs_M_gap")):
            mu, sig = band.loc[ds, "mu_" + k], band.loc[ds, "sigma_" + k]
            r["Z_" + k] = (r[col] - mu) / sig if sig and np.isfinite(sig) and sig > 0 else np.nan
        rows.append(r)
        print("  [%s] n=%d KS=%.3f M=%.3f (ref %.3f) Z|dM|=%.2f"
              % (arm, r["n"], r["ks"], r["M_gen"], r["M_ref"], r["Z_abs_M_gap"]), flush=True)

df = pd.DataFrame(rows)
cols = ["dataset", "arm", "config", "run_name", "n", "n_target_L1", "n_ref_matched",
        "ks", "Z_ks", "M_gen", "M_ref", "M_gap", "abs_M_gap", "Z_abs_M_gap",
        "ks_countcond_jit", "n_countcond_actors", "status", "note"]
df = df.reindex(columns=[c for c in cols if c in df.columns])
df.to_csv("matched_horizon_8b.csv", index=False)
print("wrote matched_horizon_8b.csv (%d rows)" % len(df))

# ---- supplementary table fragment ----
ok = df[df.get("status", "") == "ok"]
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{Matched-horizon control for the exploratory Qwen3-8B pairs.}",
         r"Each L1 lane was stopped at a round boundary; here each L0 stream is also",
         r"truncated to its paired L1 event count and re-scored identically (KS:",
         r"as-generated arm; $M$: dedup arm; $Z$ against the dataset's real--real band).",
         r"The reference slot is re-matched to each truncated stream.}",
         r"\label{tab:matchedhorizon}",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Dataset & Arm & $n$ & KS & $Z_{\mathrm{KS}}$ & $M_{\mathrm{gen}}$ & "
         r"$M_{\mathrm{ref}}$ & $Z_{|\Delta M|}$ \\", r"\midrule"]
for ds in CELLS:
    sub = ok[ok.dataset == ds]
    for _, r in sub.iterrows():
        lines.append("%s & %s & %d & %.3f & %+.1f & %+.3f & %.3f & %+.1f \\\\"
                     % (ds.replace("_", r"\_"), r["arm"], r["n"], r["ks"], r["Z_ks"],
                        r["M_gen"], r["M_ref"], r["Z_abs_M_gap"]))
    lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("tab_matched_horizon.tex", "w").write("\n".join(lines) + "\n")
print("wrote tab_matched_horizon.tex")
