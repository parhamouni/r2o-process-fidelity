#!/usr/bin/env python
# Task B -- prereg 3B.3: event-count vs conditional-timing decomposition
# (EXPLORATORY: post-data). Two routes, both reported:
#  (1) rate/shape factorization: per-actor IETs normalized by the actor's mean
#      IET -> pooled "shape" distribution; rate reported separately (duration
#      log-ratio; per-actor count agreement).
#  (2) count-conditioned resampling: per shared actor subsample both sides to
#      the same IET count, pooled KS, 20 replicates.
# KS arms per prereg 2.5: jitter is primary for distributional stats; raw also shown.
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
import gdgb_common as C

RNG = np.random.default_rng(0)


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


rows = []
for ds in C.INFO_DS:
    print("[ref] %s" % ds, flush=True)
    ref = C.load_ref(ds)
    for cfg, level in C.cells_for(ds):
        gen, run = C.load_gen(cfg)
        if gen is None:
            rows.append(dict(config=cfg, dataset=ds, level=level, status=run))
            continue
        ref_gen = C.match_ref(ref, gen)
        t_g = C.secs(gen["timestamp"]); a_g = gen["actor_id"].to_numpy()
        t_r = C.secs(ref_gen["timestamp"]); a_r = ref_gen["actor_id"].to_numpy()
        row = dict(config=cfg, dataset=ds, level=level, status="ok", run_name=run,
                   cell_flag=C.CELL_FLAGS.get(cfg, ""), n_generated=len(gen))

        # ---- RATE: duration and per-actor event-count agreement ----
        fg, fr = t_g[np.isfinite(t_g)], t_r[np.isfinite(t_r)]
        dur_g = fg.max() - fg.min() if fg.size > 1 else np.nan
        dur_r = fr.max() - fr.min() if fr.size > 1 else np.nan
        row["duration_gen_s"] = dur_g; row["duration_ref_s"] = dur_r
        row["log_duration_ratio"] = (np.log(dur_g / dur_r)
                                     if dur_g and dur_r and dur_g > 0 and dur_r > 0 else np.nan)
        cg = pd.Series(a_g).value_counts(); cr = pd.Series(a_r).value_counts()
        shared_actors = cg.index.intersection(cr.index)
        row["n_actors_gen"] = int(cg.size); row["n_actors_ref"] = int(cr.size)
        row["n_actors_shared"] = int(shared_actors.size)
        row["actor_overlap_frac"] = float(shared_actors.size / max(cr.size, 1))
        if shared_actors.size >= 10:
            rho, p = spearmanr(cg[shared_actors], cr[shared_actors])
            row["count_spearman_shared"] = float(rho); row["count_spearman_p"] = float(p)
        else:
            row["count_spearman_shared"] = np.nan; row["count_spearman_p"] = np.nan

        # ---- TIMING: raw vs shape vs count-conditioned (per-actor pooled) ----
        for arm, tt in (("as_generated", t_g), ("jitter", C.jitter(t_g))):
            g_by = C.iets_by_actor(tt, a_g)
            r_by = C.iets_by_actor(t_r, a_r)
            gi, ri = C.pooled(g_by), C.pooled(r_by)
            sfx = "" if arm == "as_generated" else "_jit"
            if gi.size > 1 and ri.size > 1:
                row["ks_raw" + sfx] = float(ks_2samp(gi, ri).statistic)
            else:
                row["ks_raw" + sfx] = np.nan
            gs, rs = shape_pool(g_by), shape_pool(r_by)
            if gs.size > 1 and rs.size > 1:
                row["ks_shape" + sfx] = float(ks_2samp(gs, rs).statistic)
                row["w1_shape" + sfx] = float(wasserstein_distance(gs, rs))
                row["burst_shape_gap" + sfx] = C.burstiness(gs) - C.burstiness(rs)
            else:
                row["ks_shape" + sfx] = np.nan
                row["w1_shape" + sfx] = np.nan
                row["burst_shape_gap" + sfx] = np.nan
            m, sd, nsh = count_conditioned_ks(g_by, r_by)
            row["ks_countcond" + sfx] = m
            row["ks_countcond_sd" + sfx] = sd
            row["n_countcond_actors" + sfx] = nsh
        rows.append(row)
        print("[done] %s" % cfg, flush=True)

df = pd.DataFrame(rows)
df.to_csv("taskB_count_conditional.csv", index=False)
print("wrote taskB_count_conditional.csv (%d rows)" % len(df))
