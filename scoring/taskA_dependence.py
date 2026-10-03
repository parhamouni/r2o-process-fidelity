#!/usr/bin/env python
# Task A -- R6 dependence metrics on all existing cells (EXPLORATORY: post-data
# per the prereg honesty ledger; confirmatory counterpart = E-REP/G1).
# Metrics: memory coefficient M (per-actor consecutive-IET Pearson), global
# lag-1 IET autocorrelation rho1, node-rhythm divergence P(dt|u) (per-actor KS,
# shared actors). Arms per G0.6/prereg 2.5: as_generated; dedup (primary for
# moment stats M); jitter (primary for distributional stats incl. rhythm KS).
# Reference side is always as-is.
import numpy as np, pandas as pd
import gdgb_common as C

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
        r_by = C.iets_by_actor(t_r, a_r)
        M_ref = C.memory_coeff(r_by)
        rho1_ref = C.rho1_global(t_r)
        arms = {"as_generated": (t_g, a_g),
                "dedup": C.dedup(t_g, a_g),
                "jitter": (C.jitter(t_g), a_g)}
        for arm, (t, a) in arms.items():
            g_by = C.iets_by_actor(t, a)
            M_gen = C.memory_coeff(g_by)
            rho1_gen = C.rho1_global(t)
            med_ks, mean_ks, n_sh = C.rhythm_divergence(g_by, r_by, min_iet=5)
            rows.append(dict(
                config=cfg, dataset=ds, level=level, arm=arm, status="ok",
                run_name=run, cell_flag=C.CELL_FLAGS.get(cfg, ""),
                n_generated=len(gen),
                M_gen=M_gen, M_ref=M_ref, M_gap=M_gen - M_ref,
                rho1_gen=rho1_gen, rho1_ref=rho1_ref, rho1_gap=rho1_gen - rho1_ref,
                rhythm_median_ks=med_ks, rhythm_mean_ks=mean_ks,
                rhythm_n_shared_actors=n_sh))
        print("[done] %s" % cfg, flush=True)

df = pd.DataFrame(rows)
df.to_csv("taskA_dependence_metrics.csv", index=False)
print("wrote taskA_dependence_metrics.csv (%d rows)" % len(df))
