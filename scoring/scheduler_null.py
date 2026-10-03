
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import gdgb_common as C

# ---- SCHEDULER-NULL CONTROL ----
# Pass each REAL horizon stream (idx 1000..9999, the scored slots) through the
# harness's round quantization: rounds of edge_delta=50 consecutive events share
# one timestamp (the round's first event time). Then dedup arm (as used for M_gen)
# and compute M. Compare M_real vs M_quantized-real vs M_generated.
ROUND = 50  # edge_delta in every config
rows = []
A = pd.read_csv("taskA_dependence_metrics.csv")
for ds in C.INFO_DS:
    ref = C.load_ref(ds)
    sl = ref.iloc[1000:10000]  # scored horizon slots, incumbent window
    t = C.secs(sl["timestamp"]); a = sl["actor_id"].astype(str).to_numpy()
    ok = np.isfinite(t); t, a = t[ok], a[ok]
    order = np.argsort(t, kind="stable"); t, a = t[order], a[order]
    M_real = C.memory_coeff(C.iets_by_actor(*C.dedup(t, a)))
    tq = t.copy()
    for i in range(0, len(tq), ROUND):
        tq[i:i+ROUND] = tq[i]
    M_quant = C.memory_coeff(C.iets_by_actor(*C.dedup(tq, a)))
    sub = A[(A.status=="ok") & (A.arm=="dedup") & (A.dataset==ds) & (A.level.isin(["L0","L1","L2"]))]
    rows.append(dict(dataset=ds, M_real=M_real, M_quantized_real=M_quant,
                     M_gen_min=sub.M_gen.min(), M_gen_max=sub.M_gen.max(),
                     attenuation_frac=(M_real-M_quant)/M_real if M_real else np.nan))
sn = pd.DataFrame(rows)
sn.to_csv("scheduler_null_control.csv", index=False)
print(sn.round(3).to_string(index=False))

# ---- RR EMPIRICAL PERCENTILES ----
pairs = pd.read_csv("taskC_rr_pairs.csv")
cells = pd.read_csv("taskC_cell_zscores.csv")
out = []
for _, r in cells.iterrows():
    p = pairs[pairs.dataset==r.dataset]
    row = dict(config=r.config, dataset=r.dataset, level=r.level)
    for m in ["ks", "abs_M_gap", "rhythm_median_ks"]:
        Drr = p[m].dropna().to_numpy() if m in p.columns else np.array([])
        Dsr = r.get("D_"+m, np.nan)
        if Drr.size and np.isfinite(Dsr):
            # empirical exceedance p with add-one: P(RR >= sim-real)
            row["p_"+m] = float((1 + (Drr >= Dsr).sum()) / (Drr.size + 1))
            row["n_rr_"+m] = int(Drr.size)
        else:
            row["p_"+m] = np.nan; row["n_rr_"+m] = 0
    out.append(row)
pr = pd.DataFrame(out)
pr.to_csv("rr_percentiles.csv", index=False)
print()
print("cells with p_ks at floor (D > all RR pairs):", int((pr.p_ks <= 1/(pr.n_rr_ks+1)).sum()), "of", pr.p_ks.notna().sum())
print(pr[["config","p_ks","p_abs_M_gap","p_rhythm_median_ks","n_rr_ks"]].round(4).to_string(index=False))
