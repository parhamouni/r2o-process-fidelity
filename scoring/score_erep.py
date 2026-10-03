
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, spearmanr
import gdgb_common as C

import os
WT = os.path.join(os.environ["R2O_GDGB_ROOT"], "GDGB-ALGO")  # E-REP runs live under the same data root
RUNS = {
 "L0": WT+"/TDGG/LLMGraph/tasks/general/configs/wikirevision_erep_l0_gpt4omini_api/generated_data/wandb/offline-run-20260825_200750-xm4f2tn4/files/media/edges.csv",
 "L1": WT+"/TDGG/LLMGraph/tasks/general/configs/wikirevision_erep_l1_gpt4omini_api/generated_data/wandb/offline-run-20260825_213813-x6xqswdp/files/media/edges.csv",
}
ref = pd.read_csv(WT+"/GDGB_dataset/WikiRevision/edge_wikirevision.csv", index_col=0, low_memory=False)
ref["original_edge_idx"] = ref.index
ref["actor_id"] = ref["user_node_id"].astype(str)

# fair-window RR band: W-REPRESENT is one 10k window; build band from FULL wikirevision
# recency-era windows (matched construction from taskC, reuse the summary already computed)
band = pd.read_csv("taskC_rr_band_summary.csv")
b = band[band.dataset=="wikirevision"].iloc[0]

rows = []
for lvl, path in RUNS.items():
    g_all = pd.read_csv(path, index_col=0, low_memory=False)
    seed = g_all["seed"].astype(str).str.lower().isin(["true","1"])
    g = g_all[~seed].copy(); g["actor_id"] = g["actor_id"].astype(str)
    idx = pd.to_numeric(g["original_edge_idx"], errors="coerce").dropna().astype(int)
    ref_gen = ref[ref.index.isin(set(idx))]
    t_g, a_g = C.secs(g["timestamp"]), g["actor_id"].to_numpy()
    t_r, a_r = C.secs(ref_gen["timestamp"]), ref_gen["actor_id"].to_numpy()
    # structural proxy available cheaply: degree-dist KS over actors (official eval has full MMD)
    dg = pd.Series(a_g).value_counts(); dr = pd.Series(a_r).value_counts()
    deg_ks = float(ks_2samp(dg.to_numpy(), dr.to_numpy()).statistic)
    for arm, (t, a) in {"as_generated":(t_g,a_g), "dedup":C.dedup(t_g,a_g), "jitter":(C.jitter(t_g),a_g)}.items():
        g_by, r_by = C.iets_by_actor(t,a), C.iets_by_actor(t_r,a_r)
        gi, ri = C.pooled(g_by), C.pooled(r_by)
        ks = float(ks_2samp(gi,ri).statistic) if gi.size>1 and ri.size>1 else np.nan
        rows.append(dict(level=lvl, arm=arm, n_gen=len(g),
            ks=ks, burst_gap=C.burstiness(gi)-C.burstiness(ri),
            M_gen=C.memory_coeff(g_by), M_ref=C.memory_coeff(r_by),
            rhythm_ks=C.rhythm_divergence(g_by,r_by)[0],
            deg_ks=deg_ks,
            Z_ks=(ks-b.mu_ks)/b.sigma_ks,
            zero_iet_frac=float((gi==0).mean()) if gi.size else np.nan))
df = pd.DataFrame(rows)
df.to_csv("erep_stage1_scores.csv", index=False)
print(df.round(3).to_string(index=False))

# rate + decomposition
for lvl, path in RUNS.items():
    g_all = pd.read_csv(path, index_col=0, low_memory=False)
    g = g_all[~g_all["seed"].astype(str).str.lower().isin(["true","1"])]
    t_g = C.secs(g["timestamp"]); fg = t_g[np.isfinite(t_g)]
    idx = pd.to_numeric(g["original_edge_idx"], errors="coerce").dropna().astype(int)
    rg = ref[ref.index.isin(set(idx))]; t_r = C.secs(rg["timestamp"]); fr = t_r[np.isfinite(t_r)]
    print(lvl, "log_dur_ratio %.2f" % np.log((fg.max()-fg.min())/(fr.max()-fr.min())),
          "| gen span %.0f d vs ref %.1f d" % ((fg.max()-fg.min())/86400, (fr.max()-fr.min())/86400))
