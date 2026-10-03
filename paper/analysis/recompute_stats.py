#!/usr/bin/env python
"""Recompute every statistic printed in the SLM-Agents submission from the
rawest available sources (precheck CSVs; run edges.csv via gdgb_common for
stats not covered by a committed CSV). Output: analysis/verified_stats.json
+ analysis/cells_main.csv (per-cell table for tex/figures).
Read-only on the data mount. """
import json, os, sys
import numpy as np, pandas as pd

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "scores")
PRE = os.path.join(SRC, "precheck")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scoring"))
import gdgb_common as C   # needs R2O_GDGB_ROOT: some statistics (zero-IET fraction, tie share, structural eval) are read from the raw runs

out = {}

# ---------- Task A: dependence metrics ----------
A = pd.read_csv(os.path.join(PRE, "taskA_dependence_metrics.csv"))
A = A[A.status == "ok"]
ded = A[A.arm == "dedup"].set_index("config")
jit = A[A.arm == "jitter"].set_index("config")
qwen_lv = {"L0", "L1", "L2"}
Aq = ded[ded.level.isin(qwen_lv)]
Aapi = ded[~ded.level.isin(qwen_lv)]
Mref = ded.groupby("dataset").M_ref.first()
out["M_ref_min"] = float(ded.M_ref.min()); out["M_ref_max"] = float(ded.M_ref.max())
out["M_ref_by_ds"] = {k: float(v) for k, v in Mref.items()}
out["M_gen_qwen_min"] = float(Aq.M_gen.min()); out["M_gen_qwen_max"] = float(Aq.M_gen.max())
out["M_gen_all_min"] = float(ded.M_gen.min()); out["M_gen_all_max"] = float(ded.M_gen.max())
out["M_gen_api"] = {i: float(v) for i, v in Aapi.M_gen.items()}
# ladder direction: L2 vs L0 and L1 vs L0 per dataset (Qwen)
lad = {}
for ds, g in Aq.reset_index().groupby("dataset"):
    m = g.set_index("level").M_gen
    lad[ds] = dict(L0=float(m["L0"]), L1=float(m["L1"]), L2=float(m["L2"]))
out["M_ladder_qwen"] = lad
out["ladder_L2_below_L0_count"] = int(sum(v["L2"] < v["L0"] for v in lad.values()))
out["ladder_L1_below_L0_count"] = int(sum(v["L1"] < v["L0"] for v in lad.values()))
# rho1
out["rho1_ref_max"] = float(ded.rho1_ref.max())
# node rhythm (jitter arm primary for distributional)
jq = jit[jit.level.isin(qwen_lv | {"L0-api", "L1-api"})]
out["rhythm_gen_min"] = float(jit.rhythm_median_ks.min())
out["rhythm_gen_max"] = float(jit.rhythm_median_ks.max())

# ---------- scheduler-null control ----------
S = pd.read_csv(os.path.join(PRE, "scheduler_null_control.csv")).set_index("dataset")
out["sched_null"] = S.round(6).to_dict("index")
out["atten_max"] = float(S.attenuation_frac.max())
out["Mq_min"] = float(S.M_quantized_real.min()); out["Mq_max"] = float(S.M_quantized_real.max())
out["gen_below_qfloor_count"] = int((S.M_gen_max < S.M_quantized_real).sum())
out["atten_by_ds"] = {k: float(v) for k, v in S.attenuation_frac.items()}

# ---------- Task B: counts / rate ----------
B = pd.read_csv(os.path.join(PRE, "taskB_count_conditional.csv"))
B = B[B.status == "ok"].set_index("config")
ratio = (B.ks_countcond / B.ks_raw).dropna()
out["countcond_ratio_median"] = float(ratio.median())
out["countcond_ratio_min"] = float(ratio.min()); out["countcond_ratio_max"] = float(ratio.max())
out["logdur_min"] = float(B.log_duration_ratio.min()); out["logdur_max"] = float(B.log_duration_ratio.max())

# ---------- Task C: z-scores / RR band / percentiles ----------
Z = pd.read_csv(os.path.join(PRE, "taskC_cell_zscores.csv")).set_index("config")
out["n_cells"] = int(len(Z))
out["Zks_min"] = float(Z.Z_ks.min()); out["Zks_max"] = float(Z.Z_ks.max())
out["Zks_outside2"] = int((Z.Z_ks.abs() > 2).sum())
out["ZM_outside2"] = int((Z.Z_abs_M_gap.abs() > 2).sum())
out["Zburst_outside2"] = int((Z.Z_abs_burst_gap.abs() > 2).sum())
out["Zburst_inband_cells"] = list(Z.index[Z.Z_abs_burst_gap.abs() <= 2])
out["Zrhythm_outside2"] = int((Z.Z_rhythm_median_ks.abs() > 2).sum())
band = pd.read_csv(os.path.join(PRE, "taskC_rr_band_summary.csv")).set_index("dataset")
out["rr_mu_ks_min"] = float(band.mu_ks.min()); out["rr_mu_ks_max"] = float(band.mu_ks.max())
out["rr_mu_rhythm_min"] = float(band.mu_rhythm_median_ks.min())
out["rr_mu_rhythm_max"] = float(band.mu_rhythm_median_ks.max())
out["rr_n_pairs"] = {k: int(v) for k, v in band.n_pairs.items()}
P = pd.read_csv(os.path.join(PRE, "rr_percentiles.csv")).set_index("config")
out["q_ks_all"] = {i: float(v) for i, v in P.p_ks.items()}
out["q_ks_max"] = float(P.p_ks.max())
out["q_ks_floor_60"] = float(1/61); out["q_ks_floor_13"] = float(1/14)
out["q_ks_all_at_floor"] = bool(all(abs(v - (1/(n+1))) < 1e-9 for v, n in zip(P.p_ks, P.n_rr_ks)))
dp_r = P[P.dataset == "dianping"].p_rhythm_median_ks
out["q_rhythm_dianping_min"] = float(dp_r.min()); out["q_rhythm_dianping_max"] = float(dp_r.max())

# ---------- Task D: baselines ----------
D = pd.read_csv(os.path.join(PRE, "taskD_timing_floor.csv"))
piv = D.pivot(index="dataset", columns="baseline", values="ks_mean")
agents = piv[["agent_L0", "agent_L1", "agent_L2"]]
out["boot_ks"] = {k: float(v) for k, v in piv.bootstrap.items()}
out["boot_ks_min"] = float(piv.bootstrap.min()); out["boot_ks_max"] = float(piv.bootstrap.max())
out["boot_beats_all_agents"] = {ds: bool(piv.bootstrap[ds] < agents.loc[ds].min()) for ds in piv.index}
out["boot_beats_count"] = int(sum(out["boot_beats_all_agents"].values()))
out["best_agent_ks"] = {ds: float(agents.loc[ds].min()) for ds in piv.index}
ba = {ds: v for ds, v in out["best_agent_ks"].items() if out["boot_beats_all_agents"][ds]}
out["best_agent_ks_min_where_beaten"] = float(min(ba.values()))
out["best_agent_ks_max_where_beaten"] = float(max(ba.values()))
bg = D.pivot(index="dataset", columns="baseline", values="burst_gap_mean")
out["hawkes_burst_gap_dianping"] = float(bg.loc["dianping", "hawkes"])
out["agent_burst_gap_dianping_min"] = float(bg.loc["dianping", ["agent_L0","agent_L1","agent_L2"]].min())
out["hawkes_beats_agents_burst_dianping"] = bool(abs(bg.loc["dianping","hawkes"]) < bg.loc["dianping",["agent_L0","agent_L1","agent_L2"]].abs().min())

# ---------- E-REP stage 1 ----------
E = pd.read_csv(os.path.join(SRC, "erep", "erep_stage1_scores.csv"))
Ej = E[E.arm == "jitter"].set_index("level")
Ed = E[E.arm == "dedup"].set_index("level")
Eg = E[E.arm == "as_generated"].set_index("level")
out["erep_ks_jit_L0"] = float(Ej.loc["L0","ks"]); out["erep_ks_jit_L1"] = float(Ej.loc["L1","ks"])
out["erep_M_ref"] = float(Ej.M_ref.iloc[0])
out["erep_M_gen_L0"] = float(Ed.loc["L0","M_gen"]); out["erep_M_gen_L1"] = float(Ed.loc["L1","M_gen"])
out["erep_Zks_jit_L0"] = float(Ej.loc["L0","Z_ks"]); out["erep_Zks_jit_L1"] = float(Ej.loc["L1","Z_ks"])
out["erep_Zks_asgen_min"] = float(Eg.Z_ks.min()); out["erep_Zks_asgen_max"] = float(Eg.Z_ks.max())
out["erep_zero_iet_L0"] = float(Eg.loc["L0","zero_iet_frac"]); out["erep_zero_iet_L1"] = float(Eg.loc["L1","zero_iet_frac"])

# ---------- structural degree MMD from official eval CSVs ----------
EV = os.path.join(C.REPO, "GDGB-ALGO/TDGG/evaluate/general/graph_structural_results")
def final_mmd(cfg, base=EV):
    p = os.path.join(base, "eval_configs_%s_latest-run.csv" % cfg)
    if not os.path.exists(p): return None
    df = pd.read_csv(p, index_col=0)
    df = df[pd.to_numeric(df["edges"], errors="coerce").notna()]
    return float(df["degree_mmd"].iloc[-1]) if len(df) else None
mmd = {}
for ds in C.INFO_DS:
    row = {}
    for lvl, sfx in C.LEVELS:
        row[lvl] = final_mmd("%s_%s" % (ds, sfx))
    mmd[ds] = row
for cfg, (ds, lvl) in C.BACKBONE_CELLS.items():
    mmd[ds][lvl] = final_mmd(cfg)
# Contamination handling (matches CELL_FLAGS provenance):
# - weibo_tech L2 structural eval: use the archived pre-contamination run.
# - weibo_daily L2 structural eval: only the contaminated newer run has an
#   eval CSV -> excluded (None).
def final_mmd_file(fn):
    p = os.path.join(EV, fn)
    if not os.path.exists(p): return None
    df = pd.read_csv(p, index_col=0)
    df = df[pd.to_numeric(df["edges"], errors="coerce").notna()]
    return float(df["degree_mmd"].iloc[-1]) if len(df) else None
mmd["weibo_tech"]["L2"] = final_mmd_file(
    "eval_configs_weibo_tech_memory_v1_offline-run-20260724_113841-3azdhedc.csv")
mmd["weibo_daily"]["L2"] = None
out["degree_mmd"] = mmd
d01 = [(mmd[ds]["L1"] - mmd[ds]["L0"]) / mmd[ds]["L0"] for ds in C.INFO_DS]
d02 = [(mmd[ds]["L2"] - mmd[ds]["L0"]) / mmd[ds]["L0"] for ds in C.INFO_DS
       if mmd[ds]["L2"] is not None]
out["mmd_L0L1_pct"] = [float(100*x) for x in d01]
out["mmd_L0L1_mean_pct"] = float(100*np.mean(d01))
out["mmd_L0L1_improved_count"] = int(sum(x < 0 for x in d01))
out["mmd_L0L2_mean_pct"] = float(100*np.mean(d02))
out["mmd_L0L2_n"] = len(d02)
out["mmd_L0L2_improved_count"] = int(sum(x < 0 for x in d02))
# KS deltas (taskB ks_raw), Qwen ladder
ksd = {}
for ds in C.INFO_DS:
    r = B[B.dataset == ds]
    k = {lvl: float(r[r.level == lvl].ks_raw.iloc[0]) for lvl in ["L0","L1","L2"]}
    ksd[ds] = k
k01 = [(ksd[ds]["L1"]-ksd[ds]["L0"])/ksd[ds]["L0"] for ds in C.INFO_DS]
k02 = [(ksd[ds]["L2"]-ksd[ds]["L0"])/ksd[ds]["L0"] for ds in C.INFO_DS]
out["ks_L0L1_mean_pct"] = float(100*np.mean(k01)); out["ks_L0L1_pct"] = [float(100*x) for x in k01]
out["ks_L0L2_mean_pct"] = float(100*np.mean(k02))
out["ks_L0L1_sign_mixed"] = bool(min(k01) < 0 < max(k01))
out["ks_L0L2_sign_mixed"] = bool(min(k02) < 0 < max(k02))
# E-REP structural MMD (worktree eval CSVs)
EV2 = os.path.join(C.REPO, "GDGB-ALGO/TDGG/evaluate/general/graph_structural_results")  # E-REP structural results (same root)
out["erep_mmd_L0"] = final_mmd("wikirevision_erep_l0_gpt4omini_api_offline-run-20260825_200750-xm4f2tn4".replace("_offline",""), base=EV2)
# fallback: exact filenames
def final_mmd_file(fn, base=EV2):
    p = os.path.join(base, fn)
    if not os.path.exists(p): return None
    df = pd.read_csv(p, index_col=0)
    df = df[pd.to_numeric(df["edges"], errors="coerce").notna()]
    return float(df["degree_mmd"].iloc[-1]) if len(df) else None
out["erep_mmd_L0"] = final_mmd_file("eval_configs_wikirevision_erep_l0_gpt4omini_api_offline-run-20260825_200750-xm4f2tn4.csv")
out["erep_mmd_L1"] = final_mmd_file("eval_configs_wikirevision_erep_l1_gpt4omini_api_latest-run.csv")

# ---------- from raw runs: zero-IET fraction, median-IET ratio, tie share of burstiness ----------
cells = []
for ds in C.INFO_DS:
    ref = C.load_ref(ds)
    for cfg, level in C.cells_for(ds):
        gen, run = C.load_gen(cfg)
        if gen is None: continue
        ref_gen = C.match_ref(ref, gen)
        t_g = C.secs(gen["timestamp"]); a_g = gen["actor_id"].to_numpy()
        t_r = C.secs(ref_gen["timestamp"]); a_r = ref_gen["actor_id"].to_numpy()
        g_by = C.iets_by_actor(t_g, a_g); r_by = C.iets_by_actor(t_r, a_r)
        pg, pr = C.pooled(g_by), C.pooled(r_by)
        zfrac = float((pg == 0).mean()) if pg.size else np.nan
        med_g = float(np.median(pg[pg > 0])) if (pg > 0).any() else np.nan
        med_r = float(np.median(pr[pr > 0])) if (pr > 0).any() else np.nan
        # burstiness: as-generated vs dedup arm
        bg_as = C.burstiness(pg) - C.burstiness(pr)
        td, ad = C.dedup(t_g, a_g)
        gd_by = C.iets_by_actor(td, ad)
        bg_dd = C.burstiness(C.pooled(gd_by)) - C.burstiness(pr)
        cells.append(dict(config=cfg, dataset=ds, level=level,
                          zero_iet_frac=zfrac, med_iet_ratio=med_g/med_r,
                          burst_gap_asgen=float(bg_as), burst_gap_dedup=float(bg_dd)))
cells = pd.DataFrame(cells).set_index("config")
out["zero_iet_min"] = float(cells.zero_iet_frac.min()); out["zero_iet_max"] = float(cells.zero_iet_frac.max())
# tie share of burstiness gap (Qwen cells): 1 - |gap_dedup|/|gap_asgen|
qc = cells[cells.level.isin(qwen_lv)]
share = 1 - (qc.burst_gap_dedup.abs() / qc.burst_gap_asgen.abs())
out["tie_share_median"] = float(share.median())
out["tie_share_min"] = float(share.min()); out["tie_share_max"] = float(share.max())
out["residual_same_sign_count"] = int(((qc.burst_gap_dedup * qc.burst_gap_asgen) > 0).sum())
out["n_qwen_cells"] = int(len(qc))
out["flip_cells"] = list(qc.index[(qc.burst_gap_dedup * qc.burst_gap_asgen) < 0])
# median IET ratio at each dataset's best structural (lowest MMD) Qwen cell
best_ratio = {}
for ds in C.INFO_DS:
    lvls = {l: v for l, v in mmd[ds].items() if l in qwen_lv and v is not None}
    best_lvl = min(lvls, key=lvls.get)
    cfg = "%s_%s" % (ds, dict(C.LEVELS)[best_lvl] if False else dict((a,b) for a,b in C.LEVELS)[best_lvl])
    best_ratio[ds] = dict(level=best_lvl, ratio=float(cells.loc[cfg, "med_iet_ratio"]))
out["best_struct_med_iet_ratio"] = best_ratio
rr = [v["ratio"] for v in best_ratio.values()]
out["best_struct_ratio_median"] = float(np.median(rr))
out["best_struct_ratio_max"] = float(np.max(rr))
out["best_struct_offby_gt2_count"] = int(sum(max(r, 1/r) > 2 for r in rr))

# ---------- per-cell main table ----------
main = []
for cfg in Z.index:
    ds = Z.loc[cfg, "dataset"]; lvl = Z.loc[cfg, "level"]
    backbone = "GPT-4o-Mini" if lvl.endswith("-api") else "Qwen3-30B-A3B"
    L = lvl.replace("-api", "")
    main.append(dict(config=cfg, dataset=ds, backbone=backbone, level=L,
                     degree_mmd=mmd[ds].get(lvl if lvl.endswith("-api") else L),
                     ks=float(Z.loc[cfg, "D_ks"]), Z_ks=float(Z.loc[cfg, "Z_ks"]),
                     M_gen=float(ded.loc[cfg, "M_gen"]), M_ref=float(ded.loc[cfg, "M_ref"]),
                     Z_M=float(Z.loc[cfg, "Z_abs_M_gap"]),
                     rhythm=float(jit.loc[cfg, "rhythm_median_ks"]),
                     q_ks=float(P.loc[cfg, "p_ks"]),
                     zero_iet=float(cells.loc[cfg, "zero_iet_frac"]),
                     logdur=float(B.loc[cfg, "log_duration_ratio"])))
main = pd.DataFrame(main)
main.to_csv(os.path.join(os.path.dirname(__file__), "cells_main.csv"), index=False)

# best structural Qwen cell per dataset -> its Z_ks
best = {}
for ds in C.INFO_DS:
    lvls = {l: v for l, v in mmd[ds].items() if l in qwen_lv and v is not None}
    bl = min(lvls, key=lvls.get)
    cfg = "%s_%s" % (ds, dict(C.LEVELS)[bl])
    best[ds] = dict(level=bl, Z_ks=float(Z.loc[cfg, "Z_ks"]))
out["best_struct_Zks"] = best
out["best_struct_Zks_min"] = float(min(v["Z_ks"] for v in best.values()))
out["best_struct_Zks_max"] = float(max(v["Z_ks"] for v in best.values()))
# backbone-swap deltas at fixed scaffold (API vs Qwen, same dataset+level)
pairs = {"dianping_l0_gpt4omini_api": ("dianping", "L0"),
         "wikirevision_l0_gpt4omini_api": ("wikirevision", "L0"),
         "wikirevision_l1_gpt4omini_api": ("wikirevision", "L1"),
         "weibo_tech_l0_gpt4omini_api": ("weibo_tech", "L0")}
bb = {}
for cfg, (ds, lvl) in pairs.items():
    qcfg = "%s_%s" % (ds, dict(C.LEVELS)[lvl])
    ks_api = float(Z.loc[cfg, "D_ks"]); ks_q = float(Z.loc[qcfg, "D_ks"])
    mmd_api = mmd[ds].get(lvl + "-api") if False else None
    bb["%s_%s" % (ds, lvl)] = dict(ks_pct=100*(ks_api/ks_q - 1))
out["backbone_ks_pct"] = {k: float(v["ks_pct"]) for k, v in bb.items()}
mmd_api_pairs = {}
for cfg, (ds, lvl) in C.BACKBONE_CELLS.items():
    a = final_mmd(cfg); q = mmd[ds].get(lvl.replace("-api",""))
    if a is not None and q is not None:
        mmd_api_pairs["%s_%s" % (ds, lvl)] = float(100*(a/q - 1))
out["backbone_mmd_pct"] = mmd_api_pairs

# ---- Exploratory Qwen3-8B cells (post-registration; recomputed from raw) ----
from scipy.stats import ks_2samp as _ks2
_rrp = pd.read_csv(os.path.join(PRE, "taskC_rr_pairs.csv"))
_bands = pd.read_csv(os.path.join(PRE, "taskC_rr_band_summary.csv")).set_index("dataset")
SLM_CELLS = [(f"{_ds}_{_l}_qwen3_8b_api", _l.upper(), _ds)
             for _ds in ["dianping", "wikirevision", "weibo_daily", "weibo_tech"]
             for _l in ["l0", "l1"]]
_ref_cache = {}
slm = {}
for _cfg, _lvl, _ds in SLM_CELLS:
    _band = _bands.loc[_ds]
    _rrw = _rrp[_rrp.dataset == _ds]["ks"].to_numpy()
    if _ds not in _ref_cache:
        _ref_cache[_ds] = C.load_ref(_ds)
    _ref = _ref_cache[_ds]
    _gen, _run = C.load_gen(_cfg)
    _rg = C.match_ref(_ref, _gen)
    _tg = C.secs(_gen["timestamp"]); _ag = _gen["actor_id"].to_numpy()
    _tr = C.secs(_rg["timestamp"]); _ar = _rg["actor_id"].to_numpy()
    _gby = C.iets_by_actor(_tg, _ag); _rby = C.iets_by_actor(_tr, _ar)
    _dt, _da = C.dedup(_tg, _ag)
    _dby = C.iets_by_actor(_dt, _da)
    _D = float(_ks2(C.pooled(_gby), C.pooled(_rby)).statistic)
    slm[f"{_ds}:{_lvl}"] = dict(
        n=int(len(_gen)), run=_run, dataset=_ds,
        ks=_D, Z_ks=float((_D - _band["mu_ks"]) / _band["sigma_ks"]),
        M_as_gen=float(C.memory_coeff(_gby)), M_dedup=float(C.memory_coeff(_dby)),
        M_ref=float(C.memory_coeff(_rby)),
        abs_M_gap=float(abs(C.memory_coeff(_gby) - C.memory_coeff(_rby))),
        Z_M=float((abs(C.memory_coeff(_gby) - C.memory_coeff(_rby)) - _band["mu_abs_M_gap"])
                  / _band["sigma_abs_M_gap"]),
        q_ks=float(((_rrw >= _D).sum() + 1) / (len(_rrw) + 1)),
        zero_iet=float((C.pooled(_gby) == 0).mean()),
        rhythm_jit=float(C.rhythm_divergence(
            C.iets_by_actor(C.jitter(_tg), _ag), _rby, min_iet=5)[0]),
        logdur=float(np.log((_tg[np.isfinite(_tg)].max() - _tg[np.isfinite(_tg)].min())
                            / (_tr[np.isfinite(_tr)].max() - _tr[np.isfinite(_tr)].min()))),
        degree_mmd=final_mmd(_cfg))
out["slm"] = slm
out["slm_eff"] = json.load(open(os.path.join(os.path.dirname(__file__), "slm_efficiency.json")))

# append the SLM rows to cells_main.csv (same columns/conventions)
slm_rows = [dict(config=_cfg, dataset=_ds, backbone="Qwen3-8B", level=_lvl,
                 degree_mmd=(_c := slm[f"{_ds}:{_lvl}"])["degree_mmd"], ks=_c["ks"], Z_ks=_c["Z_ks"],
                 M_gen=_c["M_dedup"], M_ref=_c["M_ref"], Z_M=_c["Z_M"],
                 rhythm=_c["rhythm_jit"], q_ks=_c["q_ks"],
                 zero_iet=_c["zero_iet"], logdur=_c["logdur"])
            for _cfg, _lvl, _ds in SLM_CELLS]
main = pd.concat([main, pd.DataFrame(slm_rows)], ignore_index=True)
main.to_csv(os.path.join(os.path.dirname(__file__), "cells_main.csv"), index=False)

with open(os.path.join(os.path.dirname(__file__), "verified_stats.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (dict, list))}, indent=0, default=str))
print("DICTS:", {k: out[k] for k in ["M_gen_api","best_struct_med_iet_ratio","boot_beats_all_agents","atten_by_ds","rr_n_pairs"]})
