#!/usr/bin/env python
"""Emit paper/tab_*.tex from verified sources. Deterministic; the gate re-runs
this and diffs against the committed tables."""
import os, re, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "..", "..", "scores", "precheck")
PAP = os.path.join(HERE, "..")
CAP = json.load(open(os.path.join(HERE, "table_captions.json")))
def cap(k): return "\\caption{%s}" % CAP[k]
main = pd.read_csv(os.path.join(HERE, "cells_main.csv"))
st = json.load(open(os.path.join(HERE, "verified_stats.json")))
A = pd.read_csv(os.path.join(PRE, "taskA_dependence_metrics.csv")); A = A[A.status=="ok"]
A = pd.concat([A, pd.read_csv(os.path.join(HERE, "slm_taskA_metrics.csv"))], ignore_index=True)
ded = A[A.arm=="dedup"].set_index("config")
B = pd.read_csv(os.path.join(PRE, "taskB_count_conditional.csv")); B = B[B.status=="ok"]
B = pd.concat([B, pd.read_csv(os.path.join(HERE, "slm_taskB.csv"))], ignore_index=True).set_index("config")
FLAGS = {"weibo_tech_react_v2": "s", "weibo_tech_memory_v1": "s", "weibo_daily_memory_v1": "s", "wikirevision_l1_gpt4omini_api": "s",
         "weibo_tech_l0_gpt4omini_api": "t",
         "wikirevision_l0_qwen3_8b_api": "e", "wikirevision_l1_qwen3_8b_api": "e,s",
         "dianping_l0_qwen3_8b_api": "e", "dianping_l1_qwen3_8b_api": "e,s",
         "weibo_daily_l0_qwen3_8b_api": "e", "weibo_daily_l1_qwen3_8b_api": "e,s",
         "weibo_tech_l0_qwen3_8b_api": "e", "weibo_tech_l1_qwen3_8b_api": "e,s"}
DS = ["dianping", "wikirevision", "weibo_daily", "weibo_tech"]
DSTEX = {"dianping": "dianping", "wikirevision": "wikirevision",
         "weibo_daily": "weibo\\_daily", "weibo_tech": "weibo\\_tech"}

def f3(x): return "--" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.3f}"
def f2s(x): return f"${x:+.2f}$"

# ---- Table 1: main 16-cell grid ----
rows = []
order = []
for ds in DS:
    for bb in ["Qwen3-30B-A3B", "GPT-4o-Mini", "Qwen3-8B"]:
        s = main[(main.dataset==ds) & (main.backbone==bb)].sort_values("level")
        for _, r in s.iterrows():
            order.append((ds, bb, r))
lines = [
"\\begin{table}[t]",
cap("tab_main"),
"\\label{tab:complete-configs}",
"\\centering\\scriptsize",
"\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{0.86}",
"\\begin{tabular}{lllrrrrr}",
"\\toprule",
"Dataset & Backbone & Cell & Deg.\\ MMD & KS & $Z_{\\mathrm{KS}}$ & $M$ ($M^{\\mathrm{real}}$) & $q_{\\mathrm{RR}}$ \\\\",
"\\midrule"]
prev = None
for ds, bb, r in order:
    dstex = DSTEX[ds] if (ds, bb) != prev and ds != (prev[0] if prev else None) else ""
    dcol = DSTEX[ds] if (prev is None or ds != prev[0]) else ""
    bcol = bb
    if prev is not None and ds == prev[0] and bb == prev[1]:
        bcol = ""
    flag = FLAGS.get(r.config, "")
    if r.config == "weibo_tech_l0_gpt4omini_api":
        mmdtxt = f3(r.degree_mmd) + "$^{t}$"
    else:
        mmdtxt = f3(r.degree_mmd) + ("$^{c}$" if (isinstance(r.degree_mmd,float) and np.isnan(r.degree_mmd)) or (r.config=="weibo_tech_memory_v1") else "")
    lines.append(f"{dcol} & {bcol} & {r.level}$^{{{flag}}}$ & {mmdtxt} & {r.ks:.3f} & ${r.Z_ks:+.1f}$ & {f2s(r.M_gen)} ({r.M_ref:.2f}) & {r.q_ks:.3f} \\\\" if flag
                 else f"{dcol} & {bcol} & {r.level} & {mmdtxt} & {r.ks:.3f} & ${r.Z_ks:+.1f}$ & {f2s(r.M_gen)} ({r.M_ref:.2f}) & {r.q_ks:.3f} \\\\")
    prev = (ds, bb)
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(PAP, "tab_main.tex"), "w").write("\n".join(lines) + "\n")

# ---- Table A1: per-cell controls ----
lines = [
"\\begin{table}[h]",
cap("tab_controls"),
"\\label{tab:artifact-controls}",
"\\centering\\footnotesize",
"\\begin{tabular}{lllrrrrrr}",
"\\toprule",
"Dataset & Backbone & Cell & $n$ & Zero IET & $\\log(T_g/T_r)$ & CC ratio & Shape KS & Rhythm \\\\",
"\\midrule"]
prev = None
for ds, bb, r in order:
    dcol = DSTEX[ds] if (prev is None or ds != prev[0]) else ""
    bcol = bb if (prev is None or bb != prev[1] or ds != prev[0]) else ""
    n = int(ded.loc[r.config, "n_generated"])
    cc = B.loc[r.config, "ks_countcond_jit"] / B.loc[r.config, "ks_raw_jit"]
    shp = B.loc[r.config, "ks_shape_jit"]
    lines.append(f"{dcol} & {bcol} & {r.level} & {n} & {r.zero_iet:.2f} & ${r.logdur:+.2f}$ & {cc:.3f} & {f3(shp)} & {r.rhythm:.2f} \\\\")
    prev = (ds, bb)
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(PAP, "tab_controls.tex"), "w").write("\n".join(lines) + "\n")

# ---- Table A2: RR band ----
band = pd.read_csv(os.path.join(PRE, "taskC_rr_band_summary.csv")).set_index("dataset")
lines = [
"\\begin{table}[h]",
cap("tab_rrband"),
"\\label{tab:rr-reference}",
"\\centering\\small",
"\\begin{tabular}{lrrrrr}",
"\\toprule",
"Dataset & Windows & Pairs & KS ($\\mu\\pm\\sigma$) & $|\\Delta M|$ ($\\mu\\pm\\sigma$) & Rhythm ($\\mu\\pm\\sigma$) \\\\",
"\\midrule"]
for ds in DS:
    b = band.loc[ds]
    lines.append(f"{DSTEX[ds]} & {int(b.n_windows)} & {int(b.n_pairs)} & ${b.mu_ks:.3f}\\pm{b.sigma_ks:.3f}$ & ${b.mu_abs_M_gap:.3f}\\pm{b.sigma_abs_M_gap:.3f}$ & ${b.mu_rhythm_median_ks:.2f}\\pm{b.sigma_rhythm_median_ks:.2f}$ \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(PAP, "tab_rrband.tex"), "w").write("\n".join(lines) + "\n")

# ---- Table A3: baselines ----
D = pd.read_csv(os.path.join(PRE, "taskD_timing_floor.csv"))
lines = [
"\\begin{table}[h]",
cap("tab_baselines"),
"\\label{tab:timing-baselines}",
"\\centering\\small",
"\\begin{tabular}{llrrr}",
"\\toprule",
"Dataset & Generator & Inter-event KS & Burst gap & $\\rho_1$ (real $\\rho_1$) \\\\",
"\\midrule"]
namemap = {"bootstrap": "seed-IET bootstrap", "hawkes": "Hawkes (seed MLE)", "poisson": "Poisson (seed rate)",
           "agent_L0": "agent L0", "agent_L1": "agent L1", "agent_L2": "agent L2"}
for ds in DS:
    sub = D[D.dataset==ds]
    first = True
    for blname in ["bootstrap", "hawkes", "poisson", "agent_L0", "agent_L1", "agent_L2"]:
        r = sub[sub.baseline==blname].iloc[0]
        dcol = DSTEX[ds] if first else ""
        first = False
        if r.n_sims > 1:
            lines.append(f"{dcol} & {namemap[blname]} & ${r.ks_mean:.3f}\\pm{r.ks_sd:.3f}$ & ${r.burst_gap_mean:+.3f}\\pm{r.burst_gap_sd:.3f}$ & ${r.rho1_sim_mean:+.3f}$ (${r.ref_rho1:.3f}$) \\\\")
        else:
            cfg = main[(main.dataset == ds) & (main.backbone == "Qwen3-30B-A3B") & (main.level == blname[-2:])].config.iloc[0]
            rg = ("%+.3f" % ded.loc[cfg, "rho1_gen"]).replace("-0.000", "+0.000")
            lines.append(f"{dcol} & {namemap[blname]} & ${r.ks_mean:.3f}$ & ${r.burst_gap_mean:+.3f}$ & ${rg}$ (${ded.loc[cfg, 'rho1_ref']:.3f}$) \\\\")
    lines.append("\\addlinespace[2pt]")
lines = lines[:-1]
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(PAP, "tab_baselines.tex"), "w").write("\n".join(lines) + "\n")

# ---- Table A4: scheduler-null ----
S = pd.read_csv(os.path.join(PRE, "scheduler_null_control.csv")).set_index("dataset")
lines = [
"\\begin{table}[h]",
cap("tab_sched"),
"\\label{tab:scheduler-control}",
"\\centering\\small",
"\\begin{tabular}{lrrrr}",
"\\toprule",
"Dataset & $M^{\\mathrm{real}}$ & $M^{\\mathrm{real}}_{\\mathrm{quant}}$ & Reduction & Generated $M$ (min..max) \\\\",
"\\midrule"]
for ds in DS:
    r = S.loc[ds]
    lines.append(f"{DSTEX[ds]} & {r.M_real:.3f} & {r.M_quantized_real:.3f} & ${100*r.attenuation_frac:+.0f}$\\% & ${r.M_gen_min:+.3f}$\,..\,${r.M_gen_max:+.3f}$ \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(PAP, "tab_sched.tex"), "w").write("\n".join(lines) + "\n")

# ---- Table A5: E-REP window contrast ----
E = pd.read_csv(os.path.join(HERE, "..", "..", "scores", "erep", "erep_stage1_scores.csv"))
Ej = E[E.arm=="jitter"].set_index("level"); Edp = E[E.arm=="dedup"].set_index("level")
inc = main[(main.dataset=="wikirevision") & (main.backbone=="GPT-4o-Mini")].set_index("level")
Binc = B  # taskB indexed by config
ks_inc = {"L0": B.loc["wikirevision_l0_gpt4omini_api","ks_raw_jit"], "L1": B.loc["wikirevision_l1_gpt4omini_api","ks_raw_jit"]}
mmd_fair = {"L0": st["erep_mmd_L0"], "L1": st["erep_mmd_L1"]}
lines = [
"\\begin{table}[h]",
cap("tab_erep"),
"\\label{tab:window-sensitivity}",
"\\centering\\small",
"\\begin{tabular}{lrrrr}",
"\\toprule",
" & \\multicolumn{2}{c}{Representative window} & \\multicolumn{2}{c}{Cold-start window} \\\\",
"\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
"Metric & L0 & L1 & L0 & L1 \\\\",
"\\midrule"]
lines.append(f"Inter-event KS (jitter) & {Ej.loc['L0','ks']:.3f} & {Ej.loc['L1','ks']:.3f} & {ks_inc['L0']:.3f} & {ks_inc['L1']:.3f} \\\\")
rep_md = open(os.path.join(HERE, "..", "src", "erep", "EREP_STAGE1_REPORT.md")).read() if os.path.exists(os.path.join(HERE, "..", "src", "erep", "EREP_STAGE1_REPORT.md")) else open(os.path.join(HERE, "..", "..", "docs", "EREP_STAGE1_REPORT.md")).read()
_m = re.search(r"log duration ratio\s+([\d.]+) / ([\d.]+)", rep_md); assert _m, "log duration ratio not found in E-REP report"
lines.append(f"$\\log(T_g/T_r)$ & {float(_m.group(1)):.2f} & {float(_m.group(2)):.2f} & {B.loc['wikirevision_l0_gpt4omini_api','log_duration_ratio']:.2f} & {B.loc['wikirevision_l1_gpt4omini_api','log_duration_ratio']:.2f} \\\\")
lines.append(f"Degree MMD (final) & {mmd_fair['L0']:.3f} & {mmd_fair['L1']:.3f} & {inc.loc['L0','degree_mmd']:.3f} & {inc.loc['L1','degree_mmd']:.3f} \\\\")
lines.append(f"$M$ (dedup) & ${Edp.loc['L0','M_gen']:+.2f}$ & ${Edp.loc['L1','M_gen']:+.2f}$ & ${inc.loc['L0','M_gen']:+.2f}$ & ${inc.loc['L1','M_gen']:+.2f}$ \\\\")
lines.append(f"Zero IET fraction & {E[(E.arm=='as_generated')&(E.level=='L0')].zero_iet_frac.iloc[0]:.2f} & {E[(E.arm=='as_generated')&(E.level=='L1')].zero_iet_frac.iloc[0]:.2f} & {main[(main.backbone=='GPT-4o-Mini')&(main.dataset=='wikirevision')&(main.level=='L0')].zero_iet.iloc[0]:.2f} & {main[(main.backbone=='GPT-4o-Mini')&(main.dataset=='wikirevision')&(main.level=='L1')].zero_iet.iloc[0]:.2f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(PAP, "tab_erep.tex"), "w").write("\n".join(lines) + "\n")
print("tables written")


# ---- matched-horizon table (from analysis/matched_horizon_8b.csv) ----
mh = pd.read_csv(os.path.join(HERE, "matched_horizon_8b.csv"))
mh = mh[mh.status == "ok"]
lines = [r"\begin{table}[t]", r"\centering", r"\small", cap("tab_matched_horizon"), r"\label{tab:matched-horizon}",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Dataset & Arm & $n$ & KS & $Z_{\mathrm{KS}}$ & $M_{\mathrm{gen}}$ & $M_{\mathrm{ref}}$ & $Z_{|\Delta M|}$ \\", r"\midrule"]
for ds in ["wikirevision", "dianping", "weibo_daily", "weibo_tech"]:
    g = mh[mh.dataset == ds]
    for _, r in g.iterrows():
        lines.append(r"%s & %s & %d & %.3f & $%+.1f$ & $%+.3f$ & %.3f & $%+.1f$ \\" % (
            ds.replace("_", r"\_"), r.arm, r.n, r.ks, r.Z_ks, r.M_gen, r.M_ref, r.Z_abs_M_gap))
    lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(PAP, "tab_matched_horizon.tex"), "w").write("\n".join(lines) + "\n")

# ---- synthetic validation table (from analysis/synthetic_invariance_check.csv) ----
sy = pd.read_csv(os.path.join(HERE, "synthetic_invariance_check.csv"))
NAMES = {"faithful replicate": "Faithful replicate", "intra-actor shuffle": "Actor-level shuffle",
         "renewal, same marginal": "Independent renewal with same marginal", "rate x20": "Rate multiplied by 20",
         "rate x20 + shuffle": "Rate multiplied by 20 with shuffle"}
lines = [r"\begin{table}[t]", r"\centering", cap("tab_synthetic"), r"\label{tab:synthetic-validation}", r"\small",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Transformation & IET KS & $W_1(\log\tau)$ & $|\Delta B|$ & $M$ \\", r"\midrule"]
for _, r in sy.iterrows():
    lines.append(r"%s & %.3f & %.3f & %.3f & %.3f \\" % (NAMES[r["transform"]], r.IET_KS, r.logW1, r["|dB|"], r.M_gen))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(PAP, "tab_synthetic.tex"), "w").write("\n".join(lines) + "\n")
print("tables written to", os.path.abspath(PAP))
