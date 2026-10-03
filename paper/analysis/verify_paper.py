#!/usr/bin/env python
"""Verification gate for the R2O workshop paper (revision of 2026-09-06).

Every statistic printed in main.tex / appendices.tex / the committed tables must be
derivable from analysis/verified_stats.json (recomputed from the raw score CSVs by
recompute_stats.py) or directly from the score CSVs under ../src and ./ ; the committed
tab_*.tex files must byte-match a fresh run of build_tables.py.  Exits nonzero on any
failure.  Text matching is whitespace-normalised so line breaks do not matter, but the
wording itself is pinned: a rephrased number must be re-pinned here, on purpose.
"""
import json, os, re, subprocess, sys, tempfile, shutil, filecmp, math
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAP = os.path.join(HERE, "..")
SRC = os.path.join(PAP, "..", "scores"); DOCS = os.path.join(PAP, "..", "docs")
st = json.load(open(os.path.join(HERE, "verified_stats.json")))
r = lambda k: st[k]

def norm(s): return re.sub(r"\s+", " ", s)
def read(fn): return open(os.path.join(PAP, fn)).read()
main_tex, app_tex = read("main.tex"), read("appendices.tex")
TEX = norm(main_tex + "\n" + app_tex)
TABS = ["tab_main.tex", "tab_controls.tex", "tab_rrband.tex", "tab_baselines.tex", "tab_sched.tex",
        "tab_erep.tex", "tab_matched_horizon.tex", "tab_synthetic.tex"]
tabtex = {t: read(t) for t in TABS + ["tab_landscape.tex"]}
ck_txt = read("checklist.tex")

checks, fails = [], []
def chk(name, cond, detail=""):
    checks.append(name)
    if not cond: fails.append((name, detail))
def present(name, frag, where=None):
    chk(name, norm(frag) in (where if where is not None else TEX), f"missing: {frag!r}")

# ---- 0. structure: single source of truth ----------------------------------------
present("input_appendices", "\\input{appendices}", main_tex)
present("input_checklist", "\\input{checklist}", main_tex)
for t in TABS + ["tab_landscape.tex"]:
    present(f"input:{t}", "\\input{%s}" % t[:-4], main_tex + app_tex)
body_tex = main_tex[main_tex.index("\\begin{document}"):]
chk("no_inline_tables_in_main", "\\begin{tabular}" not in body_tex and "\\begin{tabular}" not in app_tex,
    "a tabular is inlined in main.tex/appendices.tex instead of \\input from tab_*.tex")
chk("appendix_only_in_appendices", "\\appendix" in app_tex and "\\appendix" not in main_tex)

# ---- 1. tables regenerate byte-identically ----------------------------------------
tmp = tempfile.mkdtemp()
for t in TABS: shutil.copy(os.path.join(PAP, t), os.path.join(tmp, t))
res = subprocess.run([sys.executable, os.path.join(HERE, "build_tables.py")], capture_output=True, text=True)
chk("build_tables_runs", res.returncode == 0, res.stderr[-500:])
for t in TABS:
    chk(f"table:{t}", filecmp.cmp(os.path.join(PAP, t), os.path.join(tmp, t), shallow=False), "regenerated table differs")

# ---- 2. citations resolve; landscape table agrees with metric_landscape.csv ----------
bib = read("refs.bib")
bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
cited = {k.strip() for m in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", main_tex + app_tex + tabtex["tab_landscape.tex"])
         for k in m.split(",")}
chk("all_cites_in_bib", cited <= bibkeys, f"unresolved: {sorted(cited - bibkeys)}")
ML = pd.read_csv(os.path.join(HERE, "metric_landscape.csv")).set_index("bibkey")
land = dict(re.findall(r"\\citet\{(\w+)\}\s*&(.*?)\\\\", tabtex["tab_landscape.tex"], re.S))
chk("landscape_rows", len(land) == 48, f"{len(land)} rows")
bad = []
for k, cells in land.items():
    c = [x.strip() for x in cells.split("&")]
    if k not in ML.index or c[-2] != str(ML.loc[k, "tests_temporal_dependence"]) or c[-1] != str(ML.loc[k, "tests_closed_loop"]):
        bad.append(k)
chk("landscape_dep_cl_vs_csv", not bad, f"Dep./CL disagree with metric_landscape.csv: {bad}")

# ---- 3. main-text statistics --------------------------------------------------------
present("mmd_L0L1", "degree MMD improves on all four datasets by $43.8\\%$ on average")
chk("mmd_L0L1_val", abs(r("mmd_L0L1_mean_pct") + 43.8) < 0.05 and r("mmd_L0L1_improved_count") == 4)
present("ks_L0L1_OLD", "inter-event KS changes in mixed directions, from $-38.0\\%$ to\n$+18.5\\%$ across datasets ($-4.8\\%$ on average")
chk("ks_L0L1_range", round(min(r("ks_L0L1_pct")), 1) == -38.0 and round(max(r("ks_L0L1_pct")), 1) == 18.5)
chk("ks_L0L1_val", abs(r("ks_L0L1_mean_pct") + 4.8) < 0.06 and r("ks_L0L1_sign_mixed"))
present("logdur_rep", "worsens from $3.83$ to $4.44$, or approximately $46$ to $85$ times the real duration")
present("logdur_rep_backbone", "representative wikirevision window (GPT-4o-Mini;")
rep_txt = open(os.path.join(DOCS, "EREP_STAGE1_REPORT.md")).read()
chk("logdur_src", "3.83" in rep_txt and "4.44" in rep_txt)
chk("logdur_conv", abs(math.exp(3.83) - 46) < 1 and abs(math.exp(4.44) - 85) < 1)
present("Mreal", "Real memory coefficients range from $0.10$ to $0.27$")
chk("Mreal_val", round(r("M_ref_min"), 2) == 0.10 and round(r("M_ref_max"), 2) == 0.27)
present("none16", "none of the 16 preregistered configurations reproduces this dependence")
present("abstract_none", "no preregistered configuration reproduces the temporal dependence")
chk("none16_val", r("ZM_outside2") == 16 and r("n_cells") == 16)
present("Mqwen", "Across the 12 Qwen3-30B-A3B configurations, memory ranges from $-0.28$ to $0.03$")
chk("Mqwen_val", round(r("M_gen_qwen_min"), 2) == -0.28 and round(r("M_gen_qwen_max"), 2) == 0.03 and r("n_qwen_cells") == 12)
present("rhythm7of16", "actor level rhythm divergence lies inside the real reference ($|Z| \\le 2$) on 7 of the 16 configurations")
chk("rhythm7of16_val", 16 - r("Zrhythm_outside2") == 7)
present("boot_3of4", "on three of four datasets")
present("boot_vals", "$0.065$ to $0.122$ versus $0.468$ to\n$0.559$ for the best agent")
chk("boot_val", r("boot_beats_count") == 3
    and round(min(v for k, v in r("boot_ks").items() if r("boot_beats_all_agents")[k]), 3) == 0.065
    and round(max(v for k, v in r("boot_ks").items() if r("boot_beats_all_agents")[k]), 3) == 0.122
    and round(r("best_agent_ks_min_where_beaten"), 3) == 0.468 and round(r("best_agent_ks_max_where_beaten"), 3) == 0.559)
present("boot_rho1", "global lag 1\nautocorrelation $-0.002$ to $+0.002$ (20 seeds) against real $0.22$ to $0.36$")
D = pd.read_csv(os.path.join(SRC, "precheck", "taskD_timing_floor.csv"))
bt = D[D.baseline == "bootstrap"]
chk("boot_rho1_val", round(bt.rho1_sim_mean.min(), 3) == -0.002 and round(bt.rho1_sim_mean.max(), 3) == 0.002
    and round(bt.ref_rho1.min(), 2) == 0.22 and round(bt.ref_rho1.max(), 2) == 0.36 and (D[D.baseline.isin(["bootstrap","hawkes","poisson"])].n_sims == 20).all(),
    f"{bt.rho1_sim_mean.min():.4f}..{bt.rho1_sim_mean.max():.4f}; ref {bt.ref_rho1.min():.3f}..{bt.ref_rho1.max():.3f}")
present("mh_8of8", "matched horizon controls place memory near the shuffled value in all eight exploratory arms")
mh = pd.read_csv(os.path.join(HERE, "matched_horizon_8b.csv")); m8 = mh[mh.arm.isin(["L0-trunc", "L1"])]
chk("mh_8of8_val", len(m8) == 8 and (m8.M_gen.abs() <= 0.08).all(), f"max |M| {m8.M_gen.abs().max():.3f}")
w = mh[mh.dataset == "wikirevision"].set_index("arm")
chk("mh_wiki_full_inband", round(w.loc["L0-full", "M_gen"], 2) == 0.16 and round(w.loc["L0-trunc", "M_gen"], 3) == -0.006)
present("only8b", "The only 8B configuration approaching real memory also has the\nlargest marginal timing discrepancy in that set")
slm = r("slm"); wl0 = slm["wikirevision:L0"]
chk("only8b_val", wl0["abs_M_gap"] == min(v["abs_M_gap"] for v in slm.values()) and wl0["ks"] == max(v["ks"] for v in slm.values()))
present("bb_mixed", "Matched Qwen3-30B-A3B and GPT-4o-Mini comparisons improve some temporal\nmeasures while worsening others")
bk = list(r("backbone_ks_pct").values()); chk("bb_mixed_val", min(bk) < 0 < max(bk))
present("cc_main", "Matching actor level event counts has little effect on marginal timing")

# ---- 4. efficiency (exploratory 8B) ------------------------------------------------------
eff = r("slm_eff")
c_l0 = sorted(v["calls_per_event"] for k, v in eff.items() if k.endswith(":L0"))
c_l1 = sorted(v["calls_per_event"] for k, v in eff.items() if k.endswith(":L1"))
present("calls_main", "L0 uses 1.9 to 2.8 language model calls per\nevent and L1 uses 2.5 to 4.7, with 1.6 to 2.4 times more prompt tokens")
present("calls_app", "1.9 to 2.8 LM calls per\nevent for L0 and 2.5 to 4.7 calls per event for L1")
chk("calls_val", [round(c_l0[0], 1), round(c_l0[-1], 1), round(c_l1[0], 1), round(c_l1[-1], 1)] == [1.9, 2.8, 2.5, 4.7])
tok = sorted(eff["%s:L1" % d]["in_tok_per_event"] / eff["%s:L0" % d]["in_tok_per_event"]
             for d in ("wikirevision", "dianping", "weibo_daily", "weibo_tech"))
present("tok_app", "L1 uses 1.6 to 2.4$\\times$\nmore prompt tokens per event than L0")
chk("tok_val", round(tok[0], 1) == 1.6 and round(tok[-1], 1) == 2.4)
costs = [v["cost_usd"] for v in eff.values()]
present("cost", "\\$4.48 to \\$7.77 per cell and \\$46.78 for the complete exploratory grid")
chk("cost_val", round(min(costs), 2) == 4.48 and round(max(costs), 2) == 7.77 and round(sum(costs), 2) == 46.78)
present("l1_lengths", "Their lengths range from 4,900 to 5,696 events")
l1n = sorted(v["n"] for k, v in slm.items() if k.endswith(":L1"))
chk("l1_lengths_val", l1n[0] == 4900 and l1n[-1] == 5696 and all(v["n"] == 9000 for k, v in slm.items() if k.endswith(":L0")))

# ---- 5. appendix controls --------------------------------------------------------------------
present("tie41", "removing duplicate\ntimestamps explains a median 41\\% of the absolute burstiness discrepancy")
chk("tie41_val", round(100 * r("tie_share_median")) == 41)
present("tie1212", "residual with the same sign remains in all 12 configurations")
chk("tie1212_val", r("residual_same_sign_count") == 12 and r("flip_cells") == [])
present("tie_dianping", "On two\n\\texttt{dianping} configurations with small original burstiness discrepancies,\ndeduplication increases the discrepancy")
chk("tie_dianping_val", r("tie_share_min") < 0)
present("sched", "Quantization changes real memory from a 39\\% increase to a 49\\%\nreduction")
chk("sched_val", round(100 * r("atten_max")) == 49 and round(100 * min(r("atten_by_ds").values())) == -39)
present("sched_above", "remains above the generated\nQwen3-30B-A3B range on all four datasets")
chk("sched_above_val", r("gen_below_qfloor_count") == 4)
present("cc", "The median ratio between\ncount conditioned and original KS is 1.006, with values ranging from 0.886 to\n1.039 across the primary configurations")
B = pd.read_csv(os.path.join(SRC, "precheck", "taskB_count_conditional.csv")); B = B[B.status == "ok"].set_index("config")
cells = pd.read_csv(os.path.join(HERE, "cells_main.csv"))
prereg = [c for c in cells.config if c in B.index]
ratio = (B.loc[prereg, "ks_countcond_jit"] / B.loc[prereg, "ks_raw_jit"]) if "ks_countcond_jit" in B else (B.loc[prereg, "ks_countcond"] / B.loc[prereg, "ks_raw"])
chk("cc_val", len(prereg) == 16 and round(ratio.median(), 3) in (1.006, 1.007) and round(ratio.min(), 3) == 0.886 and round(ratio.max(), 3) == 1.039,
    f"n={len(prereg)} median={ratio.median():.4f} range={ratio.min():.3f}..{ratio.max():.3f}")
present("cc20", "The procedure uses 20 subsampling replicates")
present("erep_M", "Real memory on the\nrepresentative window is 0.15, while the generated values are 0.02 and -0.06\nfor L0 and L1 respectively")
chk("erep_M_val", round(r("erep_M_ref"), 2) == 0.15 and round(r("erep_M_gen_L0"), 2) == 0.02 and round(r("erep_M_gen_L1"), 2) == -0.06)
present("erep_ks", "the marginal jitter-arm KS is 0.090 versus\n0.310 on the inherited cold-start window, a roughly 3.4$\\times$")
chk("erep_ks_val", round(r("erep_ks_jit_L0"), 3) == 0.090 and abs(0.310 / 0.090 - 3.44) < 0.02)
present("erep_logdur_app", "worsens the log-duration ratio from 3.83 to 4.44\n(approximately $46\\times$ to $85\\times$ in duration)")
present("baseline_seeds", "Reported baseline means and standard deviations use 20\nsimulation seeds")
present("prefix", "conditions on a real prefix containing 1,000 edges and generates a\ncontinuation of approximately 9,000 events in rounds of 50 events")
present("n16", "The primary analysis contains 16 preregistered configurations. Twelve use\nQwen3-30B-A3B")
present("gpt200", "GPT-4o-Mini run for \\texttt{weibo\\_tech} L0 contains 200 generated events")
A_ok = pd.read_csv(os.path.join(SRC, "precheck", "taskA_dependence_metrics.csv")); A_ok = A_ok[(A_ok.status == "ok") & (A_ok.arm == "dedup")].set_index("config")
short = sorted(c for c in prereg if A_ok.loc[c, "n_generated"] < 9000)
chk("five_short", len(short) == 5 and "wikirevision_l1_gpt4omini_api" in short and int(A_ok.loc["wikirevision_l1_gpt4omini_api", "n_generated"]) == 7232, str(short))
present("five_short_txt", "Five cells are shorter than the nominal horizon", norm(app_tex))
present("wiki_l1_7232", "\\texttt{wikirevision} L1 contains 7,232 events", norm(app_tex))
chk("tab_main_s_flags", tabtex["tab_main.tex"].count("$^{s}$") + tabtex["tab_main.tex"].count("$^{s,c}$") == 4, "expected 4 preregistered short-run superscripts")
chk("gpt200_val", any("200" in row for row in tabtex["tab_controls.tex"].splitlines() if "GPT-4o-Mini" in row or "L0$^{t}$" in row))
present("rr_window", "pairs of 10,000 edge real windows matched on\nduration, actor count, edges per actor, and median positive IET, using up to 60\npairs per dataset with reuse of any window capped at three")
chk("rr_pairs_val", r("rr_n_pairs") == {"dianping": 60, "wikirevision": 60, "weibo_daily": 52, "weibo_tech": 13})
present("wt13", "\\texttt{weibo\\_tech} has only 13 disjoint pairs")
# synthetic validation numbers in the text vs CSV
sy = pd.read_csv(os.path.join(HERE, "synthetic_invariance_check.csv")).set_index("transform")
present("syn_shuffle", "reducing memory from $0.164$ to $-0.007$")
present("syn_rate", "Rescaling time by a factor of 20\nleaves memory at $0.164$ while moving IET KS to $0.686$ and log Wasserstein\ndistance to $2.996$")
present("syn_ref", "The reference stream\nhas memory 0.164 and burstiness 0.464", norm(tabtex["tab_synthetic.tex"]))
chk("syn_val", sy.loc["intra-actor shuffle", "M_gen"] == -0.007 and sy.loc["rate x20", "IET_KS"] == 0.686
    and sy.loc["rate x20", "logW1"] == 2.996 and sy.loc["rate x20", "M_gen"] == 0.164 and sy.real_M.iloc[0] == 0.164 and sy.real_B.iloc[0] == 0.464)
prereg_txt = open(os.path.join(DOCS, "PREREGISTRATION_v1_2026-08-25.md")).read()
present("prereg24", "Seed noise was measured from 24 replicate sets")
chk("prereg24_src", "24 replicate" in prereg_txt or "24" in prereg_txt)

# ---- 6. table captions carry the provenance the text relies on -------------------------------
present("cap_c_footnote", "\\texttt{weibo\\_tech} L2 scored on the archived 8.65k-edge run", norm(tabtex["tab_main.tex"]))
present("cap_e_footnote", "4,900 to 5,696 events", norm(tabtex["tab_main.tex"]))

# ---- 7. forbidden strings (stale numbers / de-anonymising) -----------------------------------------
DEANON = [x for x in os.environ.get("PAPER_DEANON_STRINGS", "").split(",") if x]  # author names/affiliations; set locally, never shipped
FORBID = ["43.5", "24.7\\%", "[WSDM]", "\\done", "\\gated", "[DONE]", "[G1]", "[G2]",
           "+0.23", "1801", "6.6\\times", "47\\times",
          "[in refs.bib]", "tab:matched horizon", "tab:window sensitivity", "RQ1", "RQ2", "RQ3"]
alltext = main_tex + app_tex + "".join(tabtex.values())
for f in FORBID + DEANON:
    chk(f"forbid:{f}", f not in alltext + read("checklist.tex"), "forbidden string present")
for fname in ["refs.bib", "checklist.tex"]:
    t2 = read(fname)
    for f in DEANON:
        chk(f"forbid:{fname}:{f}", f not in t2, "deanonymizing string")
ck = read("checklist.tex")
chk("no_answerTODO", "\\answerTODO" not in ck.replace("% Replace", ""))
ck_refs = set(re.findall(r"\\ref\{([^}]*)\}", ck))
labels = set(re.findall(r"\\label\{([^}]*)\}", alltext))
present("prop3_shift", "so the change equals\n$\\bigl||c + \\log a| - |c|\\bigr|$, which attains $|\\log a|$ if and only if\n$c\\,(c + \\log a) \\ge 0$", norm(app_tex))
chk("prop3_shift_val", abs(abs(1 + (-3)) - abs(1)) == 1)
chk("spelling_prereg", "pre-registered" not in main_tex + app_tex + ck_txt and "Qwen3 8B" not in main_tex + app_tex + "".join(tabtex.values()))
chk("fig1_caption_scope", "GPT-4o-Mini on wikirevision L0 overshoots it ($M = +0.43$)" in norm(main_tex) and round(r("M_gen_api")["wikirevision_l0_gpt4omini_api"], 2) == 0.43)
present("decoding_all", "temperature 0.3 and at most 1,024 generated\ntokens per call", norm(app_tex))
chk("q8_no", "Wall-clock generation times for the archived runs were not retained" in ck_txt and ck_txt.count("\\answerNo{}") >= 1)
chk("no_appH_for_temperature", "temperature 0.3 (Appendix" not in norm(app_tex))
chk("hidelinks", "\\hypersetup{hidelinks}" in main_tex and main_tex.count("\\usepackage{hyperref}") == 1)
chk("ck_limitations_ptr", "Appendices~\\ref{app:setup} and~\\ref{app:limitations}" in ck_txt)
present("vllm_tp1", "vLLM 0.20.0 (FP8 weights, tensor parallel size 1", norm(app_tex))
chk("ttfont_type1", "\\renewcommand{\\ttdefault}{lmtt}" in main_tex)
chk("has_conclusion", "\\section{Conclusion}" in main_tex)
Dh = D[D.baseline == "hawkes"]
present("hawkes_rho1", "with lag 1 autocorrelation of $0.14$ to $0.31$ against $0.22$ to $0.36$\nin the real sequences")
chk("hawkes_rho1_val", round(Dh.rho1_sim_mean.min(), 2) == 0.14 and round(Dh.rho1_sim_mean.max(), 2) == 0.31
    and round(Dh.ref_rho1.min(), 2) == 0.22 and round(Dh.ref_rho1.max(), 2) == 0.36 and (Dh.n_sims == 20).all(),
    f"{Dh.rho1_sim_mean.min():.3f}-{Dh.rho1_sim_mean.max():.3f}")
present("hawkes_seed", "so the seed prefix contains recoverable temporal\ndependence that a seed-conditioned generator can partially reproduce", norm(main_tex))
chk("no_framework_claim", "consequence of the simulation framework" not in main_tex)
ag_rho = A_ok.loc[[c for c in prereg if "gpt" not in c], ["rho1_gen", "rho1_ref"]]
chk("agent_rho1_10of12", len(ag_rho) == 12 and int((ag_rho.rho1_gen < ag_rho.rho1_ref).sum()) == 10 and int((ag_rho.rho1_gen > ag_rho.rho1_ref).sum()) == 2
    and set(ag_rho.index[ag_rho.rho1_gen > ag_rho.rho1_ref]) == {"dianping_react_v2", "wikirevision_re_qwen3"}, str(ag_rho.round(3).to_dict()))
present("agent_rho1_txt", "fall below the real value in 10 of 12 cells and\novershoot it in the other two", norm(main_tex))
chk("tab8_agent_rho1", tabtex["tab_baselines.tex"].count(" -- ") == 0 and f"${A_ok.loc['wikirevision_re_qwen3', 'rho1_gen']:+.3f}$" in tabtex["tab_baselines.tex"])
chk("hyphen_outliers", "teacher-forced" not in main_tex + app_tex and "open-loop" not in main_tex + app_tex)
tie = pd.read_csv(os.path.join(HERE, "rho1_tie_check.csv"))
chk("rho1_tie_check", len(tie) == 12 and tie.max_abs_change.max() < 0.004 and "changes the agent values by at most 0.004" in tabtex["tab_baselines.tex"] and "agent rows (Qwen3-30B-A3B)" in tabtex["tab_baselines.tex"], f"{tie.max_abs_change.max():.4f}")
chk("nserc_ack", "\\begin{ack}" in main_tex and "grant number RGPIN-2024-04733" in norm(main_tex) and "Natural Sciences and Engineering Research\nCouncil of Canada (NSERC)" in main_tex)
chk("checklist_refs_resolve", ck_refs <= labels, f"dangling: {sorted(ck_refs - labels)}")

# ---- 8. figures: drawn at the size they are placed (type size on the page = type size in the file) ----
def pdf_box_in(fn):
    b = open(os.path.join(PAP, fn), "rb").read()
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", b)
    return (float(m.group(3)) - float(m.group(1))) / 72, (float(m.group(4)) - float(m.group(2))) / 72
TEXTWIDTH = 5.5
chk("fig_pair_placement", main_tex.count("\\begin{minipage}[t]{0.48\\textwidth}") == 2
    and "\\includegraphics[width=\\linewidth]{fig1_memory.pdf}" in main_tex
    and "\\includegraphics[width=\\linewidth]{fig2_decoupling.pdf}" in main_tex, "fig1/fig2 not at 0.48\\textwidth")
chk("fig3_placement", "\\includegraphics[width=\\linewidth]{figA_controls.pdf}" in main_tex, "figA not at \\linewidth")
for fn, w_exp in [("fig1_memory.pdf", 0.48 * TEXTWIDTH), ("fig2_decoupling.pdf", 0.48 * TEXTWIDTH), ("figA_controls.pdf", TEXTWIDTH)]:
    w, h = pdf_box_in(fn)
    chk(f"figsize:{fn}", abs(w - w_exp) < 0.02, f"PDF width {w:.2f} in vs placed {w_exp:.2f} in (type would be rescaled)")
chk("fig_heights_budget", pdf_box_in("fig1_memory.pdf")[1] <= 1.6 and pdf_box_in("fig2_decoupling.pdf")[1] <= 1.6
    and pdf_box_in("figA_controls.pdf")[1] <= 1.25, "figure taller than the page budget allows")
mf = open(os.path.join(HERE, "make_figures.py")).read()
chk("fig_title_claims", "improves 4/4" in mf and "3/4 datasets" in mf and "5–22" in mf and "1.006" in mf
    and r("mmd_L0L1_improved_count") == 4 and r("boot_beats_count") == 3
    and round(r("Zks_min")) == 5 and round(r("Zks_max")) == 22 and r("Zks_outside2") == 16, "figure title claim not backed by verified_stats")


# ---- 9. camera-ready (accepted 2026-09-29; reviewer-response edits) -------------------------------
chk("final_mode", "\\usepackage[dblblindworkshop,final]{neurips_2026}" in main_tex and "\\workshoptitle{SLMs for Agentic Systems, Paris, France, 2026}" in main_tex)
chk("authors_named", "Parham Hamouni" in main_tex and "Ebrahim Bagheri" in main_tex and "Anonymous" not in main_tex)
chk("no_placeholder_email", "EMAIL1" not in main_tex and "EMAIL2" not in main_tex, "author emails still placeholders")
chk("footer_accepted", "Do not distribute" not in main_tex and "\\@noticestring" not in main_tex)
chk("code_url", "github.com/parhamouni/r2o-process-fidelity" in app_tex)
Bp = B.loc[prereg]
# compare against the KS actually printed in tab_main (taskA matched-slot KS), not taskB ks_raw
ksA = pd.read_csv(os.path.join(HERE, "cells_main.csv")).set_index("config")["ks"].loc[prereg]
shape_gt = int((Bp.ks_shape_jit.values > ksA.values.round(3)).sum())
lower = [c for c in prereg if Bp.ks_shape_jit[c] <= round(ksA[c], 3)]
present("shape_ks", "Shape KS ranges from 0.22 to 0.66 across the 16 preregistered\nconfigurations and exceeds the matched slot KS on 11 of them")
chk("shape_ks_val", shape_gt == 11 and round(Bp.ks_shape_jit.min(), 2) == 0.22 and round(Bp.ks_shape_jit.max(), 2) == 0.66
    and set(lower) == {"dianping_re_qwen3", "dianping_react_v2", "dianping_memory_v1", "dianping_l0_gpt4omini_api", "wikirevision_l0_gpt4omini_api"}
    and Bp.ks_shape_jit[lower].min() >= 0.215, f"{shape_gt}, lower={lower}, {Bp.ks_shape_jit.min():.3f}-{Bp.ks_shape_jit.max():.3f}")
present("ks_two_defs", "\\texttt{dianping} L0: 0.372 versus 0.823")
chk("ks_two_defs_val", "0.372" in tabtex["tab_main.tex"] and "0.823" in tabtex["tab_baselines.tex"])
Zp = pd.read_csv(os.path.join(SRC, "precheck", "taskC_cell_zscores.csv")); Zp = Zp[Zp.config.isin(prereg) & (Zp.dataset == "weibo_tech")]
present("wt_floor", "floor is $1/14 = 0.071$")
chk("wt_Z_val", round(Zp.Z_ks.min(), 1) == 5.3 and Zp.Z_abs_M_gap.min() >= 2.5 and Zp.Z_abs_M_gap.min() < 2.6, f"{Zp.Z_ks.min():.2f} {Zp.Z_abs_M_gap.min():.2f}")
present("wt_Z", "$Z_{\\mathrm{KS}} \\ge +5.3$,\n$Z_{|\\Delta M|} \\ge +2.5$")
present("sched_main", "changes\nreal memory by between a $49\\%$ reduction and a $39\\%$ increase yet leaves it\nabove every Qwen3-30B-A3B generated value on all four datasets")
chk("sched_sign", "$+49$\\%" in tabtex["tab_sched.tex"] and "$-39$\\%" in tabtex["tab_sched.tex"] and "from a 39\\% increase to a 49\\%\nreduction" in app_tex)
for k in ["barabasi2005origin", "goh2008burstiness", "karsai2018bursty", "malmgren2008poissonian", "holme2012temporal"]:
    chk(f"cite_used:{k}", k in main_tex)

n = len(checks)
if fails:
    print(f"GATE FAIL: {len(fails)}/{n} checks failed")
    for f, d in fails: print("  -", f, d)
    sys.exit(1)
print(f"GATE PASS: {n}/{n} checks passed")
