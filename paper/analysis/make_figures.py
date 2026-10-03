#!/usr/bin/env python
"""Regenerate the three paper figures from cells_main.csv + the precheck CSVs.

Figures are drawn at their PRINTED size (NeurIPS text width 5.5 in; the two
side-by-side figures are placed at 0.48\\textwidth = 2.64 in, the controls
figure at full width), so the type is 6-7.5 pt on the page. Do not place them
at a different width without changing FIG_W / FIG3_W here.

Uses the figure-style helpers (apply_figure_style, set_frame, panel_letter)
when loaded; falls back to plain matplotlib otherwise."""
import os
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "..", "..", "scores", "precheck")
PAP = os.path.join(HERE, "..")
FIG_W, FIG_H = 2.64, 1.57      # 0.48\textwidth; height matches the previous printed height
FIG3_W, FIG3_H = 5.5, 1.22     # \linewidth
BASE, SMALL, TICK = 7.5, 6.5, 6.0
try:
    apply_figure_style(sizes=(BASE, SMALL, TICK))
except NameError:
    mpl.rcParams.update({"font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
                         "legend.fontsize": SMALL, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
                         "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.01})
    def set_frame(ax, style="open"): [ax.spines[s].set_visible(False) for s in ("top", "right")]
    def panel_letter(ax, l, dx=-0.18, dy=1.02, **k): ax.text(dx, dy, l, transform=ax.transAxes, fontweight="bold", fontsize=BASE + 1.5)
mpl.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0.0, "axes.linewidth": 0.6,   # fixed canvas: what you size is what prints
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})

main = pd.read_csv(os.path.join(HERE, "cells_main.csv"))
sched = pd.read_csv(os.path.join(PRE, "scheduler_null_control.csv")).set_index("dataset")
DS = ["dianping", "wikirevision", "weibo_daily", "weibo_tech"]
DSL = ["dianping", "wikirev.", "weibo_d.", "weibo_t."]
lvls = ["L0", "L1", "L2"]
qw = main[main.backbone == "Qwen3-30B-A3B"].set_index(["dataset", "level"])
api = main[main.backbone == "GPT-4o-Mini"].set_index(["dataset", "level"])
slm = main[main.backbone == "Qwen3-8B"].set_index(["dataset", "level"])
QWEN, API, SLM = "#2b6cb0", "#dd6b20", "#38a169"          # backbone hues (threaded across figs 1-2)
REALB, QLINE, GREY = "#d9d9d9", "#555555", "#888888"

# ======================= Figure 1: memory M, one panel =======================
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
GAP = 4.0
for i, ds in enumerate(DS):
    x0 = i * GAP; xs = x0 + np.arange(3)
    sub = main[main.dataset == ds]
    lo, hi = sub.M_ref.min(), sub.M_ref.max()
    bh = max(hi - lo, 0.012); mid = (lo + hi) / 2
    ax.fill_between([x0 - 0.55, x0 + 2.55], mid - bh / 2, mid + bh / 2, color=REALB, lw=0, zorder=0)
    ax.plot([x0 - 0.55, x0 + 2.55], [sched.loc[ds, "M_quantized_real"]] * 2, ls="--", lw=0.8, color=QLINE, zorder=1)
    ax.plot(xs, [qw.loc[(ds, l), "M_gen"] for l in lvls], "-o", color=QWEN, ms=3.2, lw=1.2, zorder=3, mec="white", mew=0.4)
    for l, xi in zip(lvls, xs):
        if (ds, l) in api.index:
            ax.plot(xi, api.loc[(ds, l), "M_gen"], "D", color=API, ms=3.4, zorder=4, mec="white", mew=0.4)
        if (ds, l) in slm.index:
            ax.plot(xi, slm.loc[(ds, l), "M_gen"], "^", color=SLM, ms=3.8, zorder=5, mec="white", mew=0.4)
    ax.text(x0 + 1, -0.49, ds, ha="center", va="top", fontsize=TICK, clip_on=False)
ax.axhline(0, lw=0.5, color="#bbbbbb", zorder=0)
ax.set_xticks([i * GAP + k for i in range(4) for k in range(3)])
ax.set_xticklabels(["L0", "L1", "L2"] * 4)
ax.tick_params(axis="x", length=0, pad=1.5)
ax.set_xlim(-0.8, 3 * GAP + 2.8); ax.set_ylim(-0.34, 0.5)
ax.set_ylabel("memory coefficient $M$")
set_frame(ax)
# direct labels for the two reference marks (first group), headline value on the one overshoot
hs2 = [mpl.patches.Patch(color=REALB, label="real $M$ (range over windows)"),
       mpl.lines.Line2D([], [], ls="--", lw=0.8, color=QLINE, label="real $M$, harness-quantized")]
leg2 = ax.legend(handles=hs2, loc="lower right", frameon=False, fontsize=TICK, handlelength=1.4, handletextpad=0.4,
                 labelspacing=0.2, borderaxespad=0.1)
ax.add_artist(leg2)
ax.annotate("$+0.43$", xy=(GAP, api.loc[("wikirevision", "L0"), "M_gen"]), xytext=(GAP + 1.05, 0.44),
            fontsize=TICK, color=API, va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=API))
hs = [mpl.lines.Line2D([], [], marker="o", color=QWEN, ls="-", ms=3.2, lw=1.2, label="Qwen3-30B-A3B"),
      mpl.lines.Line2D([], [], marker="D", color=API, ls="", ms=3.4, label="GPT-4o-Mini"),
      mpl.lines.Line2D([], [], marker="^", color=SLM, ls="", ms=3.8, label="Qwen3-8B (expl.)")]
fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, fontsize=TICK,
           handletextpad=0.25, columnspacing=0.5, handlelength=1.3, borderaxespad=0)
fig.subplots_adjust(left=0.17, right=0.985, top=0.88, bottom=0.21)
fig.savefig(os.path.join(PAP, "fig1_memory.pdf")); fig.savefig(os.path.join(PAP, "fig1_memory.png"), dpi=300)
plt.close(fig)

# ======================= Figure 2: structure vs timing, two stacked panels =======================
D = pd.read_csv(os.path.join(PRE, "taskD_timing_floor.csv"))
piv = D.pivot(index="dataset", columns="baseline", values="ks_mean").loc[DS]
xs = np.arange(4); W = 0.36
dm = [(qw.loc[(ds, "L1"), "degree_mmd"] - qw.loc[(ds, "L0"), "degree_mmd"]) / qw.loc[(ds, "L0"), "degree_mmd"] * 100 for ds in DS]
dk = [(qw.loc[(ds, "L1"), "ks"] - qw.loc[(ds, "L0"), "ks"]) / qw.loc[(ds, "L0"), "ks"] * 100 for ds in DS]
fig, (a, b) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True, gridspec_kw=dict(hspace=0.55))
STRUCT, TIMING = "#2b6cb0", "#c05621"
a.bar(xs - W / 2, dm, W, color=STRUCT, label="degree MMD (structure)")
a.bar(xs + W / 2, dk, W, color=TIMING, label="inter-event KS (timing)")
a.axhline(0, lw=0.6, color="black")
a.set_ylim(-70, 64); a.set_yticks([-60, -30, 0, 30])
a.set_ylabel("$\\Delta$ (%)")
a.legend(loc="upper right", ncol=1, frameon=False, fontsize=TICK, handlelength=1.0, handletextpad=0.3,
         bbox_to_anchor=(1.01, 1.06), borderaxespad=0, labelspacing=0.2)
set_frame(a)
for i, ds in enumerate(DS):
    ag = piv.loc[ds, ["agent_L0", "agent_L1", "agent_L2"]]
    b.plot([i + 0.16] * 2, [ag.min(), ag.max()], "-", color=QWEN, lw=0.9, zorder=2)
    b.plot([i + 0.16] * 3, ag.values, "o", color=QWEN, ms=3.0, zorder=3, mec="white", mew=0.3)
    b.plot(i - 0.16, piv.loc[ds, "bootstrap"], "o", mfc="white", mec=TIMING, mew=1.0, ms=4.2, zorder=4)
    b.plot(i - 0.16, piv.loc[ds, "hawkes"], "s", mfc="white", mec=GREY, mew=0.9, ms=3.6, zorder=4)
    b.plot(i - 0.16, piv.loc[ds, "poisson"], "^", mfc="white", mec=GREY, mew=0.9, ms=3.8, zorder=4)
hs = [mpl.lines.Line2D([], [], marker="o", color=QWEN, ls="-", ms=3, lw=0.9, label="agent L0–L2"),
      mpl.lines.Line2D([], [], marker="o", mfc="white", mec=TIMING, mew=1.0, ls="", ms=4.2, label="seed-IET bootstrap"),
      mpl.lines.Line2D([], [], marker="s", mfc="white", mec=GREY, mew=0.9, ls="", ms=3.6, label="Hawkes"),
      mpl.lines.Line2D([], [], marker="^", mfc="white", mec=GREY, mew=0.9, ls="", ms=3.8, label="Poisson")]
b.legend(handles=hs, loc="upper right", ncol=2, frameon=False, fontsize=TICK, handlelength=1.0, handletextpad=0.3,
         columnspacing=0.8, bbox_to_anchor=(1.01, 1.04), borderaxespad=0, labelspacing=0.25)
b.set_ylim(0, 1.3); b.set_yticks([0, 0.4, 0.8])
b.set_ylabel("KS")
b.set_xticks(xs); b.set_xticklabels(DSL); b.tick_params(axis="x", length=0, pad=1.5); b.set_xlim(-0.5, 3.5)
set_frame(b)
b.text(1.0, -0.36, "lower = better (both panels)", transform=b.transAxes, fontsize=TICK, color="#666666", ha="right", va="top")
fig.subplots_adjust(left=0.16, right=0.985, top=0.9, bottom=0.17)
TITLES2 = {"a": "Tool loop (L0$\\to$L1): structure improves 4/4, timing mixed",
           "b": "Seed-IET bootstrap beats every agent on KS, 3/4 datasets"}
for ax_, letter in [(a, "a"), (b, "b")]:
    y = ax_.get_position().y1 + 0.02
    fig.text(0.01, y, letter, fontweight="bold", fontsize=BASE + 1, va="bottom", ha="left")
    fig.text(0.07, y, TITLES2[letter], fontsize=TICK, va="bottom", ha="left")
fig.savefig(os.path.join(PAP, "fig2_decoupling.pdf")); fig.savefig(os.path.join(PAP, "fig2_decoupling.png"), dpi=300)
plt.close(fig)

# ======================= Figure 3 (controls): full width, two panels =======================
Zc = pd.read_csv(os.path.join(PRE, "taskC_cell_zscores.csv"))
Bc = pd.read_csv(os.path.join(PRE, "taskB_count_conditional.csv")); Bc = Bc[Bc.status == "ok"]
dscol = {"dianping": "#2b6cb0", "wikirevision": "#c05621", "weibo_daily": "#38a169", "weibo_tech": "#805ad5"}
fig, (a, b) = plt.subplots(1, 2, figsize=(FIG3_W, FIG3_H), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1.15]))
for ds, lab in zip(DS, DSL):
    s2 = Bc[Bc.dataset == ds]
    a.plot(s2.ks_raw_jit, s2.ks_countcond_jit, "o", color=dscol[ds], ms=3.6, label=ds, mec="white", mew=0.3)
a.plot([0, 0.5], [0, 0.5], "-", color="#aaaaaa", lw=0.7, zorder=0)
a.text(0.47, 0.41, "$y=x$", fontsize=TICK, color=GREY, ha="left", va="top")
a.set_xlim(0, 0.52); a.set_ylim(0, 0.52); a.set_xticks([0, 0.25, 0.5]); a.set_yticks([0, 0.25, 0.5])
a.set_xlabel("inter-event KS, as scored", labelpad=1)
a.set_ylabel("KS, count-\nconditioned")
a.legend(loc="upper left", frameon=False, fontsize=TICK, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25); set_frame(a)
rng = np.random.default_rng(1)
mets = ["Z_ks", "Z_abs_burst_gap", "Z_abs_M_gap", "Z_rhythm_median_ks"]
b.axhspan(-2, 2, color="#e8e8e8", zorder=0, lw=0)
for j, col in enumerate(mets):
    for _, row in Zc.iterrows():
        b.plot(j + rng.uniform(-0.18, 0.18), row[col], "o", ms=3.0, color=dscol[row.dataset], mec="white", mew=0.3)
b.set_yscale("symlog", linthresh=2)
b.set_yticks([-2, 0, 2, 10, 100]); b.set_yticklabels(["−2", "0", "2", "10", "100"]); b.set_ylim(-3.5, 130)
b.set_xticks(range(4)); b.set_xticklabels(["inter-event\nKS", "|burstiness\ngap|", "|$M$ gap|", "node\nrhythm"])
b.tick_params(axis="x", length=0, pad=1.5); b.set_xlim(-0.5, 3.5)
b.set_ylabel("$Z$ (RR units)")
b.text(3.45, -3.0, "real-vs-real band ($\\pm2\\sigma$)", fontsize=TICK, color="#777777", ha="right", va="bottom")
set_frame(b)
fig.subplots_adjust(left=0.105, right=0.99, top=0.79, bottom=0.235)
TITLES3 = {"a": "Count conditioning leaves timing error\nunchanged (median ratio 1.006)",
           "b": "Inter-event KS sits 5–22$\\sigma$ outside the real band;\nnode rhythm is nearest the $\\pm2\\sigma$ band"}
for ax_, letter in [(a, "a"), (b, "b")]:
    x = ax_.get_position().x0 - (0.085 if ax_ is a else 0.07); y = ax_.get_position().y1 + 0.03
    fig.text(x, y, letter, fontweight="bold", fontsize=BASE + 1, va="bottom", ha="left")
    fig.text(x + 0.03, y, TITLES3[letter], fontsize=BASE, va="bottom", ha="left", linespacing=1.1)
fig.savefig(os.path.join(PAP, "figA_controls.pdf")); fig.savefig(os.path.join(PAP, "figA_controls.png"), dpi=300)
plt.close(fig)
print("figures written to", os.path.abspath(PAP))
