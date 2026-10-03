# E-REP STAGE 1 REPORT — fair-window replication, API backbone (gpt-4o-mini)
Date: 2026-08-26. Runs: wikirevision_erep_{l0,l1}_gpt4omini_api on W-REPRESENT
(rows 2360000-2369999, 2019-12-04..2020-01-16), protocol byte-identical to incumbent API
cells; only the window differs. Branch next-paper, commits b220f39 (setup) + this one.
STATUS PER PREREG: Stage 1 of the two-stage E-REP (deviation entry 3). EXPLORATORY-tier
n=1 per cell; the E-REP falsifier is judged on Stage 2 (Qwen). Cost: ~$14 API.

## HEADLINE: the E-REP falsifier does NOT fire on stage 1

Falsifier required BOTH the decoupling AND the residual timing gap to vanish. Neither does:

1. DECOUPLING REPRODUCES. L0->L1 (agentic loop) on the fair window: final-window degree MMD
   0.078 -> 0.061 (-22%; incumbent window: 0.151 -> 0.121, -20%). Timing, jitter-arm KS:
   0.090 -> 0.088 (unchanged). Structure improves, distributional timing does not.
2. PROCESS FAILURES REPRODUCE. Temporal memory: real M = 0.15; generated L0 +0.02, L1 -0.04
   (erased; L1 slightly negative). Node-rhythm KS 0.38-0.40 (incumbent: 0.37). Rate: L0
   spans 1801 days vs the window's 39 (log-ratio 3.83); L1 3317 days (4.44) -- the loop makes
   the rate WORSE while improving structure. Zero-IET fraction 0.30-0.37 (tie artifact
   reproduces). Z vs RR band: as-generated KS +14 to +18 sigma; jitter-arm +2.8/+2.9 sigma.
3. WINDOW-SPECIFIC COMPONENT CONFIRMED (the nuance N3 predicted). Tie-corrected marginal KS
   is much smaller here (0.09) than on the cold-start window (0.31): a large part of the
   MARGINAL distribution error was window-specific. The DEPENDENCE failures are not. This
   validates the R6 decision to put dependence metrics in S_temporal: on marginal KS alone,
   stage 1 would have looked like partial falsification.
4. BURSTINESS GAP survives but shrinks: L0 +0.40, L1 +0.23 (dedup arm: +0.35/+0.16) vs
   +0.56 incumbent. Direction unchanged (over-bursty).

## Deltas vs incumbent-window anchors (same backbone, n=1 vs n=1)
metric (jitter arm)      fair window        cold-start window
KS inter-event           L0 0.090 L1 0.088  L0 0.310 L1 0.383
memory coeff gap         L0 -0.13 L1 -0.19  L0 +0.23 L1 -0.25
node rhythm KS           0.40 / 0.38        0.37 / 0.40
log duration ratio       3.83 / 4.44        3.98 / 3.62
degree MMD (final)       0.078 / 0.061      0.151 / 0.121
Note: M flips from overshoot (+0.23, cold-start L0) to erasure (-0.13, fair L0) on the SAME
backbone -- direction of the dependence failure is regime-dependent; its presence is not.
Same pattern as G0.6's dianping flip. The paper claims mis-specification, not a signed law.

## Implications
- Spine survives its first out-of-window test; proceed to Stage 2 (Qwen) for the
  pre-registered verdict. No design change needed.
- Structural numbers are BETTER on the fair window (degree MMD roughly halves): the
  incumbent cold-start window has been understating structural fidelity too. Protocol
  finding strengthened.
- G1 should stratify by window regime; the marginal-vs-dependence split must be reported
  per window (pre-registered anyway via R5/R6).

## Files
erep_stage1_scores.csv (both levels x 3 arms, full panel);
official eval CSVs under the worktree's graph_structural_results/.
