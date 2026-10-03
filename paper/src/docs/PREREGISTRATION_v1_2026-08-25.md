# PRE-REGISTRATION — Roll-Forward Temporal Validity of LLM-Agent DyTAG Generators
Version 1.0 | Date: 2026-08-25 | Status: PRE-DATA for G1/G2; POST-DATA for G0.5/G0.6/N1 (declared)
Companion evidence: G05_REPORT.md, G06_REPORT.md, N1_REPORT.md, N2_REPORT.md, N3_REPORT.md

===============================================================================================
## 0. HONESTY LEDGER — what is already observed vs what is genuinely pre-registered

ALREADY OBSERVED (cannot be pre-registered; reported as exploratory/confirmatory-of-record):
  G0.5 decoupling on 12 cells; G0.6 granularity control; N1 backbone contrast; N3 window-position
  finding. These were computed from sunk compute BEFORE this document. They are labelled
  EXPLORATORY in the paper, and each has a pre-registered confirmatory counterpart below.

GENUINELY PRE-REGISTERED (no data seen): G1 (open vs closed loop), G2 (baseline Pareto),
  H7 (leakage sensitivity on the matched window pair), the representative-window replication
  (E-REP), and every analysis rule in Sections 2-6.

===============================================================================================
## 1. HYPOTHESES AND FALSIFIERS

H1 DECOUPLING (confirmatory re-test). Structural and temporal fidelity are not coupled across
   configurations within a dataset.
   PRIMARY EVIDENCE: paired within-dataset contrasts (repeated measures), NOT the pooled
   cross-cell correlation. The pooled Spearman is reported as DESCRIPTIVE ONLY, because the 12
   cells are 3 levels x 4 datasets and cells within a dataset share data and setup.
   FALSIFIED IF: within-dataset, temporal error improves monotonically with structural error
   across the ladder on >=3 of 4 datasets.

H2 SCAFFOLD-INVARIANCE, BACKBONE-SENSITIVITY (revised after N1; now the paper's core claim).
   Temporal fidelity is invariant to the inference-time agent scaffold (loop, memory) that
   improves structural fidelity, but NOT to the backbone. Scaffold and backbone act on different
   axes.
   FALSIFIED IF: (a) a scaffold intervention improves S_temporal by >= the magnitude it improves
   S_macro (standardized), on >=3 of 4 datasets; or (b) backbone swaps move S_temporal by less
   than seed-level noise.

H3 CLOSED-LOOP SPECIFICITY (gate G1, fully pre-registered).
   Delta(h) = S_temporal^closed(h) - S_temporal^open(h) increases with horizon h.
   FALSIFIED IF: the pre-registered slope of Delta on h has a 95% CI containing 0 across seeds.
   INTERPRETATION IF FALSIFIED: the failure is temporal-process modelling / forecast uncertainty,
   not agent feedback -> do NOT build a feedback-oriented method. This is a real decision, and we
   commit to it in writing.

H4 DISPERSION PRECEDES STRUCTURE. Behavioural dispersion collapse precedes structural decay.
   EVIDENCE STANDARD: (a) temporal precedence, (b) across-seed association, (c) CONTROLLED
   PERTURBATION -- force dispersion preservation and test whether structural decay is delayed.
   Regression-based mediation inside a recursive stochastic process is NOT acceptable evidence
   and will not be claimed.

H6 DRIFT BEHAVIOUR. Agents may degrade differently from observation-fit neural generators under
   regime shift. NO theoretical prior is claimed: arXiv 2607.24662 proves an error floor for
   observation-based correction of MASKED FLOW-MATCHING generators; it does not show agents
   escape it, and they may do worse. Either direction ships.

H7 LEAKAGE SENSITIVITY (single dataset, pre-registered). Fidelity is inflated on a pre-cutoff
   window relative to a covariate-matched later window.
   THIS IS A SENSITIVITY TEST, NOT CONTAMINATION ELIMINATION. Cutoffs are fuzzy, later
   fine-tunes leak, platform regularities transfer.

H8 PERSISTENCE. Persistence preserves node rank-order individuality, with the gap widening in h:
   test the persistence x horizon interaction on S_individual.

E-REP REPRESENTATIVE-WINDOW REPLICATION (new, from N3). The G0.5/G0.6/N1 findings reproduce when
   the protocol is run on a covariate-representative window instead of edges 0..9999.
   FALSIFIED IF: the decoupling and the residual burstiness gap both vanish on the
   representative window -> the findings are cold-start artifacts, and the paper's claim
   narrows to "GDGB's window choice induces the apparent temporal failure" (still publishable,
   different title).

===============================================================================================
## 2. SCORING: composites, direction, and the reference band

2.1 DIRECTION NORMALIZATION (mandatory, applied before any aggregation).
    Every metric is oriented so LARGER = WORSE. Sign-flipped metrics: Spearman rank correlation,
    BERTScore / text similarity, coverage. Distance-like metrics (MMD, JSD, KS, Wasserstein,
    |burstiness gap|, CRPS) are already worse-is-larger. The orientation of each metric is fixed
    in this document and may not be changed after seeing results.

2.2 STANDARDIZATION against a MATCHED real-vs-real band.
    Z_m = ( D_m(sim, real) - mu_m^RR ) / sigma_m^RR
    The RR band is built from REAL-vs-REAL window pairs matched on: duration, edge count, actor
    count, edges-per-actor, and activity regime. Unmatched windows are NOT used: natural
    nonstationarity inflates the band, and an inflated band makes any simulator look "within
    natural variability". Construction: >=30 matched real pairs per dataset by block bootstrap
    within regime, blocks preserving actor identity.

2.3 FOUR COMPOSITES (pre-declared PRIMARY endpoints), each the mean of its family's Z_m:
    S_micro      next-partner top-k / MRR, edge-text similarity, per-node activity-rate error
    S_temporal   inter-event Wasserstein-1, |burstiness gap| (ties-removed arm), memory
                 coefficient (inter-event autocorrelation rho_1), temporal-motif transition JSD,
                 circadian/weekly JSD, AND a conditional-dependence term: node-rhythm divergence
                 P(dt|u) vs real. [ADDED 2026-08-25 per review R6: the spine is process not
                 marginal fidelity, so the panel must include a DEPENDENCE statistic, not only
                 marginal-distribution statistics. Marginal timing and conditional timing are
                 reported as separate sub-scores.]
    S_macro      degree MMD, spectra MMD, clustering, core-periphery, assortativity
    S_individual per-node degree/activity rank-trajectory Spearman
                 (RENAMED from S_identity: rank trajectories reflect graph position and exposure,
                 not intrinsic identity. "Identity" is reserved for H8, where persistence is
                 actually manipulated.)
    Forecast scoring (reported alongside, not inside the composites): CRPS per scalar trajectory,
    ensemble coverage of the real trajectory, rank histograms.

2.4 COMPOSITES NEVER APPEAR ALONE. A composite can mask sign-flips within its family, so the
    component panel is published alongside every composite value, always.

2.5 ARM ASSIGNMENT FOR TIE-CORRECTION (from G0.6, pre-registered).
    Moment statistics (burstiness): TIES-REMOVED arm is primary. Removing duplicate
    (actor, timestamp) events is a clean control for a CV-based statistic.
    Distributional statistics (KS, Wasserstein): JITTER-NULL arm is primary. Dedup changes the
    support of the distribution, so dedup-arm KS is not like-for-like; jitter preserves sample
    size and support.
    Both arms are reported for both statistic types.

2.6 DEGENERACY EXCLUSIONS (fixed now, from G0.5/N3).
    A dataset is excluded from ALL temporal claims if its REFERENCE window is temporally
    degenerate. Excluded: sephora, cora (median inter-event 0s, >90% duplicate timestamps at both
    ends of the timeline), imdb, wikilife (integer-year timestamps).
    Temporal claims are scoped to: dianping, wikirevision, weibo_daily, weibo_tech.
    Structural claims may still use all 8.

2.7 TEXT-METRIC SCOPE (from N3). Median comment length on wikirevision is 20 chars in the early
    window vs 100 in the recency window. Text-conditioned metrics are therefore NOT comparable
    across the early/late split; they are either scoped within-window or length-renormalized,
    and the choice is declared per table.

===============================================================================================
## 3. WINDOW SELECTION RULE (new, forced by N3)

3.1 The incumbent protocol scores original_edge_idx 0..9999 -- the FIRST 10k edges. On
    wikirevision that window ranked 138th of 138 candidate 10k-edge windows on covariate
    distance to the recent regime. Window choice is therefore a declared design variable, not an
    inherited default.

3.2 Windows used in this work, all fixed before G1 data:
    W-INCUMBENT   idx 0..9999             (for comparability with published numbers)
    W-REPRESENT   idx 2360000..2369999    wikirevision, 2019-12-04..2020-01-16 (best matched)
    W-POST        idx 2768732..2778731    wikirevision, 2024-10-14..2024-12-03 (recency)
    W-PRE-ALT     idx 2580000..2589999    wikirevision, 2022-05-13..2022-06-27 (robustness PRE)
    Matching variables: duration, actor count, edges-per-actor, median inter-event time
    (log-ratio distance). W-POST vs W-REPRESENT distance = 0.156; W-POST vs W-PRE-ALT = 0.162.

3.3 H7 is a SINGLE-DATASET test (wikirevision only). Per N3, no other GDGB dataset supports it:
    cora is degenerate at the tail, sephora ends 2023-03, dianping 2012, weibo_daily 2023-12,
    weibo_tech spans 7 days. This limitation is stated in the abstract, not buried.

3.4 PER-BACKBONE CUTOFF STATEMENTS. "Post-cutoff" is claimed per model, never generically. For
    each backbone used we state the training-cutoff basis and whether W-POST (Oct-Dec 2024) falls
    after it. Where a cutoff is unpublished or ambiguous, the arm is reported as
    "cutoff-unverified" and excluded from H7's headline.

===============================================================================================
## 3B. ROLL-FORWARD PROTOCOL AND LEAKAGE AUDIT (added 2026-08-25 per review R4/R5)

3B.1 DEFINITION. Our setting is REFERENCE-FREE CONTINUATION FROM A REAL WARM-START (semi-open-
     loop): warm-start on real history up to cutoff t_c, then generate with the model's own
     output re-entering context. Distinguished from teacher-forced (real history keeps entering)
     and fully-synthetic (no real warm-start). The OPEN-loop control injects real history each
     step; the CLOSED-loop condition feeds generated output back. Delta(h) is their difference.

3B.2 WHAT VANISHES AFTER t_c (pre-committed). For t > t_c: no real edges, no real timestamps,
     no real texts, and NO FUTURE-DERIVED INFORMATION of any kind. Audited channels, each of
     which must be shown to carry no post-cutoff information: (a) node metadata/personas,
     (b) prompt templates, (c) retrieval / candidate-node sets, (d) memory summaries,
     (e) normalization statistics (the RR band is built from PRE-cutoff real data only),
     (f) any graph feature computed over the full timeline. Each channel is checked and the
     check reported; an unaudited channel blocks the H7/G1 claim.

3B.3 EVENT-COUNT vs CONDITIONAL-TIMING (mandatory decomposition). A simulator that emits the
     wrong NUMBER of events trivially shifts the inter-event distribution. We therefore report
     BOTH: (i) activity-rate error (events per unit time vs real), and (ii) conditional timing
     error GIVEN matched event count -- via count-conditioned resampling or an explicit
     rate/shape factorization. No timing claim is made without the count-conditioned version,
     so "timing failure" is never a restatement of "activity-rate failure".

===============================================================================================
## 4. POWER PLAN (measured, not asserted)

4.1 MEASURED SEED-LEVEL NOISE (pooled within-arm SD across 21 fast-tier replicate arms on
    wikirevision + weibo_daily, n=3-4 each, 65 scored cells):
        KS(inter-event), as-generated ......... SD = 0.0537
        KS(inter-event), ties-removed ......... SD = 0.0642
        |burstiness gap|, ties-removed ........ SD = 0.0890
        zero-inter-event fraction ............. SD = 0.0611

4.2 REQUIRED N PER ARM (two-sided alpha=0.05, power=0.80), fast tier, by detectable effect and
    seed correlation from shared warm-starts:
        metric            effect   rho=0    rho=0.5   rho=0.8
        KS                 0.05      19       10         4
        KS                 0.10       5        3         1
        |burst gap|        0.05      50       25        10
        |burst gap|        0.10      13        7         3
    Paired designs with shared warm-starts are therefore mandatory: at rho=0.8 the same power
    costs 4-5x fewer runs. All G1/G2 arms share warm-starts by construction.
    OPERATIONAL DEFAULT: n=10 seeds per arm at the fast tier, which powers KS effects >=0.05 and
    burstiness effects >=0.10. A 5-seed pilot re-estimates SD before the final N is fixed; if the
    pilot SD exceeds 1.5x the value above, N is raised and the change is logged with its reason.

4.3 FULL-PROTOCOL VARIANCE IS UNMEASURED -- DO NOT ASSUME IT IS SMALL.
    Exactly ONE full-protocol replicate set exists on an informative dataset
    (wikirevision l0_gpt4omini_api, n=3), and its SDs are far smaller than fast tier:
    KS 0.0032 (16.8x smaller), KS ties-removed 0.0017 (37.8x),
    |burstiness gap| 0.0021 (42.4x), zero-IET fraction 0.0032
    (19.1x). Note the spread across metrics is itself large, and the biggest gap
    (42.4x) is on |burstiness gap| -- the metric the headline claim depends on --
    so a single pooled ratio must not be quoted. That arm is an API backbone whose sampling
    determinism may
    not represent local-backbone variance, and n=3 from a single arm cannot license "n=1
    suffices". Pre-registered rule: the first full-protocol experiment runs >=3 seeds on >=2
    arms to estimate full-tier SD, and only then is the final full-tier N set. No full-protocol
    claim rests on a single run.

4.4 INFERENCE. Paired differences with block-bootstrap CIs (blocks = actors). Within-dataset
    Wilcoxon signed-rank for the ladder contrasts. Across the 4 datasets we report direction and
    per-dataset effect, not a pooled p-value: n=4 cannot support a pooled test, and the
    decoupling claim is a within-dataset repeated-measures result.

4.5 MULTIPLICITY. The four composites are the primary family: Holm-Bonferroni across them.
    Component metrics are secondary and reported with Benjamini-Hochberg, descriptively.
    No component result is promoted to a headline claim after the fact.

===============================================================================================
## 5. GATES AND PRE-COMMITTED DECISIONS

G0   Reproducibility -- ALREADY SATISFIED (WSDM work; best framework cell matches-or-beats the
     GPT-4o-Mini GAG-General on 7/8 datasets; eval-validity bug found and fixed).
G0.5 Decoupling -- PASSED (exploratory; confirmatory counterpart = H1 + E-REP).
G0.6 Granularity control -- PASSED with scope restriction (ties explain a median 41% of the
     burstiness gap; residual survives on 10/12 cells; direction is regime-dependent).
N1   Backbone contrast -- DONE; H2 revised to scaffold-invariance / backbone-sensitivity.
N3   Window position -- DONE; window selection is now a declared variable.
E-REP (PRIORITY ZERO, before any expensive G1): re-score on a covariate-representative window
     (W-REPRESENT) instead of the cold-start W-INCUMBENT. If the decoupling and residual gap both
     vanish, the main result was regime-dependent and G1's design changes. Runs first.
G1   OPEN vs CLOSED LOOP. Delta(h) grows -> feedback-driven degradation identified, proceed to
     method work. Delta(h) ~ 0 -> the failure is temporal-process modelling; do not build a
     feedback method. BOTH outcomes ship.
G2   BASELINE PARETO (EdgeBank, Hawkes, one neural temporal generator, cost-normalized).
     Framed as "which component justifies an LLM?", not as a single verdict. Expected shape:
     Hawkes owns timing, neural TG owns structure, LLM owns text/semantics -> motivates the
     factorized architecture.
     STOP RULE: if agents are Pareto-dominated on ALL FOUR composites at matched cost, we stop
     and reconsider the programme rather than proceeding to build a method.
G3   METHOD SELECTION, decided by G1/G2, from the pre-declared menu: (i) lambda(history only),
     (ii) lambda(agent state, history), (iii) Delta t emitted by the LLM, (iv) hybrid -- LLM
     decides whom/what, temporal model decides when. Guardrails: non-inferiority on S_macro and
     edge-text similarity. The outcome "LLMs should not generate time" (iv wins, iii loses) is
     an ACCEPTABLE and pre-declared result, not a fallback.

===============================================================================================
## 6. WHAT WOULD MAKE US ABANDON THE PAPER
  * E-REP falsified AND G1 shows Delta(h) ~ 0: the temporal failure is a cold-start artifact of
    the incumbent window with no rollout-specific component. Remaining contribution is the
    protocol critique alone -- a short paper, not this one.
  * G2 stop rule triggered (Pareto domination on all four composites at matched cost).
  * The measured full-tier SD (4.3) turns out large enough that no affordable N can resolve the
    effects in 4.2 -- in which case the honest output is a methods/instrument paper reporting
    that current simulator comparisons at this scale are underpowered.

## 7. DEVIATION LOG
Any departure from this document is recorded here with date, reason, and whether it was made
before or after seeing the affected data. An empty log is itself a claim; keep it honest.
  2026-08-25 | §4.3 corrected post-hoc (pre-data, no experimental results affected): the
  fast/full SD gap had been written as "~15-30x" but the computed per-metric ratios are
  KS 16.8x, KS-dedup 37.8x, |burstiness gap| 42.4x, zero-IET 19.1x. The range understated
  the gap on |burstiness gap|, the metric the headline claim depends on. Replaced with
  per-metric ratios; the §4.3 rule (>=3 seeds on >=2 arms before any full-tier N is set)
  is unchanged and is if anything better supported by the larger spread.

  2026-08-25 | §2.2 amended PRE-DATA for G1/E-REP (motivated by the RR-band PILOT on real data
  only, taskC_rr_band_summary.csv; no simulator-vs-real result was affected): on weibo_tech,
  disjoint 10k-edge windows cannot yield >=30 matched pairs (only 10 windows exist; 13 pairs at
  reuse cap 3). The prereg's own block-bootstrap construction (§2.2) is therefore MANDATORY on
  weibo_tech, and the disjoint-window shortcut is permitted only where it reaches n>=30
  (dianping, wikirevision, weibo_daily). Falls back to the stricter, already-specified method;
  no metric or hypothesis changes.

  2026-08-25 | §2.3 clarified PRE-DATA for G1/E-REP: Wasserstein-1 on inter-event times is
  Z-scored on LOG-transformed IETs (log(1+dt), seconds). Reason (pilot, real-data band only):
  in raw seconds the RR-band sigma is numerically tiny relative to simulator distances,
  producing meaningless Z values (observed up to ~4.6e4). The metric itself is unchanged;
  only the unit convention for standardization is fixed, and it is fixed before any
  confirmatory data exists. Raw-seconds W1 is still reported descriptively.

  2026-08-25 | §1 E-REP amended PRE-DATA: E-REP runs as a TWO-STAGE design. Stage 1 (pilot):
  GPT-4o-Mini via OpenRouter API on W-REPRESENT — the backbone with existing incumbent-window
  comparison cells (l0/l1_gpt4omini_api) and measured near-zero sampling variance (§4.3).
  Stage 2 (confirmatory for the headline): Qwen3-30B on local GPU, same window, since the
  headline exploratory findings are Qwen cells. Reason: GPU queue is occupied by WSDM work;
  the API stage buys an early read at dollar cost. E-REP's falsifier is evaluated on Stage 2;
  Stage 1 is reported as the cross-backbone replicate. API model version is logged per call
  and pinned by date.

  2026-08-25 | §1 E-REP scope declared PRE-DATA: if Stage 2 REPRODUCES the decoupling and
  residual gap on W-REPRESENT, a second-dataset E-REP (weibo_daily, best-matched 10k window
  by the §3.2 covariates) MAY be run as a robustness add-on and will be reported wherever it
  lands; if Stage 2 FALSIFIES, the second dataset is REQUIRED before accepting the
  cold-start-artifact conclusion (one window on one dataset must not kill the paper's spine
  by itself). Declared now so the option cannot be exercised selectively after results.

  2026-08-26 | §4/G1 primary estimand amended PRE-DATA (no G1 data exists; motivated by
  external methodological review, not by results): the linear slope of Delta(h) on h is
  DEMOTED to secondary. Primary estimand: horizon-weighted AUC_Delta = sum_h w_h *
  (S_temporal^closed(h) - S_temporal^open(h)), uniform weights unless the pilot shows
  horizon-variance heterogeneity (then inverse-variance), inference by seed-level bootstrap
  over PAIRED rollouts (horizon points within a rollout are dependent; the slope treated
  them as independent regression points, which was wrong). H3's falsifier becomes: 95% CI
  of AUC_Delta contains 0. Reason: rollout error may be nonlinear (saturating, delayed
  divergence); a slope is misspecified under those shapes, AUC is robust to them. The
  horizon-resolved count-vs-conditional-timing decomposition (3B.3) is promoted from
  supplementary to PRIMARY analysis for G1: any feedback claim must attribute closed-loop
  divergence among count error, conditional-timing error, and structure/text feedback.

  2026-08-26 | §2.2 RR-band inference clarified PRE-DATA for confirmatory runs: Z=0 means
  the sim-real discrepancy equals the MEAN real-real discrepancy, not statistical
  indistinguishability. Primary inferential report for confirmatory claims: empirical
  exceedance p_RR = (1 + #{D_i^RR >= D^SR}) / (N_RR + 1) (add-one), i.e. the sim-real
  discrepancy's position in the matched RR distance distribution; Z-scores retained for
  visualization and cross-metric comparability. RR band size N_RR reported wherever p_RR
  is reported. Applied retroactively to exploratory results as a robustness view (16/16
  cells at the exceedance floor on inter-event KS), which cannot change any exploratory
  conclusion (all were band-exterior under both views).

  2026-08-26 | §4/G1 estimand FULLY specified PRE-DATA (completes entry 6; no G1 data
  exists): AUC_Delta weights are UNIFORM, w_h = 1/H, over horizon checkpoints equally
  spaced in EVENT COUNT (the protocol's native clock; wall-clock checkpoints reported
  descriptively). Direction: composites are direction-normalized so larger = worse; the
  directional alternative is H1: AUC_Delta > 0 (closed loop worse), judged by a two-sided
  95% bootstrap CI over paired seed-level rollouts. The inverse-variance option floated in
  entry 6 is WITHDRAWN to eliminate a data-dependent weighting choice; uniform weights are
  final. Bootstrap replicates in RR-band construction estimate band uncertainty only and
  are never counted as additional null observations in q_RR (renamed from p_RR: it is a
  descriptive exceedance fraction, not a hypothesis-test p-value).

  2026-08-27 | §3B.1 closed-arm actor scheduling declared PRE-DATA (no G1 data exists; the
  instrument has not yet produced a single successful run): past the cutoff round, the
  closed (future-blind) arm cannot use the reference actor schedule, so the INSTRUMENT
  supplies one: actors are sampled with replacement from the recency-weighted empirical
  activity distribution of the simulator's own seed + generated history (exponential decay,
  half-life = one round of events, i.e. edge_delta=50 events; per-seed RNG). Slot count per
  round stays edge_delta, so the arms are count-matched by construction and 3B.3's
  count channel is controlled; the simulator's temporal claim is measured on its EMITTED
  timestamps, never on the schedule. DECLARED LIMITATION: this scheduling policy is part of
  the instrument, not the simulator under test; closed-arm results are conditional on it,
  and a schedule-sensitivity check (uniform-over-active-actors alternative) is a
  pre-declared robustness arm for the G1 pilot. Any change to the policy after pilot data
  exists goes through this log as post-data.
