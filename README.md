# R2O: From Structural to Temporal Fidelity in Dynamic Graph Generation

Code, score tables, and paper source for

> Parham Hamouni and Ebrahim Bagheri. *From Structural to Temporal Fidelity in Dynamic Graph Generation.*
> 1st Workshop on Small Language Models for Agentic Systems (SLM-Agents), NeurIPS 2026.
> [OpenReview](https://openreview.net/forum?id=u9Uy8jmI75)

R2O evaluates whether a simulated event stream reproduces the *temporal process* of a real one, not just its
state statistics. It scores three axes — **Rate** (pace), **Rhythm** (inter-event-time distribution) and
**Order** (dependence between consecutive intervals) — and calibrates each against the variability between
matched windows of real data. The paper shows that LM-agent DyTAG simulators that improve structural fidelity
do not reproduce real temporal dependence, and that state/marginal statistics cannot detect this by construction.

## Layout

| Path | Contents |
|---|---|
| `r2o/` | The metric library, data-free: `iets_by_actor`, `memory_coeff`, `rho1_global`, `burstiness`, `rhythm_divergence`, `shape_pool`, `count_conditioned_ks`, `jitter`, `dedup`, plus `z_score` / `q_rr` calibration. These are the exact functions that produced `scores/`. |
| `scoring/` | Scripts that produced the score tables from the GDGB runs (tasks A–D, scheduler null control, E-REP scorer). Need the GDGB data root (`R2O_GDGB_ROOT`); see `scoring/README.md`. |
| `scores/` | The score tables the paper is built from (per-cell metrics, real-vs-real band and pairs, timing baselines, scheduler control, E-REP). |
| `paper/` | Camera-ready LaTeX source, generated tables/figures, and `paper/analysis/` — `recompute_stats.py`, `build_tables.py`, `make_figures.py`, and `verify_paper.py`, a gate that re-derives every number printed in the paper from `scores/`. |
| `docs/` | Pre-registration (2026-08-25) and the E-REP stage-1 report. |

## Quick start

```bash
pip install -r requirements.txt   # or: pip install -e .
python -c "import r2o; print(r2o.__version__)"
```

Score two event streams (times in seconds, one actor id per event):

```python
import numpy as np, r2o
from scipy.stats import ks_2samp

g_by = r2o.iets_by_actor(t_gen, actor_gen)     # actor -> inter-event times
r_by = r2o.iets_by_actor(t_real, actor_real)

rhythm_ks = ks_2samp(r2o.pooled(g_by), r2o.pooled(r_by)).statistic   # Rhythm
order_gap = abs(r2o.memory_coeff(g_by) - r2o.memory_coeff(r_by))     # Order (memory coefficient M)
shape_ks  = ks_2samp(r2o.shape_pool(g_by), r2o.shape_pool(r_by)).statistic  # Rhythm with Rate removed
```

Calibrate against a real-vs-real band (per-pair values from `scores/precheck/taskC_rr_pairs.csv`, or your own):

```python
Z = r2o.z_score(order_gap, rr_pairs["abs_M_gap"])   # band units; |Z| > 2 is outside the real reference
q = r2o.q_rr(order_gap, rr_pairs["abs_M_gap"])      # empirical tail position, floor 1/(N_RR+1)
```

## Reproducing the paper from the score tables

```bash
cd paper/analysis
python build_tables.py         # scores/ -> ../tab_*.tex        (byte-identical to the committed tables)
python make_figures.py         # scores/ -> ../fig*.pdf
python verify_paper.py         # re-derives every printed number from verified_stats.json + scores/; must print GATE PASS
cd .. && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`paper/analysis/verified_stats.json` is committed. Regenerating it (`python recompute_stats.py`) additionally
needs the raw GDGB runs (`R2O_GDGB_ROOT`, see `scoring/README.md`), because a few quantities — zero-IET
fractions, timestamp tie shares, the structural-evaluation CSVs — are read from the runs rather than from `scores/`.
The same holds for `matched_horizon_8b.py`.

## Reproducing the score tables from the GDGB runs

Requires the GDGB datasets and the archived generation runs (not redistributed here; see `scoring/README.md`).

## Data and models

The four datasets (`dianping`, `wikirevision`, `weibo_daily`, `weibo_tech`) are from the GDGB benchmark
(Peng et al., 2025) and are subject to its terms; the generation lineage is GAG (Ji et al., 2025).
This repository redistributes only derived score tables, not the datasets or generated streams.

## Citation

See `CITATION.cff`.

## License

MIT (code and score tables). The paper text is © the authors.
