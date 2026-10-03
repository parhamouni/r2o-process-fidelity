# Scoring scripts

These produced the tables in `../scores/` from the GDGB reference streams and the archived GAG-lineage runs.
They are kept as run, with one change: the data root is read from the environment instead of a hard-coded path.

```bash
export R2O_GDGB_ROOT=/path/to/root      # must contain GDGB-ALGO/ (run configs) and GDGB_dataset/cache/*_edge.csv
cd scoring
python taskA_dependence.py              # Order: memory coefficient M, lag-1 autocorrelation, node rhythm  -> taskA_dependence_metrics.csv
python taskB_count_conditional.py       # Rate/Rhythm decomposition: shape KS, count-conditioned KS, duration ratio -> taskB_count_conditional.csv
python taskC_rr_band.py                 # real-vs-real band: covariate-matched disjoint 10k-edge windows, Z-scores -> taskC_rr_*.csv, taskC_cell_zscores.csv
python taskD_timing_floor.py            # timing baselines (Poisson, seed-IET bootstrap, Hawkes), 20 seeds -> taskD_timing_floor.csv
python scheduler_null.py                # scheduler-quantization null control (reads task A/C outputs) -> scheduler_null_control.csv
python score_erep.py                    # E-REP stage-1 window (GPT-4o-Mini wikirevision L0/L1) -> erep_stage1_scores.csv
```

`slurm/task{A,B,C,D}.sh` are the batch wrappers used (single CPU, 16 GB, up to 3 h each).

Expected layout under `R2O_GDGB_ROOT`:

```
GDGB_dataset/cache/{dianping,wikirevision,weibo_daily,weibo_tech}_edge.csv     # reference streams (actor_id, timestamp, original_edge_idx)
GDGB-ALGO/TDGG/LLMGraph/tasks/general/configs/<cell>/generated_data/wandb/<run>/files/media/edges.csv   # generated streams
```

Cell names, the run overrides for the two contaminated weibo L2 cells, and the short-run flags are declared in
`gdgb_common.py` (`BACKBONE_CELLS`, `RUN_OVERRIDES`, `CELL_FLAGS`) and match the paper's cell taxonomy (Appendix A).

## Notes on provenance

* The exploratory Qwen3-8B cells (`paper/analysis/slm_taskA_metrics.csv`, `slm_taskB.csv`, `slm_taskC_zscores.csv`)
  were scored with the same panel (the `score()` function in `paper/analysis/matched_horizon_8b.py`, which also
  produces the matched-horizon table); the one-off driver that wrote those three files is not included.
* `taskD_timing_floor.py` appends agent global-stream rows from an archived table when it is present under the data
  root; without it only the baseline rows are written (the paper uses the archived rows for Table 8's agent lines).
* Timestamps are parsed with `pandas.to_datetime(errors="coerce")`; the tables in `../scores/` were produced with
  pandas 1.4.3.
* The score tables are a **frozen snapshot** of the runs as of 2026-08-25 (the run directory of every scored cell is
  recorded in the `run_name` column of `scores/precheck/taskA_dependence_metrics.csv`). A working tree in which
  `latest-run` has since moved, or in which new structural evaluations were added, will give slightly different
  values for the run-derived quantities in `recompute_stats.py` (zero-IET fraction, structural MMD of later cells);
  `paper/analysis/verify_paper.py` checks the paper against the committed snapshot, not against a live tree.
