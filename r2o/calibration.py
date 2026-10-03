"""Calibration against a matched real-vs-real (RR) reference band.

Given a discrepancy statistic d for a generated-vs-real pair and the same statistic over
N_RR matched real-vs-real window pairs, R2O reports
  Z    = (d - mean_RR) / sd_RR                     (band units; +-2 marks the band)
  q_RR = (1 + #{RR pairs with statistic >= d}) / (N_RR + 1)   (empirical tail position; floor 1/(N_RR+1))
Band construction (covariate-matched disjoint 10k-edge windows, reuse cap 3) is in
scoring/taskC_rr_band.py; the resulting bands for the four GDGB datasets are in
scores/precheck/taskC_rr_band_summary.csv and the per-pair values in taskC_rr_pairs.csv.
"""
import numpy as np


def z_score(d, rr_values):
    rr = np.asarray(rr_values, float); rr = rr[np.isfinite(rr)]
    if rr.size < 2 or rr.std(ddof=1) == 0:
        return np.nan
    return float((d - rr.mean()) / rr.std(ddof=1))


def q_rr(d, rr_values):
    rr = np.asarray(rr_values, float); rr = rr[np.isfinite(rr)]
    return float((1 + (rr >= d).sum()) / (rr.size + 1))
