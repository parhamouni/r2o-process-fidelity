#!/usr/bin/env python
"""Robustness of the agent global lag-1 IET autocorrelation (Table 8) to round-based timestamp ties.
For each preregistered Qwen3-30B-A3B cell, recompute rho1 on (i) the deduplicated stream (as printed),
(ii) the stream with all zero IETs removed, (iii) the stream collapsed to distinct timestamps.
Needs R2O_GDGB_ROOT (reads the archived runs as currently resolved; for three cells the resolved run differs from the
August score snapshot by <= 0.005 in rho1, immaterial for this check). Writes rho1_tie_check.csv next to this script."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "src", "precheck"), os.path.join(HERE, "..", "..", "scoring"), os.environ.get("R2O_PRECHECK", "")):
    if p and os.path.exists(os.path.join(p, "gdgb_common.py")): sys.path.insert(0, p); break
import gdgb_common as C
cells = pd.read_csv(os.path.join(HERE, "cells_main.csv")); cells = cells[cells.backbone == "Qwen3-30B-A3B"]
def r1(d): return float(np.corrcoef(d[:-1], d[1:])[0, 1]) if d.size > 11 else np.nan
rows = []
for cfg in cells.config:
    gen, run = C.load_gen(cfg); t = C.secs(gen["timestamp"]); a = gen["actor_id"].to_numpy()
    td, _ = C.dedup(t, a); s = np.sort(td); d = np.diff(s); du = np.diff(np.unique(s))
    rows.append(dict(config=cfg, rho1_dedup=C.rho1_global(td), rho1_nonzero_iets=r1(d[d > 0]), rho1_distinct_timestamps=r1(du),
                     zero_iet_frac=float((d == 0).mean()), n_distinct_timestamps=int(len(du) + 1)))
df = pd.DataFrame(rows); df["max_abs_change"] = (df[["rho1_nonzero_iets", "rho1_distinct_timestamps"]].sub(df.rho1_dedup, axis=0)).abs().max(axis=1)
df.to_csv(os.path.join(HERE, "rho1_tie_check.csv"), index=False); print(df.round(3).to_string()); print("max |change| over cells: %.4f" % df.max_abs_change.max())
