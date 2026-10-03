"""R2O: Rate / Rhythm / Order temporal-process fidelity for generated event streams."""
from .metrics import (secs, iets_by_actor, pooled, burstiness, memory_coeff, rho1_global, jitter, dedup,
                      rhythm_divergence, shape_pool, count_conditioned_ks)
from .calibration import z_score, q_rr
__version__ = "1.0.0"
