"""Synthetic check of Prop. (i)-(ii) in Sec. 3: intra-actor shuffle and rate
rescaling applied to a copula-AR(1) lognormal event stream. Writes
synthetic_invariance_check.csv (Table in App. proofs). Seed fixed."""
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(0)
def burst(t): s, m = t.std(), t.mean(); return (s-m)/(s+m)
def memory(t): return np.corrcoef(t[:-1], t[1:])[0,1]
def gen_actor(k, rho, sigma=1.5):
    z = np.empty(k); z[0] = rng.standard_normal()
    for i in range(1,k): z[i] = rho*z[i-1] + np.sqrt(1-rho**2)*rng.standard_normal()
    return np.exp(sigma*z)
def pooled_stats(actors):
    pooled = np.concatenate(actors)
    return pooled, burst(pooled), np.mean([memory(a) for a in actors if len(a) > 5])
real = [gen_actor(rng.integers(20,200), 0.35) for _ in range(300)]
pr, Br, Mr = pooled_stats(real); rows = []
def score(name, actors):
    pg, Bg, Mg = pooled_stats(actors)
    rows.append((name, round(stats.ks_2samp(pg, pr).statistic,3),
                 round(stats.wasserstein_distance(np.log(pg), np.log(pr)),3),
                 round(abs(Bg-Br),3), round(Mg,3), round(abs(Mg-Mr),3)))
score("faithful replicate", [gen_actor(rng.integers(20,200), 0.35) for _ in range(300)])
score("intra-actor shuffle", [rng.permutation(a) for a in real])
score("renewal, same marginal", [gen_actor(rng.integers(20,200), 0.0) for _ in range(300)])
score("rate x20", [20*a for a in real])
score("rate x20 + shuffle", [20*rng.permutation(a) for a in real])
df = pd.DataFrame(rows, columns=["transform","IET_KS","logW1","|dB|","M_gen","|dM|"])
df.insert(1,"real_M",round(Mr,3)); df.insert(2,"real_B",round(Br,3))
df.to_csv(__file__.replace(".py",".csv"), index=False); print(df.to_string(index=False))
