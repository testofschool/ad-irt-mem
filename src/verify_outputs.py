#!/usr/bin/env python3
"""Verify all manuscript claims from packaged CSV files (<1 second)."""
import pandas as pd, numpy as np, sys, os

out = os.path.join(os.path.dirname(__file__), "..", "output")
ok = True

def check(name, condition, detail=""):
    global ok
    status = "PASS" if condition else "FAIL"
    if not condition: ok = False
    print(f"  [{status}] {name}" + (f" ({detail})" if detail else ""))

print("=== AD-IRT-Mem: Verify Manuscript Claims ===\n")

df = pd.read_csv(os.path.join(out, "exp1_main.csv"))
df3 = pd.read_csv(os.path.join(out, "exp3_lltm.csv"))

# Table 2: IRR by budget
agg = df.groupby(["method","budget_frac"])["mean_irr"].mean().reset_index()
u30 = agg[(agg["method"]=="uniform")&(agg["budget_frac"]==0.3)]["mean_irr"].values[0]
check("Table 2: Uniform at 30%", abs(u30 - 0.155) < 0.001, f"{u30:.3f}")

# Win rates
for bl, expected_pct, expected_delta in [
    ("topk", 98.4, +0.057), ("fisher", 55.2, +0.001),
    ("uniform", 20.0, -0.006), ("recency", 40.0, -0.002)]:
    w, t, ds = 0, 0, []
    for _, g in df.groupby(["seed","budget_frac","difficulty_gap"]):
        a = g[g["method"]=="adirt"]["mean_irr"].values
        b = g[g["method"]==bl]["mean_irr"].values
        if len(a) and len(b):
            t += 1; d = a[0]-b[0]; ds.append(d)
            if d > 1e-6: w += 1
    pct = 100*w/t; delta = np.mean(ds)
    check(f"vs {bl}: {pct:.1f}% (expect {expected_pct}%)",
          abs(pct - expected_pct) < 0.2, f"Δ={delta:+.4f}")

# Table 3: D=1 win rates
sub = df[(df["budget_frac"]==0.3)&(df["difficulty_gap"]==1.0)]
aw = sum(1 for _, sg in sub.groupby("seed")
         if sg[sg["method"]=="adirt"]["mean_irr"].values[0] >
            sg[sg["method"]=="uniform"]["mean_irr"].values[0] + 1e-6)
check("Table 3: D=1 AD-IRT win 60%", aw == 3, f"{aw}/5")

# LLTM
check(f"LLTM weight rank ρ = 1.0", df3["w_rank_rho"].mean() > 0.999,
      f"{df3['w_rank_rho'].mean():.3f}")
check(f"LLTM R² ≈ 0.147", abs(df3["r2"].mean() - 0.147) < 0.01,
      f"{df3['r2'].mean():.3f}")

# IRT recovery
check(f"b recovery ρ ≈ 0.744", abs(df["b_rho"].mean() - 0.744) < 0.01,
      f"{df['b_rho'].mean():.3f}")

print(f"\n{'='*40}")
print(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
sys.exit(0 if ok else 1)
