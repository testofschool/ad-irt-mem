# AD-IRT-Mem: Psychometric Token Allocation for OCR-Memory Systems

**Author:** Jung Min Kang | ORCID: 0009-0007-9599-2792

| Result | Value |
|---|---|
| AD-IRT-Mem vs Top-k | **98.4%** win (ΔIRR = +0.057) |
| AD-IRT-Mem vs Uniform | 20% win (ΔIRR = −0.006) |
| Concavity insight | Uniform ≈ optimal when marginal returns homogeneous |
| LLTM weight rank ρ | 1.000 |
| IRT difficulty ρ | 0.744 |

> These are the originally published numbers (`output/`). They were produced with a query-ability
> indexing bug; see **Correction (2026-09-29)** below for corrected numbers. The headline
> conclusion is unchanged.

## Quick verification (<1 second)

```bash
python src/verify_outputs.py
```

Recomputes all manuscript claims from packaged CSVs. No simulation needed.

## Full reproduction

```bash
pip install -r requirements.txt
# corrected code (default, --theta_mode own)
PYTHONHASHSEED=0 python src/experiment.py --out_dir ./results/fixed_theta --seeds 5
# original published run (reproduces output/*.csv; see 'Tie order' below)
PYTHONHASHSEED=0 python src/experiment.py --out_dir ./output --seeds 5 --theta_mode legacy
```

Runtime varies by CPU and BLAS backend. For a smoke test, use `--seeds 1`.

## Splits: Train 0-39 | CV 40-59 | Eval 60-79

## Correction (2026-09-29)

**Bug.** In `src/experiment.py` the per-query ability used by the IRT Fisher and AD-IRT-Mem
allocators was indexed as `et[qi % 40]`, where `et` holds θ estimates for training queries 0–39 only.
So CV queries 40–59 used the θ of training queries 0–19, and evaluation queries 60–79 used the θ of
training queries 20–39 — i.e. another query's ability, not their own. Affected: `select_w`
(CV), `experiment_1` and `experiment_2`. `experiment_3` (LLTM) and the difficulty-recovery ρ do not
use query θ and are unchanged. Uniform, Recency and Top-k do not use θ and are unchanged.

**Fix.** New default `--theta_mode own`: each query 40–79 gets a θ estimated from its *own*
simulated historical rows (same observation protocol as training — 80 observed chunks per query in
Exp. 1, sparsity-matched count in Exp. 2 — drawn from an independent RNG stream), with item
parameters held fixed at the training estimates and the same learning rate / iterations / prior as
`fit_2pl`. The training fit and all other random streams are untouched. `--theta_mode legacy`
reproduces the original run (`output/*.csv` byte-identical on Linux x86_64, 2026-09-29; see *Tie order*);
`--theta_mode oracle` uses the true simulated θ as an upper-bound sensitivity check only.
Diagnostic: mean Spearman ρ between the θ actually used and the true θ for eval queries 60–79
(5 seeds × gap {1,3,5}) is 0.079 under the bug and 0.938 after the fix.

**Tie order (2026-09-29).** An external audit re-ran the code on macOS arm64 and got different
Uniform results at budget 0.1 (25 rows of `exp1_main.csv`), which changed the AD-IRT-Mem vs Uniform
win counts. Cause: `_distribute` picks the top `n_aff` candidates with `np.argsort(-weights)`, and
under Uniform all weights are equal, so the pick depended on the platform's sort order for ties.
The call now uses `np.argsort(-weights, kind="stable")`, which keeps ties in candidate order on every
platform. On Linux x86_64 this reproduces the committed `output/*.csv` (legacy) and
`results/fixed_theta/*.csv` (own) byte-for-byte, so no reported number changes; it is not yet re-run on macOS.

**Corrected numbers** (Exp. 1, 125 seed×budget×gap cells; AD-IRT-Mem minus baseline).
Original values are kept above and in `output/`; corrected outputs are in `results/fixed_theta/`
and `results/results_fixed_theta.json`.

| Comparison | Original (bug) | Corrected (own θ) | Oracle θ (sensitivity) |
|---|---|---|---|
| vs Top-k | 98.4% win, ΔIRR +0.0574 | 99.2% win, ΔIRR +0.0584 | 99.2%, +0.0588 |
| vs Uniform | 20.0% win, ΔIRR −0.0060 | 24.0% win, ΔIRR −0.0050 | 25.6%, −0.0046 |
| vs Recency | 40.0% win, ΔIRR −0.0017 | 49.6% win, ΔIRR −0.0007 | 52.8%, −0.0003 |
| vs IRT Fisher | 55.2% win, ΔIRR +0.0010 | 49.6% win, ΔIRR +0.0014 | 57.6%, +0.0017 |
| D=1, budget 30%: seeds where AD-IRT-Mem > Uniform | 3/5 | 4/5 | 4/5 |
| Mean IRR, AD-IRT-Mem / IRT Fisher | 0.1446 / 0.1436 | 0.1457 / 0.1443 | 0.1460 / 0.1444 |
| LLTM weight rank ρ, IRT difficulty ρ | 1.000, 0.744 | unchanged | unchanged |

**Does the headline survive?** Yes, qualitatively. AD-IRT-Mem still beats Top-k almost always, and
still does **not** beat Uniform on average (it wins only about a quarter of cells and has a small
negative mean ΔIRR) — the concavity finding that uniform allocation is near-optimal stands. The
fix moves AD-IRT-Mem slightly closer to Uniform and Recency but does not change any sign. The
AD-IRT-Mem vs IRT Fisher comparison is a coin flip in every mode (49.6–57.6% win, ΔIRR ≤ +0.002)
and should not be read as an advantage. The paper PDF in `arxiv/` still reports the original
numbers and the "historical estimate" approximation described in its §Setup.

## License

CC0 1.0 Universal — see `LICENSE`. (An earlier version of this README said "CC BY 4.0", which did
not match the `LICENSE` file; corrected 2026-09-29 to match `LICENSE`.)
