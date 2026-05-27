# AD-IRT-Mem: Psychometric Token Allocation for OCR-Memory Systems

**Author:** Jung Min Kang | ORCID: 0009-0007-9599-2792

| Result | Value |
|---|---|
| AD-IRT-Mem vs Top-k | **98.4%** win (ΔIRR = +0.057) |
| AD-IRT-Mem vs Uniform | 20% win (ΔIRR = −0.006) |
| Concavity insight | Uniform ≈ optimal when marginal returns homogeneous |
| LLTM weight rank ρ | 1.000 |
| IRT difficulty ρ | 0.744 |

## Quick verification (<1 second)

```bash
python src/verify_outputs.py
```

Recomputes all manuscript claims from packaged CSVs. No simulation needed.

## Full reproduction

```bash
pip install -r requirements.txt
PYTHONHASHSEED=0 python src/experiment.py --out_dir ./output --seeds 5
```

Runtime varies by CPU and BLAS backend. For a smoke test, use `--seeds 1`.

## Splits: Train 0-39 | CV 40-59 | Eval 60-79

## License: CC BY 4.0
