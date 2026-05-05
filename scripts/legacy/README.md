# Legacy Script Wrappers

The numbered scripts under `train/radiomics/` are preserved for traceability during
the refactor. Prefer the unified CLI for new work:

```bash
prostate-radiomics build-features --config configs/experiments/classical_5fold.yaml
prostate-radiomics train-classical --config configs/experiments/classical_5fold.yaml
prostate-radiomics train-deep --config configs/experiments/deep_5fold.yaml
prostate-radiomics compare --config configs/reports/clinical_comparison.yaml
```

Wrappers in this directory print a deprecation warning and call the original script.
