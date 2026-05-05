#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "train" / "radiomics" / "2_modeling" / "4_run_deep_tabular_suite.py"

print("DEPRECATED: use `prostate-radiomics train-deep` instead.", file=sys.stderr)
raise SystemExit(subprocess.call([sys.executable, str(TARGET), *sys.argv[1:]]))
