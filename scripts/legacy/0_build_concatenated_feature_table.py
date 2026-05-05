#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "train" / "radiomics" / "2_modeling" / "0_build_concatenated_feature_table.py"

print("DEPRECATED: use `prostate-radiomics build-features` instead.", file=sys.stderr)
raise SystemExit(subprocess.call([sys.executable, str(TARGET), *sys.argv[1:]]))
