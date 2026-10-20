"""Shared pytest setup: make python/fpp.py importable and use a non-interactive backend."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("FPP_DATA", str(ROOT / "data"))
REF = ROOT / "tests" / "reference"
