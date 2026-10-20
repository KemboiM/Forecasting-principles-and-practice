#!/usr/bin/env bash
# Run ALL model tests (R + Python) from the project root:  bash tests/run_tests.sh [chapter-number]
set -e
cd "$(dirname "$0")/.."
CH=${1:-}
if [ -n "$CH" ]; then CH=$(printf "%02d" "$CH"); fi
echo "================ R model tests ================"
if [ -n "$CH" ]; then
  Rscript -e "testthat::test_file('tests/R/test-ch$CH.R', reporter='summary')"
else
  Rscript -e "testthat::test_dir('tests/R', reporter='summary')"
fi
echo "================ Python model tests ================"
if [ -n "$CH" ]; then python3 -m pytest -q "tests/python/test_ch$CH.py"; else python3 -m pytest -q tests/python; fi
