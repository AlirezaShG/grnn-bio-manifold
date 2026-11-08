#!/usr/bin/env bash
# D1 tuning: six validation-only searches, one after another, each on a full pool.
cd "$(dirname "$0")/.."
PY=/c/ProgramData/anaconda3/python.exe
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for cfg in configs/v11/tune/t_*.json; do
  name=$(basename "$cfg" .json)
  echo "=== $name $(date +%T)"
  $PY -m grnn_bio tune --config "$cfg" --output "results/v11/tune/$name" --jobs 26 --resume > "results/v11/tune_$name.log" 2>&1 || echo "FAILED $name"
done
echo "=== all done $(date +%T)"
