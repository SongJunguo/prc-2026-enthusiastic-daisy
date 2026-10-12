#!/bin/bash
# Final prediction file in final_submitting order: per seed the fallback run replaces the
# trajectory run on rows without trajectory coverage and the visible rules are applied; the
# five seeds are averaged.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python}
T=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile_traj_tprof_tprofx_nmw
B=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile_nmw
ARGS=""
for suffix in "" _s2 _s3 _s4 _s5; do
  ARGS="$ARGS --model ${T}${suffix}_full_r1500=0.2 --fallback ${T}${suffix}_full_r1500=${B}${suffix}_full_r1500"
done
$PY scripts/final_write_submission.py --target final $ARGS --present-rollover --lirf-delay-zones \
  --schedule-floor --join-bound --experiment final_five_seeds \
  --output submissions/final/enthusiastic-daisy_final.parquet
