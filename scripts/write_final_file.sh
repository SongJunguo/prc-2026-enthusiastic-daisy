#!/bin/bash
# Prediction file in final_submitting order.  Per seed the fallback run replaces the trajectory
# run on rows without trajectory coverage and the visible rules are applied; the seeds are averaged.
#
#   bash scripts/write_final_file.sh five-seed    # blocks ... traj tprof tprofx nmw, five seeds
#   bash scripts/write_final_file.sh three-seed   # safety configuration: ... traj tprof, three seeds
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python}
case "${1:-five-seed}" in
  five-seed)  T=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile_traj_tprof_tprofx_nmw; B=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile_nmw
              SUFFIXES=("" _s2 _s3 _s4 _s5); W=0.2; OUT=submissions/final/final_five_seeds.parquet ;;
  three-seed) T=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile_traj_tprof; B=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile
              SUFFIXES=("" _s2 _s3); W=0.3333333333333333; OUT=submissions/final/final_three_seeds.parquet ;;
  *) echo "usage: $0 [five-seed|three-seed]" >&2; exit 2 ;;
esac
ARGS=""
for suffix in "${SUFFIXES[@]}"; do
  ARGS="$ARGS --model ${T}${suffix}_full_r1500=$W --fallback ${T}${suffix}_full_r1500=${B}${suffix}_full_r1500"
done
$PY scripts/final_write_submission.py --target final $ARGS --present-rollover --lirf-delay-zones \
  --schedule-floor --join-bound --experiment "final_${1:-five-seed}" --output "$OUT"
