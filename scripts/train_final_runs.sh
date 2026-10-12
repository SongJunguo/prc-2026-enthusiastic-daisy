#!/bin/bash
# Fit the ten runs of the five-seed configuration: per seed a trajectory run and a fallback run
# without the trajectory blocks.  A run counts as finished only when run.json and both prediction
# files exist; an unfinished run folder is removed and fitted again, so the script can be restarted.
# TabM members use seed, seed+1 and seed+2; the seeds are three apart.
# The three-seed safety configuration uses the same loop with --extra "arrtaxi deptaxi inb refile
# traj tprof" / "arrtaxi deptaxi inb refile" and the first three seeds.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python}
R=artifacts/runs/final_models
COMMON="--family xgbmix --mode final --rounds 1500 --class-weight delta2 --reg-lgb --reg-cat --reg-tabm --tabm-seeds 3 --tabm-epochs 20 --tabm-weight 0.5 --class-tabm --class-tabm-seeds 3 --class-tabm-skip-group lirf_missing_gt6h --rollover-state"
T=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile_traj_tprof_tprofx_nmw
B=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile_nmw
mkdir -p logs
fit() {  # fit <run name> <seed> <log> <extra blocks...>
  local name=$1 seed=$2 log=$3; shift 3
  if [ -f "$R/$name/run.json" ] && [ -f "$R/$name/pred_final.parquet" ] && [ -f "$R/$name/pred_public.parquet" ]; then return 0; fi
  rm -rf "${R:?}/$name"
  $PY scripts/final_train_predict.py $COMMON --seed "$seed" --extra "$@" --name "$name" > "logs/$log" 2>&1
}
for spec in ":20261008" "_s2:20261011" "_s3:20261014" "_s4:20261017" "_s5:20261020"; do
  suffix=${spec%%:*}; seed=${spec##*:}
  fit "${T}${suffix}_full_r1500" "$seed" "trajectory${suffix}.log" arrtaxi deptaxi inb refile traj tprof tprofx nmw
  fit "${B}${suffix}_full_r1500" "$seed" "fallback${suffix}.log" arrtaxi deptaxi inb refile nmw
done
