#!/bin/bash
# Fit the ten runs of the final configuration: five seeds, each with a trajectory run and a
# fallback run without the trajectory blocks.  Finished runs are skipped, so the script can be
# restarted.  TabM members use seed, seed+1 and seed+2; the seeds are three apart.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
PY=${PYTHON:-python}
R=artifacts/runs/final_models
COMMON="--family xgbmix --mode final --rounds 1500 --class-weight delta2 --reg-lgb --reg-cat --reg-tabm --tabm-seeds 3 --tabm-epochs 20 --tabm-weight 0.5 --class-tabm --class-tabm-seeds 3 --class-tabm-skip-group lirf_missing_gt6h --rollover-state"
T=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile_traj_tprof_tprofx_nmw
B=xgbmix_wlct3c3_r_arrtaxi_deptaxi_inb_refile_nmw
mkdir -p logs
for spec in ":20261008" "_s2:20261011" "_s3:20261014" "_s4:20261017" "_s5:20261020"; do
  suffix=${spec%%:*}; seed=${spec##*:}
  [ -f "$R/${T}${suffix}_full_r1500/run.json" ] || $PY scripts/final_train_predict.py $COMMON --seed "$seed" \
    --extra arrtaxi deptaxi inb refile traj tprof tprofx nmw --name "${T}${suffix}_full_r1500" > "logs/trajectory${suffix}.log" 2>&1
  [ -f "$R/${B}${suffix}_full_r1500/run.json" ] || $PY scripts/final_train_predict.py $COMMON --seed "$seed" \
    --extra arrtaxi deptaxi inb refile nmw --name "${B}${suffix}_full_r1500" > "logs/fallback${suffix}.log" 2>&1
done
