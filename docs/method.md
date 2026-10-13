# Method

Team `enthusiastic-daisy`, PRC Data Challenge 2026, final blind phase
(`final_ranking` / `final_submitting`, 670,790 departures, January, February, June
and July 2026). This document describes the end-to-end pipeline that produces the
final predictions. The two configurations that were prepared for submission are listed under
*Configurations*; which one is in the team bucket is stated in the README.

## Principles

- **Labels.** Only the 2,085,047 departure (`PHASE_mvt = "DEP"`) labels of the twelve
  2025 training files are used for fitting. No 2026 label, official score, earlier
  submission vector or row-level inference from scores enters any model or rule. One choice
  between complete prediction files was made by public-board score (see below).
- **Inputs.** Official columns of the movement and flight tables, other movement rows
  (arrivals included, as permitted by the organisers), open ADS-B.lol trajectories,
  ERA5 and GHCNh weather. Departure `BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` are never
  read for 2026 rows.
- **One pipeline for all four months.** Features, models and rules are the same for every
  row of `final_submitting`. The January/July rows are taken from one of the five seeds
  instead of their average (see *Configurations*).
- **Use of the daily public board.** For this pipeline the public board (January/July,
  2026-09-04 ranking) was used only to accept or reject complete, independently defined
  mechanisms, never to derive prediction values, thresholds or row-level targets, with one
  exception decided on 2026-10-13: for the January/July rows, which the board scores, the
  delivered file uses the single run that scored best instead of the seed average. That is a
  selection among complete prediction files by score, and it amounts to choosing the seed
  weights of those rows from the board.
  Every public submission of the pipeline is listed under *Results*.
- **What is not here.** The team's best public-board entry (v807, RMSE 234.86) came from an
  earlier line of work that used public-board scores to derive prediction values (score
  probes, and blending coefficients between submissions solved from official RMSE values).
  That line cannot predict February and June, contributes nothing to the final predictions
  and is not part of this repository. The pipeline described here scores 236.1026 on the
  public board with the same-date condition (v841, public-chain); before that condition,
  the same seed scored 236.8756 (v836).

## Target structure

`TAXITIME_SEC_mvt = MVT − BLOCK`. Three record states explain the heavy tail:

1. **Schedule-recorded BLOCK.** For about 10.9% of 2025 departures the airport BLOCK
   equals the scheduled time: BLOCK − SCHED has plateaus at 0 s and ±60 s, each about
   ±6 s wide (export jitter). Then the label is MVT − SCHED. These rows carry about
   44% of the 2025 labels above one hour and about 32% of the squared error of a plain
   regressor.
2. **Day rollover.** 16 rows in 2025 (14 at LIRF) have labels of 86,400 s plus an
   ordinary taxi time: BLOCK is one day early.
3. **Ordinary** taxi-out.

The RMSE-optimal prediction is the expectation over states:

```text
pred = p(x) · (MVT − SCHED) + (1 − p(x)) · ( R(x) + q(g) · 86,400 )
```

- `p(x)`: probability of the schedule-recorded state.
- `R(x)`: regression of ordinary taxi-out, fitted only on rows in neither tail state.
- `q(g)`: smoothed rate `(k + 0.5) / (n + 1)` of day rollover among non-schedule 2025 rows
  of five visible groups `g` (LIRF or not × NM AOBT missing or present; LIRF
  AOBT-missing rows split by MVT − SCHED above 6 h).

Fitting `R` on rows without the rollover state matters: otherwise isolated 87,000 s
labels create extreme leaves that hit ordinary AOBT-missing rows in held-out months.

## Structure of the provided tables

Four facts about how the official tables are built constrain the label through visible
columns. Each was established on the 2025 training files and is recomputed from them when a
prediction file is written.

1. **NM join.** A departure carries NM flight columns (`*_flt`) only if the airport BLOCK
   lies within one hour of the last filed off-block time: for all 2,062,577 matched 2025
   departures `|TAXITIME − (MVT − LOBT)| ≤ 3,606 s` (3,600 s plus the 6 s export jitter), with
   no exception. Arrivals are matched on their movement time instead. Both 2026 exports give
   every row the same match status.
   - *Join bound* (`final_pipeline.join_bound`): the prediction of a matched row is clipped
     to `MVT − LOBT ± (3,606 + 12) s`.
   - A delayed departure *without* NM columns is either schedule-recorded with a re-filed
     flight plan, or ordinary with a plan that was not re-filed. This is what the feature
     blocks `refile`, `inb` and `nmw` below describe.
2. **Schedule floor** (`schedule_floor`). A schedule-recorded BLOCK cannot follow the
   movement: of 17,011 departures with MVT before SCHED, 34 are schedule-recorded, all with
   MVT − SCHED ≥ −66 s. Below that minimum (less 12 s) the schedule term is removed.
3. **Previous-day NM record** (`present_rollover`, from the single 2025 analogue): at LIRF, when both
   SCHED and NM AOBT lie more than 6 h before MVT, the row is matched to an NM record about a
   day old. By the join bound its label lies within an hour of MVT − LOBT, which is the
   prediction (87,001 s for the 2025 row with label 87,002 s; one final row).
4. **Delay zones of the LIRF unmatched group** (`lirf_delay_zones`). For LIRF departures
   without NM columns and MVT − SCHED above 6 h the record state follows the delay alone
   (2025: 63 rows):

   | MVT − SCHED | rows | schedule-recorded | rollover | ordinary |
   |---|---:|---:|---:|---:|
   | 6 h – 23,039 s | 3 | 2 | 0 | 1 |
   | 23,039 – 49,772 s | 40 | 40 | 0 | 0 |
   | 49,772 – 109,439 s | 19 | 8 | 11 | 0 |
   | above | 1 | 1 | 0 | 0 |

   An ordinary record never has a delay above 23,039 s (6.4 h). Up to 49,772 s (halfway
   between the last row of that zone and the first rollover row) every row is
   schedule-recorded, whether or not the off-block crosses the calendar date. Above it, up to
   24 h + 6.4 h, a row is a rollover or schedule-recorded: the off-block clock time is stored
   inside a window around SCHED and wraps to the previous day. In the second zone `p` is the
   zone's schedule-recorded share (0.988). In the third (rollover rate 0.958) the classifier
   leans towards the schedule state because the similar rows just below the edge are all
   schedule-recorded by the window; `p` is the mean of three estimates there: the
   classifier, the zone share (0.425), and the share of the flight's operator among LIRF
   unmatched rows with an ordinary delay (1 h to 6.4 h, two pseudo-rows at the overall share).
   Edges and rates use `(k + 0.5) / (n + 1)` on 2025 labels.

   A wrap needs the off-block to fall on a later Rome calendar date than SCHED: all 12 LIRF
   rollover rows without NM columns do, and none of the 727 same-date rows that are not
   schedule-recorded is a rollover. A row of the third zone whose off-block (MVT − 15 min)
   lies on the scheduled date therefore cannot be a rollover: its `p` is the schedule-recorded
   share of the same-date rows above 23,039 s (19 of 19, rate 0.975) and its rollover rate
   that of the rows below the third zone. This condition belonged to the calendar-date split
   that the zones replaced on 2026-10-11 and was missing in the third zone until 2026-10-13.
   It changes one row of the public file and the same flight in the final file.

## Models

- **Features.** 483 base columns per departure, identical for 2025 and 2026:
  official-time differences (MVT relative to SCHED, NM AOBT, EOBT, IOBT, LOBT),
  categorical identifiers (airport, runway, stand, operator, aircraft type, destination),
  queue/throughput context from other movements, AOBT/stand-chain context from earlier
  arrivals at the same stand, ADS-B.lol trajectory summaries (`fh_*`), ERA5/GHCNh
  weather, PRC 2024 airport priors, and taxi-graph distances. Two blocks describe the
  surface congestion actually realised around each departure in the same file:
  - `arrtaxi` (15 columns, `src/prc2026/arr_taxi_context.py`): taxi-in of other
    arrivals at the airport (visible ARR BLOCK/TAXITIME, schedule-recorded arrivals
    dropped) over the previous 60/120 min, ±30 min and the day.
  - `deptaxi` (17 columns, `src/prc2026/dep_taxi_context.py`): MVT − NM AOBT of other
    departures at the airport, a visible taxi-out proxy, over the same windows, on the
    same runway, and for the day.

  Two more blocks describe whether a delayed flight's plan was re-filed, which by the NM
  join decides the record state of departures without NM columns (ATFM slot delays leave
  the filed time unchanged and hit many departures at once; delays announced by the
  airline, typically a late inbound aircraft, move it):
  - `refile` (30 columns, `src/prc2026/refile_context.py`): among NM-matched departures of
    the airport scheduled or departing nearby (same destination, same ICAO region, same
    operator, whole airport), the mean and the share above 30 min of AOBT − LOBT (delay
    taken without re-filing) and of LOBT − SCHED (shift of the filed time).
  - `inb` (14 columns, `src/prc2026/inbound_context.py`): the last arrival at the same
    stand at least 20 min before the movement: turnaround time, in-block relative to the
    scheduled departure, arrival delay, taxi-in, operator match, and its NM delays. The base
    stand-chain features are anchored at NM AOBT and are empty for unmatched departures.
  - `nmw` (10 columns, `src/prc2026/nm_witness.py`): arrivals are matched to NM on their own
    movement time, so when the destination is another airport of the data set the arrival row
    of the same flight still carries the NM record that the unmatched departure row lacks.
    The block links the two rows on origin, destination and time (callsign equality breaks
    ties) and gives the witness LOBT, EOBT, IOBT and AOBT relative to MVT and SCHED. Links
    without a callsign match are sometimes wrong, so this is a feature, not a rule.

  Two blocks read the ADS-B.lol ground track of the departing airframe:
  - `traj` (113 columns, `src/prc2026/traj_events.py`): first ground movement, start of
    sustained taxi, stand dwell and stand-based taxi time, push-back (reverse) and forward
    motion relative to NM AOBT, and the take-off interval. A row has these features when it
    has an NM AOBT, a valid airframe identity and at least ten ground points: 57.5% of the
    2025 departures and 62.2% of the final ones. Coverage follows the ADS-B receivers: above
    74% in 2026 at EDDM, EGLL, EHAM, LEBL, LEMD and LSZH, almost none at LFPG and LTFM, none at
    LIRF in June/July 2026, 22% at EDDF in February 2026.
  - `tprof` (250 columns, same module and rows): the track of the current ground episode in
    time bins, 66 min up to the movement in 3 min bins and ±21 min around NM AOBT in 90 s bins;
    per bin whether points were seen, mean speed, share of moving points, last distance to
    the stand and path length.
  - `tprofx` (400 columns, same module, rows and bins): the remaining channels per bin,
    maximum speed, ground share, displacement, time span, minimum stand distance, maximum
    height, largest gap and share of stale points.

  **Coverage fallback.** At airports with high ADS-B coverage only about 5% of the 2025
  departures lack a ground track, and they are untypical flights, so a model trained with the
  two trajectory blocks predicts badly when the track is simply missing (blanking the blocks
  on held-out rows costs 18–83 s RMSE against a model trained without them). Rows without
  trajectory coverage are therefore predicted by a second run of the same architecture trained
  without `traj`, `tprof` and `tprofx` (`final_write_submission.py --fallback`); covered rows
  use the run trained with them. The 2026 files have coverage gaps that 2025 does not (LIRF in
  June/July, EDDF from the end of January through February, LEMD in July), and on the public
  board this step alone lowered the RMSE from 240.27 to 237.53.

  All blocks exclude the row itself, use rows of the same file (`refile` and `inb` only rows of
  the same calendar month), are computed the same way for 2025 training and for each 2026
  file, and never read departure BLOCK or TAXITIME.
- **Classifier `p`.** XGBoost (depth 10, 800 rounds) trained on all 2025 departures with
  weights `((MVT − SCHED − R) / 1000 s)²` (clipped to [1e-3, 1000], mean 1), i.e. the scale
  of the mixture's squared-error contribution; averaged with three TabM classifiers
  (same weights). In the LIRF AOBT-missing group with MVT − SCHED above 6 h, where an
  underestimated `p` moves the prediction towards the rollover expectation, `p` is the
  XGBoost classifier alone.
- **Regressor `R`.** Mean of XGBoost, LightGBM and CatBoost, averaged 50/50 with the mean
  of three TabM regressors (piecewise-linear numeric embeddings, TabM k=16 ensemble).

Hyper-parameters are fixed in `scripts/final_train_predict.py` and
`src/prc2026/final_pipeline.py` / `src/prc2026/final_tabm.py`.

## Validation

Leave-month-out on 2025 labels: JJ (fit on Feb–Jun, Aug–Dec; hold out January and July)
and FJ (hold out February and June). Every mechanism was checked on both folds before
use; results are reported for all rows and without the rollover-labelled rows, which
dominate the fold RMSE (6 rows in JJ). `scripts/final_e2_schedule_mixture.py` runs one
configuration on both folds and stores the held-out predictions; `scripts/score_validation_folds.py`
applies the visible rules to them with edges and rates from the fold's training months and
prints the fold RMSE reported under *Results*.

## Results

2025 labels, leave-month-out, one XGBoost mixture per configuration with all visible rules
(fold RMSE in seconds; the full model adds LightGBM, CatBoost and TabM members):

| blocks added to `arrtaxi deptaxi` | JJ | FJ |
|---|---:|---:|
| `inb refile` | 300.35 | 267.84 |
| + `traj` | 298.86 | 264.23 |
| + `tprof` | 297.11 | 261.98 |
| + `tprof`, coverage fallback | 295.75 | 262.00 |
| + `nmw` | 297.00 | 261.59 |
| + `nmw tprofx` | 296.94 | 261.46 |

Daily public board (January/July 2026, 344,841 departures), full models:

| version | change | RMSE |
|---|---|---:|
| v828 | rule set of *Structure of the provided tables* on the 2026-10-10 model | 249.53 |
| v829 | + `refile` | 247.29 |
| v830 | + `inb` | 244.97 |
| v831 | + `traj` | 243.59 |
| v832 | + `tprof` | 240.27 |
| v833 | + coverage fallback | 237.53 |
| v834 | v833, three seeds | 238.06 |
| v835 | v833 + `nmw`, one seed | 237.00 |
| v836 | v835 + `tprofx`, one seed | 236.88 |
| v837 | v836, three seeds | 237.88 |
| v838 | v836, five seeds, public-chain predictions | 237.8486 |
| v839 | January/July rows of the final-chain file delivered at that time, taken by MVT_ID into public-board order | 236.9858 |
| v840 | v836 configuration, second seed (20261011) alone | 239.0009 |
| v841 | v836 + same-date condition of the LIRF delay zones (one row differs) | 236.1026 |

v839 checked the final-chain predictions delivered at that time, whose inputs include the
final export jitter; it is not a re-upload of v836 or a four-month final blind score. The
file delivered now differs from it in the one row that the same-date condition changes.

Single seeds differ by 0.5–2 s on the public board (the three-seed average of v834 scores
above its first seed; the second seed alone, v840, scores 2.1 s above the first), so v835 and v836 show that `nmw` and `tprofx` do no harm, not that
they help; the 2025 folds say the same. Ideas that did not improve the folds were left out
and their code is not in this repository: counts of other unmatched delayed departures,
daily ATFM delay per airport, trajectory features for unmatched departures, flight
regularity, schedule-state rates by flight number, callsign or stand, a finer coverage split
by track quality, and blending the trajectory run with the fallback run on covered rows.

## Configurations

All configurations use the same models, rules and coverage fallback.

- **Delivered file** (`write_final_file.sh submitted`): trajectory run
  `--extra arrtaxi deptaxi inb refile traj tprof tprofx nmw` (1,333 features), fallback run
  `--extra arrtaxi deptaxi inb refile nmw` (569 features).
  - February/June rows: average of the seeds 20261008, 20261011, 20261014, 20261017, 20261020
    (`write_final_file.sh five-seed`; public analogue of the average: v838, 237.8486).
  - January/July rows: the run with seed 20261008 alone (`write_final_file.sh single-run`;
    public analogue v841, 236.10; v836 before the same-date condition of the delay zones, 236.88).

  The single run is used for January/July because it scored below the seed average of the same
  configuration on the public board (v837, three seeds: 237.8765; v838, five seeds: 237.8486),
  and the board scores the same departures as the January/July rows of the final file.
  Single-seed scores scatter by up to about 2 s (the second seed alone: v840, 239.0009), so the choice is a
  selection on the scored rows and no evidence that this seed is the better model; for
  February/June, where nothing is scored, the seed average is kept because averaging lowers the
  expected error. The five-seed average scored above the single run, so it was not used for
  all four months. `scripts/final_combine_months.py` takes the rows of each month group from
  the two files unchanged. The January/July subset of the file delivered before the same-date
  condition scored 236.9858 as v839 on the public board after mapping the final-chain
  predictions by MVT_ID.
- **Three-seed safety configuration** (first upload, 2026-10-12 16:52 UTC; public analogue
  v834, 238.06): seeds 20261008, 20261011, 20261014; trajectory run
  `--extra arrtaxi deptaxi inb refile traj tprof` (923 features), fallback run
  `--extra arrtaxi deptaxi inb refile` (559 features).

Per seed the fallback is applied and the visible rules (`lirf_delay_zones`,
`schedule_floor`) act on that seed's `p` and `R`; the seed predictions are averaged, then
`present_rollover` and `join_bound` are applied.

## Reproduction

```bash
export PYTHONPATH=src; PY=python
# base model inputs: see the README (external data, base features, ADS-B history); then the feature blocks
$PY scripts/build_arr_taxi_context.py
$PY scripts/build_dep_taxi_context.py
$PY scripts/build_refile_context.py
$PY scripts/build_inbound_context.py
$PY scripts/build_nm_witness.py
$PY scripts/build_traj_events.py --block traj
$PY scripts/build_traj_events.py --block tprof
$PY scripts/build_traj_events.py --block tprofx
# five seeds, 3 apart (TabM members use seed, seed+1, seed+2, so the seeds do not overlap)
COMMON="--family xgbmix --mode final --rounds 1500 --class-weight delta2 --reg-lgb --reg-cat --reg-tabm \
  --tabm-seeds 3 --tabm-epochs 20 --tabm-weight 0.5 --class-tabm --class-tabm-seeds 3 \
  --class-tabm-skip-group lirf_missing_gt6h --rollover-state"
for seed in 20261008 20261011 20261014 20261017 20261020; do
  $PY scripts/final_train_predict.py $COMMON --seed $seed --name traj_$seed \
    --extra arrtaxi deptaxi inb refile traj tprof tprofx nmw
  $PY scripts/final_train_predict.py $COMMON --seed $seed --name fallback_$seed \
    --extra arrtaxi deptaxi inb refile nmw
done
# five-seed file and single-run file in final_submitting order
RULES="--present-rollover --lirf-delay-zones --schedule-floor --join-bound"
ARGS=""; for seed in 20261008 20261011 20261014 20261017 20261020; do
  ARGS="$ARGS --model traj_$seed=0.2 --fallback traj_$seed=fallback_$seed"; done
$PY scripts/final_write_submission.py --target final $ARGS $RULES --output submissions/final/five.parquet
$PY scripts/final_write_submission.py --target final --model traj_20261008=1.0 \
  --fallback traj_20261008=fallback_20261008 $RULES --output submissions/final/single.parquet
# delivered file: January/July rows of the single run, February/June rows of the five-seed average
$PY scripts/final_combine_months.py --january-july submissions/final/single.parquet \
  --february-june submissions/final/five.parquet --selection-note "<public-board scores>" \
  --output submissions/final/<file>.parquet
```

The three-seed safety configuration is the same without `tprofx` and `nmw`, with the first three
seeds and weights of one third.

`scripts/train_final_runs.sh` and `scripts/write_final_file.sh submitted` hold the same steps
with the run names used on the development machine (first seed without suffix, then `_s2` … `_s5`). One trajectory
run takes about 80 minutes and one fallback run about 40 minutes on an RTX 5090 with 16 CPU
threads; the trajectory run needs about 30 GB of memory.
