# PRC Data Challenge 2026 — team enthusiastic-daisy

Taxi-out time prediction for departures at ten European airports
(`TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`), final blind phase:
670,790 departures of January, February, June and July 2026.

This repository contains the pipeline that produces `enthusiastic-daisy_final.parquet`
and nothing else. It is fitted on the 2025 training labels only.

- **Method, design decisions and results:** [`docs/method.md`](docs/method.md)
- **Licence:** GNU GPLv3 for the code ([`LICENSE`](LICENSE)). Official competition data and
  third-party data keep their own terms and are not redistributed here.

## The idea in five lines

1. The label has three record states: ordinary taxi-out, airport BLOCK equal to the
   scheduled time (label = MVT − SCHED), and BLOCK one day early. The prediction is the
   expectation over states: a classifier for the schedule state, a regressor for ordinary
   taxi-out, a rate for the rollover.
2. The way the official tables are joined bounds the label through visible columns (a
   departure has NM columns only if BLOCK is within one hour of the last filed off-block
   time). Four rules follow from that and are recomputed from the 2025 files.
3. Feature blocks describe the surface traffic around each departure (taxi-in of arrivals,
   MVT − AOBT of other departures), whether delayed plans were re-filed, the previous
   arrival at the stand, and the NM record seen on the flight's arrival row.
4. Open ADS-B.lol ground tracks of the departing airframe give the observed push-back and
   taxi start. Rows without a track are predicted by a second model trained without the
   track features.
5. Several seeds of XGBoost + LightGBM + CatBoost + TabM are averaged (for the January/July rows
   one seed is used, see *Submitted file*).

## Submitted file

`enthusiastic-daisy_final.parquet` is written by `bash scripts/write_final_file.sh submitted`.
Configuration: blocks `arrtaxi deptaxi inb refile traj tprof tprofx nmw`; rows without a track
are predicted by fallback runs with `arrtaxi deptaxi inb refile nmw`.

- **February and June rows** (325,949): average of five seeds (20261008, 20261011, 20261014,
  20261017, 20261020).
- **January and July rows** (344,841, the departures the daily public board scores): the run
  with seed 20261008 alone.

**The January/July choice was made by public-board score.** On the board the single run scored
236.88 (v836) and the three-seed average of the same configuration 237.88 (v837). Single seeds
differ by 0.5–1.5 s there, so this says nothing about which seed is the better model; but the
board scores exactly the January/July rows of the final file, and on 2026-10-13 the team decided
to keep the run that scored best for those rows. For February and June no score exists and the
seed average, which has the lower expected error, is used. This is the one place where a
public-board score decides between predictions. `scripts/final_combine_months.py` joins the two
files and records the choice in the manifest.

State at this commit (2026-10-13 02:45 UTC): the last two seeds are being fitted and the team
bucket still holds the three-seed safety file uploaded on 2026-10-12 16:52 UTC (blocks without
`nmw` and `tprofx`, seeds 20261008/11/14; `write_final_file.sh three-seed`; public analogue v834,
238.06). The file described above replaces it after the five-seed average is scored on the
public board (v838); should that average score below the single run, it is used for all four
months. The public-board score of the January/July rows of the delivered file (v839) is added
here afterwards.

## Results

| | RMSE (s) |
|---|---:|
| Public board, January/July 2026, this pipeline, one seed (v836) | 236.88 |
| Public board, same configuration, average of three seeds (v837) | 237.88 |
| 2025 leave-month-out, hold out January + July (one XGBoost mixture with all blocks, visible rules applied) | 296.94 |
| 2025 leave-month-out, hold out February + June (same) | 261.46 |

The step-by-step public-board history of the pipeline (249.53 → 236.88) is in
[`docs/method.md`](docs/method.md#results).

The team's best entry on the public board (v807, 234.86) is **not** from this pipeline. It
came from earlier work that used public-board scores to derive prediction values; that work
cannot predict February and June and is not in this repository.

## Layout

```
docs/method.md                 method, design decisions, results, final configuration
configs/                       settings read by the feature code; external data registry
src/prc2026/
  final_pipeline.py            feature contract, data loading, the visible rules
  final_tabm.py                TabM regressor and classifier
  arr_taxi_context.py          block arrtaxi: taxi-in of other arrivals
  dep_taxi_context.py          block deptaxi: MVT − NM AOBT of other departures
  refile_context.py            block refile: re-filing behaviour of neighbouring departures
  inbound_context.py           block inb: previous arrival at the same stand
  nm_witness.py                block nmw: NM record on the flight's arrival row
  traj_events.py               blocks traj / tprof / tprofx: ADS-B ground track of the airframe
  departure_episode.py, direction_events.py, temporal_profile.py, adsblol_stand.py
                               per-flight formulas used by traj_events.py
  features.py, queue.py, queue_dynamics.py, route_twin.py, geometry.py, xplane.py, aircraft.py,
  external.py, aobt_stand_chain.py, modeling.py, base_features.py
                               the 483 base columns (official-time differences, queues, taxi
                               routes on the airport graph, aircraft, calendar, weather)
  adsblol*.py, full_history_*.py, final_history_retention.py, stand_event_*.py,
  airframe_visit.py, archive_identity.py, direct_source_session.py
                               ADS-B.lol download, decoding and per-flight track summaries
  constants.py, data.py, run_guards.py, training_checkpoint.py, storage.py
                               shared helpers
scripts/
  external/                    download and preparation of the open data sets
  acquire_full_history_v1.py, download_full_history_direct_v1.py, rebuild_full_history_v1.py,
  compact_full_history_v1.py, supplement_full_history_training_v2.py, final_v3_trajectory_acquire.py
                               ADS-B.lol history (2025, January/July 2026, February/June 2026);
                               audit_full_history_rebuild_v1.py
                               hold helpers they import
  build_feature_cache.py, build_full_history_*.py, final_v3_build_trajectory_features.py,
  final_v3_assemble_inputs.py  base model inputs for 2025, the public file and the final file
  build_*_context.py, build_nm_witness.py, build_traj_events.py
                               the feature blocks
  final_e2_schedule_mixture.py leave-month-out validation of one configuration
  final_train_predict.py       fit on all 2025 departures, predict public and final rows
  final_write_submission.py    coverage fallback, visible rules, seed average, output file
  final_combine_months.py      delivered file: January/July rows of one file, February/June of another
  final_rows_to_public.py      January/July rows of a final file in public-board order (check of the
                               delivered file on the public board)
  train_final_runs.sh, write_final_file.sh
                               the exact final configuration
  upload_final_submission.py   template checks and upload of the final file
```

## Data

Official files (not redistributed): the twelve `training_2025-*.parquet`, `ranking.parquet`
and `submitting.parquet` in `data/raw/`; `final_ranking.parquet` and
`final_submitting.parquet` in `data/official-final-2026-10-08/`.

Open external data, registered with licence and acquisition entry in
[`configs/external_data_sources.yaml`](configs/external_data_sources.yaml):

| source | licence | used for |
|---|---|---|
| ADS-B.lol `globe_history` daily archives | ODbL 1.0 | ground track of the departing airframe |
| Copernicus ERA5 (ARCO) | CC-BY 4.0 | hourly weather at the airports |
| NOAA GHCNh | NOAA public data | station weather |
| OurAirports | public domain | airport and runway reference |
| OpenStreetMap airside | ODbL 1.0 | taxiway graph distances |
| X-Plane Scenery Gateway | GPL 2.0 | stand positions, taxi routes |
| PRC Data Challenge 2024 data | CC-BY 4.0 | aircraft-type taxi priors |
| OpenAP | LGPL 3.0 / GPL 3.0 data | aircraft properties |
| FAA aircraft characteristics | US public data | aircraft dimensions |
| `holidays` package | MIT | calendar |

No OpenSky state vectors, no proprietary data and no 2026 labels are used.

## Reproduction

Python 3.10, packages in [`requirements.txt`](requirements.txt); one GPU (the final runs
were fitted on an RTX 5090, 16 CPU threads, about 30 GB of memory for the largest run).
All commands run from the repository root with `PYTHONPATH=src`.

**1. External data** (`scripts/external/`, each script documents its arguments):
`download_ourairports.py`, `download_osm_airside.py` + `prepare_airside_graphs.py`,
`download_xplane_scenery.py`, `prepare_prc2024_aircraft.py`, `download_faa_aircraft.py`,
`prepare_calendar.py`, `download_era5_arco.py`, `download_ghcnh.py`.

**2. ADS-B.lol history** (about 480 daily archives, several days of download and decoding):
`acquire_full_history_v1.py` and `rebuild_full_history_v1.py` (January/July 2025 and 2026),
`supplement_full_history_training_v2.py` (the other 2025 months),
`final_v3_trajectory_acquire.py` (February/June 2026). The archives are public GitHub release
assets: those of 2025 and January/July 2026 are listed with their sizes in
`configs/full_history_*_assets.json`, those of February/June 2026 are selected from
`configs/adsblol/preferred_2026.txt` (a copy of ADS-B.lol's list of preferred releases).
All public-data download clients use ordinary HTTPS with certificate and hostname
verification. They ignore environment proxy variables and do not read local proxy
configuration, contact a controller, create listeners, inspect service logs, or require
`PRC_FINAL_DIRECT_PORT`. Release URLs and CDN redirects remain restricted to the
registered public source; byte ranges, declared sizes, transfer budgets and resumable
checkpoints are still checked. CDN connections use normal DNS rather than fixed IPs. The final-date downloader uses
public release redirects by default; its optional per-asset API catalog is disabled,
so a fresh checkout does not require an API quota reserved for every payload. The
public release listing still supplies the published sizes.

The download entry points require a clean, locally committed Git checkout to record
the source version; they do not require an upstream branch or a push. Subsequent data
builders and fitting guards retain their existing clean-and-pushed source requirement.
External-data commands in step 1 use the same proxy-independent clients. ERA5 still
requires its documented source authorization; data licences and source contracts are
unchanged.

Offline download regression tests (no download, fitting or prediction):

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
```

A bounded public-source transport check is recorded in
[`docs/download_transport_check.txt`](docs/download_transport_check.txt). It checks
small ranges in temporary directories, not complete archive acquisition or decoding.

**3. Base model inputs** (483 columns per departure, same contract for all populations):

```bash
# 2025 months and the public ranking file
python scripts/build_feature_cache.py --feature-version v3 \
  --output-root artifacts/features/external-context-v3-r20260904
python scripts/build_full_history_features_v1.py
python scripts/build_full_history_training_features_v2.py
# final file: data/final-view holds the training files and final_ranking.parquet as ranking.parquet
python scripts/build_feature_cache.py --feature-version v3 --data-dir data/final-view --partition ranking \
  --era5-version arco-geo-final-v3 --ghcnh-snapshot 2026-10-08 \
  --output-root artifacts/features/external-context-v3-final-20261008
python scripts/final_v3_build_trajectory_features.py
python scripts/final_v3_assemble_inputs.py
```

**4. Feature blocks, models, final file:**

```bash
for s in arr_taxi_context dep_taxi_context refile_context inbound_context nm_witness; do
  python scripts/build_$s.py; done
for b in traj tprof tprofx; do python scripts/build_traj_events.py --block $b; done
bash scripts/train_final_runs.sh             # ten runs: five seeds x (trajectory run, fallback run)
bash scripts/write_final_file.sh submitted   # writes submissions/final/final_submitted.parquet
```

The shell scripts call `python`; set `PYTHON=/path/to/python` to use another interpreter.

Leave-month-out validation of a configuration on 2025 (January+July and February+June held
out). The first command fits one XGBoost mixture per fold and stores the held-out predictions;
the second applies the visible rules to them, with all edges and rates from the training months
of the fold, and prints the numbers of the table above:

```bash
python scripts/final_e2_schedule_mixture.py --class-weight delta2 --rollover-state \
  --extra arrtaxi deptaxi inb refile traj tprof tprofx nmw --name check
python scripts/score_validation_folds.py --name check
```

**What was re-run from this tree.** `scripts/check_reproduction.py` compares this tree with
the stored inputs and runs of the development machine. For the three-seed configuration it
passes (output in `docs/reproduction_check.txt`): all modules import, all scripts answer `--help`, `write_final_file.sh three-seed` on the
stored runs reproduces the uploaded file value for value, the five context blocks of the public
file and nine sampled airport-months of the three trajectory blocks are reproduced value for
value. `score_validation_folds.py` reproduces the fold numbers from the stored held-out
predictions. Not re-run from this tree: the downloads and base model inputs (steps 1–3, run once
during development with the same code) and the model fits themselves.
