#!/usr/bin/env bash
# Adds C7 and C8 pipelines to a latency folder with the T2 protocol (30 images, 5 warm-up, 3 repeats).
# Usage (repo root): bash scripts/latency_c7_c8.sh results/T2
set -u
OUT=${1:-results/T2}; ROOT=$(pwd)
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=$ROOT/results/A3/bars_things.json
BANK=$ROOT/results/P1/prototypes.npz; N="--images 30 --warmup 5 --repeats 3 ${EXTRA:-}"
for spec in "fs_lostfound:FS_LostFound" "road_anomaly:RoadAnomaly" "smiyc_obstacle:SMIYC_Obstacle_val" "smiyc_anomaly:SMIYC_Anomaly_val"; do
  IFS=: read ds md <<< "$spec"; M2A=$ROOT/results/score_maps/m2a/$md
  (cd external/DetSeg-official && export PYTHONPATH=. && \
   conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-s_$ds.py --checkpoint $CK $N --bars $BARS --prowl-bank $BANK --reject-t7 0.35 --out $ROOT/$OUT/S_c7_$ds.json > $ROOT/$OUT/S_c7_$ds.log 2>&1
   conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-s_$ds.py --checkpoint $CK $N --bars $BARS --m2a-dir $M2A --verify-t 0.5 --prowl-bank $BANK --reject-t7 0.35 --out $ROOT/$OUT/S_c8_$ds.json > $ROOT/$OUT/S_c8_$ds.log 2>&1
   conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-r_m2a_$ds.py --checkpoint $CK $N --bars $BARS --m2a-dir $M2A --verify-t 0.5 --prowl-bank $BANK --reject-t7 0.35 --out $ROOT/$OUT/R_c8_$ds.json > $ROOT/$OUT/R_c8_$ds.log 2>&1)
  echo "$ds done"
done
