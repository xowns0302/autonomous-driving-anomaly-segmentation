#!/usr/bin/env bash
# Latency baseline (T2). Same protocol must be used for every later comparison:
#   4 datasets x {M2A, DetSeg-S released, DetSeg-S paper bars, DetSeg-S + C5, DetSeg-R, DetSeg-R + C5}
#   30 timed images (small datasets cycled) after 5 warm-up images, 3 repeats, batch 1,
#   nothing else running on the GPU.
# Usage (repo root): bash scripts/latency_baseline.sh results/T2
set -u
OUT=${1:-results/T2}
ROOT=$(pwd)
mkdir -p "$OUT"
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth
BARS=$ROOT/results/A3/bars_things.json
N="--images 30 --warmup 5 --repeats 3"

{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu | grep "Model name"; date; git rev-parse --short HEAD; } > "$OUT/environment.txt"

for spec in "fs_lostfound:FS_LostFound:images/*.png" "road_anomaly:RoadAnomaly:frames/*.webp" \
            "smiyc_obstacle:SMIYC_Obstacle_val:images/*.webp" "smiyc_anomaly:SMIYC_Anomaly_val:images/*.png"; do
  IFS=: read ds md glob <<< "$spec"
  M2A=$ROOT/results/score_maps/m2a/$md
  (cd external/DetSeg-official && export PYTHONPATH=. && \
   conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-s_$ds.py --checkpoint $CK $N --out $ROOT/$OUT/S_released_$ds.json > $ROOT/$OUT/S_released_$ds.log 2>&1
   conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-s_$ds.py --checkpoint $CK $N --bars $BARS --out $ROOT/$OUT/S_paper_$ds.json > $ROOT/$OUT/S_paper_$ds.log 2>&1
   conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-s_$ds.py --checkpoint $CK $N --bars $BARS --m2a-dir $M2A --verify-t 0.5 --out $ROOT/$OUT/S_c5_$ds.json > $ROOT/$OUT/S_c5_$ds.log 2>&1
   conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-r_m2a_$ds.py --checkpoint $CK $N --out $ROOT/$OUT/R_released_$ds.json > $ROOT/$OUT/R_released_$ds.log 2>&1
   conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-r_m2a_$ds.py --checkpoint $CK $N --bars $BARS --m2a-dir $M2A --verify-t 0.5 --out $ROOT/$OUT/R_c5_$ds.json > $ROOT/$OUT/R_c5_$ds.log 2>&1)
  conda run --no-capture-output -n m2a python src/profile_m2a_latency.py --m2a-repo external/Mask2Anomaly \
      --weights ~/datasets/m2a_validation_dl/Mask2Anomaly/best_contrastive.pth \
      --images "external/DetSeg-official/data/$md/$glob" --n 30 --warmup 5 --repeats 3 --out $OUT/M2A_$ds.json > $OUT/M2A_$ds.log 2>&1
  echo "$ds done"
done
