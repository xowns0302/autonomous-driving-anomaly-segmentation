#!/usr/bin/env bash
# C8 (C5 t=0.5 + C7 t7=0.35) held-out evaluation + DetSeg-R. Run from external/DetSeg-official.
export PYTHONPATH=.
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=../../results/A3/bars_things.json
BANK=../../results/P1/prototypes.npz; O=work_dirs/I6_eval; mkdir -p $O
R="--bars $BARS --stat bar_min --verify-t 0.5 --prowl-bank $BANK --reject-t7 0.35"
for spec in "smiyc_obstacle:SMIYC_Obstacle_val" "smiyc_anomaly:SMIYC_Anomaly_val"; do ds=${spec%%:*}; md=${spec#*:}; n=S_c8_$ds
  conda run --no-capture-output -n detseg python ../../src/run_with_bars.py $R --m2a-dir ../../results/score_maps/m2a/$md -- ../../configs/detseg-s_$ds.py $CK --work-dir $O/$n > $O/$n.log 2>&1
  echo "$n EXIT $? $(grep -o 'IoU: [0-9.]*' $O/$n.log | tail -1)"; done
for spec in "fs_lostfound:FS_LostFound" "road_anomaly:RoadAnomaly" "smiyc_obstacle:SMIYC_Obstacle_val" "smiyc_anomaly:SMIYC_Anomaly_val"; do ds=${spec%%:*}; md=${spec#*:}; n=R_c8_$ds
  conda run --no-capture-output -n detseg python ../../src/run_with_bars.py $R --m2a-dir ../../results/score_maps/m2a/$md -- ../../configs/detseg-r_m2a_$ds.py $CK --work-dir $O/$n > $O/$n.log 2>&1
  echo "$n EXIT $? $(grep -o 'pixel/AUPRC: [0-9.]*  pixel/FPR@95TPR: [0-9.]*  pixel/AUROC: [0-9.]*  oracle/BestIoU: [0-9.]*' $O/$n.log | tail -1)"; done
