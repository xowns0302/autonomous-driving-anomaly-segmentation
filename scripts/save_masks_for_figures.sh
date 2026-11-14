#!/usr/bin/env bash
# Phase 4: save DetSeg-S masks of 4 methods on 4 datasets for qualitative figures. Run from external/DetSeg-official.
export PYTHONPATH=.
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=../../results/A3/bars_things.json
BANK=../../results/P1/prototypes.npz; O=../../results/F1/masks; L=work_dirs/F1; mkdir -p $O $L
for spec in "fs_lostfound:FS_LostFound" "road_anomaly:RoadAnomaly" "smiyc_obstacle:SMIYC_Obstacle_val" "smiyc_anomaly:SMIYC_Anomaly_val"; do ds=${spec%%:*}; md=${spec#*:}
  for m in released paper c5 c8; do
    case $m in
      released) X="--bars released";;
      paper) X="--bars $BARS --stat bar_min";;
      c5) X="--bars $BARS --stat bar_min --m2a-dir ../../results/score_maps/m2a/$md --verify-t 0.5";;
      c8) X="--bars $BARS --stat bar_min --m2a-dir ../../results/score_maps/m2a/$md --verify-t 0.5 --prowl-bank $BANK --reject-t7 0.35";;
    esac
    conda run --no-capture-output -n detseg python ../../src/run_with_bars.py $X --fast --save-masks $O/$ds/$m -- ../../configs/detseg-s_$ds.py $CK --work-dir $L/${m}_$ds > $L/${m}_$ds.log 2>&1
    echo "$ds $m EXIT $? $(grep -o 'IoU: [0-9.]*' $L/${m}_$ds.log | tail -1)"
  done
done
