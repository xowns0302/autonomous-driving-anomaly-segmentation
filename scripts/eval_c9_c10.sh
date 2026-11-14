#!/usr/bin/env bash
# I7 held-out evaluation (1 run) with the values chosen on the tuning sets, plus mask saving for F2.
# usage (from external/DetSeg-official): bash ../../scripts/eval_c9_c10.sh <v for C9> <t7m for C10>
export PYTHONPATH=.
V=$1; T=$2
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=../../results/A3/bars_things.json
BANK=../../results/P1/prototypes.npz; O=work_dirs/I7_eval; mkdir -p $O
C9="--reject-t7 0.35 --veto-v $V"; C10="--reject-mask-t7 $T"
ALL="fs_lostfound:FS_LostFound road_anomaly:RoadAnomaly smiyc_obstacle:SMIYC_Obstacle_val smiyc_anomaly:SMIYC_Anomaly_val"
s_run() {  # method ds md args...   DetSeg-S, saves masks for F1/F2
  m=$1; ds=$2; md=$3; shift 3; n=S_${m}_$ds
  conda run --no-capture-output -n detseg python ../../src/run_with_bars.py --bars $BARS --stat bar_min \
    --m2a-dir ../../results/score_maps/m2a/$md --verify-t 0.5 --prowl-bank $BANK "$@" \
    --save-masks ../../results/F1/masks/$ds/$m -- ../../configs/detseg-s_$ds.py $CK --work-dir $O/$n > $O/$n.log 2>&1
  echo "$n EXIT $? $(grep -o 'IoU: [0-9.]*' $O/$n.log | tail -1)"
}
r_run() {  # DetSeg-R (C9 only)
  ds=$1; md=$2; n=R_c9_$ds
  conda run --no-capture-output -n detseg python ../../src/run_with_bars.py --bars $BARS --stat bar_min \
    --m2a-dir ../../results/score_maps/m2a/$md --verify-t 0.5 --prowl-bank $BANK $C9 \
    -- ../../configs/detseg-r_m2a_$ds.py $CK --work-dir $O/$n > $O/$n.log 2>&1
  echo "$n EXIT $? $(grep -o 'pixel/AUPRC: [0-9.]*  pixel/FPR@95TPR: [0-9.]*  pixel/AUROC: [0-9.]*  oracle/BestIoU: [0-9.]*' $O/$n.log | tail -1)"
}
for s in $ALL; do s_run c9 ${s%%:*} ${s#*:} $C9; done
for s in $ALL; do s_run c10 ${s%%:*} ${s#*:} $C10; done
for s in $ALL; do r_run ${s%%:*} ${s#*:}; done
