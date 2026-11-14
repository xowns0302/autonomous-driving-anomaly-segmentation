#!/usr/bin/env bash
# C8 (= C5 + C7) tuning on FS L&F + RA. Run from external/DetSeg-official.
export PYTHONPATH=.
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=../../results/A3/bars_things.json
BANK=../../results/P1/prototypes.npz; O=work_dirs/I6_tune; mkdir -p $O
for t7 in 0.25 0.30 0.35; do for spec in "fs_lostfound:FS_LostFound" "road_anomaly:RoadAnomaly"; do ds=${spec%%:*}; md=${spec#*:}; n=c8_t7_${t7}_$ds
  conda run --no-capture-output -n detseg python ../../src/run_with_bars.py --bars $BARS --stat bar_min \
    --m2a-dir ../../results/score_maps/m2a/$md --verify-t 0.5 --prowl-bank $BANK --reject-t7 $t7 \
    -- ../../configs/detseg-s_$ds.py $CK --work-dir $O/$n > $O/$n.log 2>&1
  echo "$n EXIT $? $(grep -o 'IoU: [0-9.]*' $O/$n.log | tail -1)"; done; done
