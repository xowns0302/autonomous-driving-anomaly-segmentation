#!/usr/bin/env bash
# I5 tuning stage 3: best t6 from stage 1 (0.50) + each t7. Run from external/DetSeg-official.
export PYTHONPATH=.
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=../../results/A3/bars_things.json
BANK=../../results/P1/prototypes.npz; O=work_dirs/I5_tune; mkdir -p $O
for t7 in 0.25 0.30 0.35; do for ds in fs_lostfound road_anomaly; do n=c6_0.50_c7_${t7}_$ds
  conda run --no-capture-output -n detseg python ../../src/run_with_bars.py --bars $BARS --stat bar_min --prowl-bank $BANK \
    --protect-t6 0.50 --reject-t7 $t7 -- ../../configs/detseg-s_$ds.py $CK --work-dir $O/$n > $O/$n.log 2>&1
  echo "$n EXIT $? $(grep -o 'IoU: [0-9.]*' $O/$n.log | tail -1)"; done; done
