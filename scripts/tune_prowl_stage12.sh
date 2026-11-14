#!/usr/bin/env bash
# I5 tuning stages 1-2 (docs/05_improvement_plan.md section 7). Run from external/DetSeg-official.
export PYTHONPATH=.
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=../../results/A3/bars_things.json
BANK=../../results/P1/prototypes.npz; O=work_dirs/I5_tune; mkdir -p $O
run() { n=$1; shift; for ds in fs_lostfound road_anomaly; do
  conda run --no-capture-output -n detseg python ../../src/run_with_bars.py --bars $BARS --stat bar_min --prowl-bank $BANK "$@" \
    -- ../../configs/detseg-s_$ds.py $CK --work-dir $O/${n}_$ds > $O/${n}_$ds.log 2>&1
  echo "${n}_$ds EXIT $? $(grep -o 'IoU: [0-9.]*' $O/${n}_$ds.log | tail -1)"; done; }
run sanity --protect-t6 1.1 --reject-t7 -1
for t in 0.35 0.40 0.45 0.50; do run c6_$t --protect-t6 $t; done
for t in 0.25 0.30 0.35; do run c7_$t --reject-t7 $t; done
