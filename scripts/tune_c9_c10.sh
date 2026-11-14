#!/usr/bin/env bash
# I7: C9 (C8 + M2A veto) and C10 (C5 + mask-level C7) on FS L&F + RA (tuning only).
# Plan: docs/05_improvement_plan.md §9. Run from external/DetSeg-official.
export PYTHONPATH=.
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=../../results/A3/bars_things.json
BANK=../../results/P1/prototypes.npz; O=work_dirs/I7_tune; mkdir -p $O
run() {  # name ds md extra-args
  n=$1; ds=$2; md=$3; shift 3
  conda run --no-capture-output -n detseg python ../../src/run_with_bars.py --bars $BARS --stat bar_min \
    --m2a-dir ../../results/score_maps/m2a/$md --verify-t 0.5 --prowl-bank $BANK "$@" \
    -- ../../configs/detseg-s_$ds.py $CK --work-dir $O/$n > $O/$n.log 2>&1
  echo "$n EXIT $? $(grep -o 'IoU: [0-9.]*' $O/$n.log | tail -1)"
}
SETS="fs_lostfound:FS_LostFound road_anomaly:RoadAnomaly"
# implementation checks (FS L&F): must equal C8 38.44 and C5 32.67
run sanity_c9_v1.1_fs_lostfound fs_lostfound FS_LostFound --reject-t7 0.35 --veto-v 1.1
# done separately: run sanity_c10_t-1_fs_lostfound fs_lostfound FS_LostFound --reject-mask-t7 -1
for v in 0.5 0.6 0.7 0.8; do for s in $SETS; do run c9_v${v}_${s%%:*} ${s%%:*} ${s#*:} --reject-t7 0.35 --veto-v $v; done; done
for t in 0.25 0.30 0.35 0.40; do for s in $SETS; do run c10_t${t}_${s%%:*} ${s%%:*} ${s#*:} --reject-mask-t7 $t; done; done
