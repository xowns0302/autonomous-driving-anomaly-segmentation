#!/usr/bin/env bash
# New held-out set (plan docs/05 §10): Lost and Found test, 955 images, all settings frozen. Run once.
# Run from external/DetSeg-official. Masks go to results/F1/masks/laf_test/<method> for the F2 counts.
export PYTHONPATH=.
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=../../results/A3/bars_things.json
BANK=../../results/P1/prototypes.npz; M2A=../../results/score_maps/m2a/LAF_test
O=../../results/F1/masks/laf_test; L=work_dirs/LAF; mkdir -p $O $L
for m in released paper c5 c8 c9; do
  if grep -q 'IoU: [0-9]' $L/$m.log 2>/dev/null; then echo "laf_test $m already done $(grep -o 'IoU: [0-9.]*' $L/$m.log | tail -1)"; continue; fi
  case $m in
    released) X="--bars released";;
    paper) X="--bars $BARS --stat bar_min";;
    c5) X="--bars $BARS --stat bar_min --m2a-dir $M2A --verify-t 0.5";;
    c8) X="--bars $BARS --stat bar_min --m2a-dir $M2A --verify-t 0.5 --prowl-bank $BANK --reject-t7 0.35";;
    c9) X="--bars $BARS --stat bar_min --m2a-dir $M2A --verify-t 0.5 --prowl-bank $BANK --reject-t7 0.35 --veto-v 0.7";;
  esac
  conda run --no-capture-output -n detseg python ../../src/run_with_bars.py $X --fast --save-masks $O/$m -- ../../configs/detseg-s_laf_test.py $CK --work-dir $L/$m > $L/$m.log 2>&1
  echo "laf_test $m EXIT $? $(grep -o 'IoU: [0-9.]*' $L/$m.log | tail -1)"
done
