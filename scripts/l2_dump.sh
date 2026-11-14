#!/usr/bin/env bash
# L2: dump per-frame boxes on the LAF test set for VPHM (public bars and C5). DetSeg-S config: DetSeg-R evaluation runs out of RAM on 955 images; boxes are the same.
# The final metric step on 955 images also runs out of RAM (it crashed WSL), so each run is stopped as soon as all 955 box files exist.
cd /home/taejun/projects/autonomous-driving-anomaly-segmentation; ROOT=$(pwd)
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=$ROOT/results/A3/bars_things.json; M2A=$ROOT/results/score_maps/m2a/LAF_test
cd external/DetSeg-official; export PYTHONPATH=.
dump() {  # $1 = output dir, $2 = log, rest = run_with_bars options
  out=$1; log=$2; shift 2
  [ "$(ls $out 2>/dev/null | wc -l)" -ge 955 ] && [ -f $out/.done ] && return
  rm -rf $out; mkdir -p $out
  setsid conda run --no-capture-output -n detseg python $ROOT/src/run_with_bars.py "$@" --fast --dump-boxes $out -- $ROOT/configs/detseg-s_laf_test.py $CK --work-dir work_dirs/L2/$(basename $out) > $log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ "$(ls $out | wc -l)" -ge 955 ]; then sleep 5; pkill -f -- "--dump-boxes $out"; break; fi
    sleep 10
  done
  wait $pid 2>/dev/null; [ "$(ls $out | wc -l)" -ge 955 ] && touch $out/.done
}
dump $ROOT/results/L2/boxes_released work_dirs/L2_released.log --bars released
echo "released dumped $(ls $ROOT/results/L2/boxes_released | wc -l)"
dump $ROOT/results/L2/boxes_c5 work_dirs/L2_c5.log --bars $BARS --stat bar_min --m2a-dir $M2A --verify-t 0.5
echo "c5 dumped $(ls $ROOT/results/L2/boxes_c5 | wc -l)"
