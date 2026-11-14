#!/usr/bin/env bash
# L1 (paper limitation "Efficiency"): FP16 detector. Accuracy (C5, 4 datasets, DetSeg-S IoU and DetSeg-R AP)
# and latency (FS L&F, T2 protocol: 30 images, 5 warm-up, 3 repeats) with and without --amp.
cd /home/taejun/projects/autonomous-driving-anomaly-segmentation
ROOT=$(pwd); CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth; BARS=$ROOT/results/A3/bars_things.json
cd external/DetSeg-official; export PYTHONPATH=.; O=work_dirs/L1; mkdir -p $O
for spec in "fs_lostfound:FS_LostFound" "road_anomaly:RoadAnomaly" "smiyc_obstacle:SMIYC_Obstacle_val" "smiyc_anomaly:SMIYC_Anomaly_val"; do ds=${spec%%:*}; md=${spec#*:}
  C5="--bars $BARS --stat bar_min --m2a-dir $ROOT/results/score_maps/m2a/$md --verify-t 0.5 --fast"
  n=S_c5_amp_$ds; grep -q 'IoU: [0-9]' $O/$n.log 2>/dev/null || conda run --no-capture-output -n detseg python $ROOT/src/run_with_bars.py $C5 --amp -- $ROOT/configs/detseg-s_$ds.py $CK --work-dir $O/$n > $O/$n.log 2>&1
  echo "$n $(grep -o 'IoU: [0-9.]*' $O/$n.log | tail -1)"
  for amp in "" "--amp"; do n=R_c5${amp:+_amp}_$ds
    grep -q 'AUPRC' $O/$n.log 2>/dev/null || conda run --no-capture-output -n detseg python $ROOT/src/run_with_bars.py $C5 $amp -- $ROOT/configs/detseg-r_m2a_$ds.py $CK --work-dir $O/$n > $O/$n.log 2>&1
    echo "$n $(grep -o 'pixel/AUPRC: [0-9.]*  pixel/FPR@95TPR: [0-9.]*' $O/$n.log | tail -1)"
  done
done
M2A=$ROOT/results/score_maps/m2a/FS_LostFound; N="--images 30 --warmup 5 --repeats 3 --fast"
for amp in "" "--amp"; do
  conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-s_fs_lostfound.py --checkpoint $CK $N $amp --bars $BARS --m2a-dir $M2A --verify-t 0.5 --out $ROOT/results/L1/lat_S_c5${amp:+_amp}.json > $ROOT/results/L1/lat_S_c5${amp:+_amp}.log 2>&1
  conda run --no-capture-output -n detseg python $ROOT/src/profile_latency.py --config $ROOT/configs/detseg-r_m2a_fs_lostfound.py --checkpoint $CK $N $amp --bars $BARS --m2a-dir $M2A --verify-t 0.5 --out $ROOT/results/L1/lat_R_c5${amp:+_amp}.json > $ROOT/results/L1/lat_R_c5${amp:+_amp}.log 2>&1
  echo "latency $amp done"
done
echo L1_DONE
