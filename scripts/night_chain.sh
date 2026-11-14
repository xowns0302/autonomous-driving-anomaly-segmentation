#!/usr/bin/env bash
# Resume chain after WSL restarts: L2 box dumps (skip if complete) -> L1 FP16 (its script skips finished runs).
cd /home/taejun/projects/autonomous-driving-anomaly-segmentation
for t in released c5; do
  n=$(ls results/L2/boxes_$t 2>/dev/null | wc -l)
  echo "boxes_$t has $n"
done
if [ "$(ls results/L2/boxes_released 2>/dev/null | wc -l)" -lt 955 ] || [ "$(ls results/L2/boxes_c5 2>/dev/null | wc -l)" -lt 955 ]; then
  bash scripts/l2_dump.sh
fi
echo "L2 boxes: released $(ls results/L2/boxes_released | wc -l), c5 $(ls results/L2/boxes_c5 | wc -l)"
bash scripts/l1_amp.sh
echo CHAIN_DONE
