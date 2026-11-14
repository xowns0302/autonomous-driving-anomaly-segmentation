"""DetSeg-S IoU on the Lost and Found held-out set from the saved masks (plan docs/05 §10).

The official test run was killed by the OS (out of RAM) at the metric step after all
955 masks were saved, so the score is recomputed from those masks with the same rule
as src/make_qualitative.py (checked equal to the official IoU on all four sets, F1):
per-image IoU over non-ignored pixels, averaged over images that contain anomaly pixels
(the official evaluator skips images without them).
  python src/iou_laf.py  -> results/LAF/iou.md, results/LAF/per_image_iou.csv
"""
import csv
import glob
import os

import cv2
import numpy as np

from make_qualitative import iou

GT = 'external/DetSeg-official/data/LAF_test/labels_masks'
MASKS = 'results/F1/masks/laf_test'
METHODS = ['released', 'paper', 'c5', 'c8', 'c9']


def main():
    os.makedirs('results/LAF', exist_ok=True)
    stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f'{GT}/*.png'))
    rows, acc = [], {m: [] for m in METHODS}
    for s in stems:
        gt = cv2.imread(f'{GT}/{s}.png', cv2.IMREAD_UNCHANGED)
        if not (gt == 1).any():
            continue
        r = {m: iou(gt, cv2.imread(f'{MASKS}/{m}/{s}.png', 0) > 127) for m in METHODS}
        for m in METHODS:
            acc[m].append(r[m])
        rows.append([s] + [f'{r[m] * 100:.2f}' for m in METHODS])
    with open('results/LAF/per_image_iou.csv', 'w', newline='') as f:
        csv.writer(f).writerows([['image'] + METHODS] + rows)
    with open('results/LAF/iou.md', 'w') as f:
        f.write(f'| 방법 | IoU (이상 물체가 있는 {len(rows)}장 평균) |\n|---|---|\n')
        for m in METHODS:
            f.write(f'| {m} | {np.mean(acc[m]) * 100:.2f} |\n')
    print(open('results/LAF/iou.md').read())


if __name__ == '__main__':
    main()
