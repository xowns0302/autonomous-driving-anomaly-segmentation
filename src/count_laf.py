"""Object-level counts (F2 rules) on the new held-out Lost and Found test set (plan docs/05 §10).

  python src/count_laf.py   -> results/LAF/object_counts.csv, results/LAF/summary.md

Same rules as src/count_missed_objects.py: GT objects >= 50 px, found if >= 50% of the
object is painted; a predicted blob (>= 50 px, outside the ignore region) that touches
no GT pixel is a false positive. Also reports false positives on the images that have
no anomaly at all (every painted blob there is a false alarm).
"""
import csv
import glob
import os

import cv2
import numpy as np

from count_missed_objects import COVER, MIN_AREA, components

GT = 'external/DetSeg-official/data/LAF_test/labels_masks'
MASKS = 'results/F1/masks/laf_test'
METHODS = ['released', 'paper', 'c5', 'c8', 'c9']
NAMES = {'released': '공개 코드', 'paper': '베이스라인', 'c5': 'C5', 'c8': 'C8', 'c9': 'C9 (최종)'}


def main():
    os.makedirs('results/LAF', exist_ok=True)
    stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f'{GT}/*.png'))
    tot = {m: dict(gt=0, found=0, fp=0, fp_clean=0, imgs_clean_fp=0) for m in METHODS}
    rows, n_clean = [], 0
    for s in stems:
        g = cv2.imread(f'{GT}/{s}.png', cv2.IMREAD_UNCHANGED)
        gt, void = g == 1, g == 255
        objs = components(gt)
        clean = not gt.any()
        n_clean += clean
        for m in METHODS:
            p = cv2.imread(f'{MASKS}/{m}/{s}.png', 0) > 127
            found = sum(1 for o in objs if p[o].mean() >= COVER)
            fp = sum(1 for c in components(p & ~void) if not (c & gt).any())
            t = tot[m]
            t['gt'] += len(objs); t['found'] += found; t['fp'] += fp
            if clean:
                t['fp_clean'] += fp; t['imgs_clean_fp'] += int(fp > 0)
            rows.append([s, m, len(objs), found, fp, int(clean)])
    with open('results/LAF/object_counts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'method', 'gt_objects', 'found', 'false_positives', 'no_anomaly_image'])
        w.writerows(rows)
    with open('results/LAF/summary.md', 'w') as f:
        f.write(f'# Lost and Found test (새 검증셋) 물체 단위 집계 — {len(stems)}장, 이상 물체 없는 사진 {n_clean}장\n\n')
        f.write('| 방법 | 정답 물체 | 미탐 | 오탐 | 이상 물체 없는 사진의 오탐 | 오탐이 1개 이상 나온 그 사진 수 |\n|---|---|---|---|---|---|\n')
        for m in METHODS:
            t = tot[m]
            f.write(f"| {NAMES[m]} | {t['gt']} | {t['gt'] - t['found']} | {t['fp']} | {t['fp_clean']} | {t['imgs_clean_fp']} / {n_clean} |\n")
    print(open('results/LAF/summary.md').read())


if __name__ == '__main__':
    main()
