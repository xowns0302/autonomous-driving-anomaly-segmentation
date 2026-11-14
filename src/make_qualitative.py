"""Phase 4 qualitative figures from masks saved by scripts/save_masks_for_figures.sh.

Per dataset, images are chosen by fixed rules (no hand picking):
  * the 2 images with the largest IoU gain of C5 over the paper-bar baseline
  * the 1 image with the largest IoU loss of C8 relative to C5 (failure case)
Only images with a real change (> 1 IoU point) are picked.
Each row: input with GT outline | released | baseline (paper-style bars) | C5 | C8,
prediction in red, GT outline in green, void pixels (ignored by the metric) darkened,
per-image IoU in the corner.
Also writes per-image IoU of all methods to results/F1/per_image_iou.csv.

  python src/make_qualitative.py
"""
import csv
import glob
import os

import cv2
import numpy as np

D = 'external/DetSeg-official/data'
METHODS = [('released', '공개본'), ('paper', 'baseline (paper bars)'), ('c5', 'C5'), ('c8', 'C8')]
LABELS = ['released', 'baseline', 'C5', 'C8']
DATASETS = {
    'fs_lostfound': (f'{D}/FS_LostFound/images/{{}}.png', f'{D}/FS_LostFound/labels_masks/{{}}.png', 1),
    'road_anomaly': (f'{D}/RoadAnomaly/frames/{{}}.webp', f'{D}/RoadAnomaly/frames/{{}}.labels/labels_semantic.png', 2),
    'smiyc_obstacle': (f'{D}/SMIYC_Obstacle_val/images/{{}}.webp', f'{D}/SMIYC_Obstacle_val/labels_masks/{{}}.png', 1),
    'smiyc_anomaly': (f'{D}/SMIYC_Anomaly_val/images/{{}}.png', f'{D}/SMIYC_Anomaly_val/labels_masks/{{}}.png', 1),
}


def iou(gt, pred):
    inter = np.logical_and(gt == 1, pred).sum()
    union = np.logical_and(gt != 255, np.logical_or(gt == 1, pred)).sum()
    return inter / union if union else 0.0


def tile(img, gt, pred, text, width=640):
    over = img.copy()
    void = gt == 255
    over[void] = (0.45 * img[void]).astype(np.uint8)
    over[pred] = (0.45 * img[pred] + np.array([0, 0, 140])).astype(np.uint8)
    cnt, _ = cv2.findContours((gt == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnt, -1, (0, 255, 0), max(2, img.shape[1] // 600))
    h = int(img.shape[0] * width / img.shape[1])
    t = cv2.resize(over, (width, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(t, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    os.makedirs('assets/figures', exist_ok=True)
    rows_csv = []
    for ds, (img_t, gt_t, ood) in DATASETS.items():
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(f'results/F1/masks/{ds}/paper/*.png'))
        scores = {}
        for s in stems:
            gt = cv2.imread(gt_t.format(s), cv2.IMREAD_UNCHANGED)
            gt = np.where(gt == ood, 1, np.where(gt == 255, 255, 0))
            scores[s] = {m: iou(gt, cv2.imread(f'results/F1/masks/{ds}/{m}/{s}.png', 0) > 127) for m, _ in METHODS}
            rows_csv.append([ds, s] + [f'{scores[s][m] * 100:.2f}' for m, _ in METHODS])
        gain = [s for s in sorted(stems, key=lambda s: scores[s]['c5'] - scores[s]['paper'], reverse=True)
                if scores[s]['c5'] - scores[s]['paper'] > 0.01][:2]
        loss = [s for s in sorted(stems, key=lambda s: scores[s]['c8'] - scores[s]['c5'])
                if scores[s]['c8'] - scores[s]['c5'] < -0.01][:1]
        picks = [(s, 'C5 gain') for s in gain] + [(s, 'C8 loss') for s in loss if s not in gain]
        rows = []
        for s, why in picks:
            img = cv2.imread(img_t.format(s))
            gt = cv2.imread(gt_t.format(s), cv2.IMREAD_UNCHANGED)
            gt = np.where(gt == ood, 1, np.where(gt == 255, 255, 0))
            tiles = [tile(img, gt, np.zeros(gt.shape, bool), f'GT  {s[:28]}  [{why}]')]
            for (m, _), lab in zip(METHODS, LABELS):
                pred = cv2.imread(f'results/F1/masks/{ds}/{m}/{s}.png', 0) > 127
                tiles.append(tile(img, gt, pred, f'{lab}  IoU {scores[s][m] * 100:.1f}'))
            rows.append(np.hstack(tiles))
        cv2.imwrite(f'assets/figures/qual_{ds}.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(ds, 'picked', picks)
    with open('results/F1/per_image_iou.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'image'] + [m for m, _ in METHODS])
        w.writerows(rows_csv)


if __name__ == '__main__':
    main()
