"""Object-level reliability counts from saved DetSeg-S masks (F2).

Definitions (fixed before running):
  GT object   = 8-connected component of the OoD label, area >= MIN_AREA px
  found       = at least COVER of the object's pixels are predicted
  missed      = GT object not found
  false alarm = predicted 8-connected component (outside void, area >= MIN_AREA)
                that touches no GT OoD pixel

Inputs: results/F1/masks/<ds>/<method>/<image>.png (scripts/save_masks_for_figures.sh)
Outputs: results/F2/object_counts.csv, results/F2/summary.md

  conda run -n detseg python src/count_missed_objects.py
"""
import csv
import glob
import os

import cv2
import numpy as np

D = 'external/DetSeg-official/data'
MIN_AREA = 50
COVER = 0.5
METHODS = ['released', 'paper', 'c5', 'c8']
NAMES = {'released': '공개 코드', 'paper': '베이스라인 (보정 기준선)', 'c5': 'C5 (제안)', 'c8': 'C8 (대안)'}
DATASETS = {
    'fs_lostfound': ('FS L&F', f'{D}/FS_LostFound/labels_masks/{{}}.png', 1),
    'road_anomaly': ('Road Anomaly', f'{D}/RoadAnomaly/frames/{{}}.labels/labels_semantic.png', 2),
    'smiyc_obstacle': ('SMIYC Obstacle', f'{D}/SMIYC_Obstacle_val/labels_masks/{{}}.png', 1),
    'smiyc_anomaly': ('SMIYC Anomaly', f'{D}/SMIYC_Anomaly_val/labels_masks/{{}}.png', 1),
}


def components(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(lab == i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]


def main():
    os.makedirs('results/F2', exist_ok=True)
    rows, summary = [], {}
    for ds, (title, gt_t, ood) in DATASETS.items():
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(f'results/F1/masks/{ds}/paper/*.png'))
        tot = {m: [0, 0, 0] for m in METHODS}          # gt objects, found, false alarms
        for s in stems:
            g = cv2.imread(gt_t.format(s), cv2.IMREAD_UNCHANGED)
            gt, void = g == ood, g == 255
            objs = components(gt)
            for m in METHODS:
                p = cv2.imread(f'results/F1/masks/{ds}/{m}/{s}.png', 0) > 127
                found = sum(1 for o in objs if p[o].mean() >= COVER)
                fa = sum(1 for c in components(p & ~void) if not (c & gt).any())
                tot[m][0] += len(objs); tot[m][1] += found; tot[m][2] += fa
                rows.append([ds, s, m, len(objs), found, fa])
        summary[ds] = (title, len(stems), tot)
        print(title, {m: tuple(v) for m, v in tot.items()})
    with open('results/F2/object_counts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'image', 'method', 'gt_objects', 'found', 'false_alarms'])
        w.writerows(rows)
    with open('results/F2/summary.md', 'w') as f:
        f.write(f'# F2 물체 단위 집계 (정답 물체 {MIN_AREA}px 이상, 면적 {int(COVER * 100)}% 이상 칠하면 찾음)\n\n')
        f.write('| 데이터셋 | 장수 | 정답 물체 | ' + ' | '.join(f'{NAMES[m]} 찾음 / 놓침 / 잘못 경보' for m in METHODS) + ' |\n')
        f.write('|---|---|---|' + '---|' * len(METHODS) + '\n')
        for ds, (title, n, tot) in summary.items():
            cells = [f'{tot[m][1]} / {tot[m][0] - tot[m][1]} / {tot[m][2]}' for m in METHODS]
            f.write(f'| {title} | {n} | {tot["paper"][0]} | ' + ' | '.join(cells) + ' |\n')
    print(open('results/F2/summary.md').read())


if __name__ == '__main__':
    main()
