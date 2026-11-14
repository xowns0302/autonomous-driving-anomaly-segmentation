"""I2: are the remaining false OoD boxes on or off the predicted road?

For final OoD boxes under a bar setting, report how many TP-like / FP-like
boxes have their bottom-center inside a predicted road box (label 'road',
ID-branch score > 0.2). Uses candidates saved by diag_ie_nms_sources.py.

  python src/analyze_road_context.py results/A2/fs_lostfound/candidates results/A2/road_anomaly/candidates
"""
import json
import sys

import glob
import numpy as np

sys.path.insert(0, 'src')
from sweep_ie_nms_bars import CLASSES, bars, ie_nms, iou_matrix  # noqa: E402

V1 = [json.load(open('results/A3/bars_things.json'))['classes'][c].get('bar_min', 0.2) for c in CLASSES[1:]]
for cd in sys.argv[1:]:
    for name, bl in [('released', bars(0.2, 0.5)), ('V1 (paper bars)', V1)]:
        c = {('TP', True): 0, ('TP', False): 0, ('FP', True): 0, ('FP', False): 0}
        noroad = 0
        files = sorted(glob.glob(cd + '/*.npz'))
        for f in files:
            d = np.load(f); B, S, L = d['bboxes'], d['scores'], d['labels']
            keep = ie_nms(B, S, L, bl)
            fin = [k for k in keep if L[k] == 0 and S[k] > 0.2]
            road = B[(L == 1) & (S > 0.2)]
            if not len(road):
                noroad += 1
            gtb = d['gt_boxes'][d['gt_areas'] >= 10]
            for k in fin:
                x, y = (B[k, 0] + B[k, 2]) / 2, B[k, 3] - 1
                on = bool(len(road)) and bool(((road[:, 0] <= x) & (x <= road[:, 2]) & (road[:, 1] <= y) & (y <= road[:, 3])).any())
                t = 'TP' if gtb.size and iou_matrix(B[k:k + 1], gtb).max() >= 0.1 else 'FP'
                c[(t, on or not len(road))] += 1
        tp, fp = c[('TP', True)] + c[('TP', False)], c[('FP', True)] + c[('FP', False)]
        print(f"{cd.split('/')[-2]:14s} {name:16s} TP {tp:4d} (on road {c[('TP', True)] / max(tp, 1) * 100:3.0f}%) | "
              f"FP {fp:4d} (on road {c[('FP', True)] / max(fp, 1) * 100:3.0f}%, off road {c[('FP', False)]}) | images without road box {noroad}/{len(files)}")
