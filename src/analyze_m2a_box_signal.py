"""Can Mask2Anomaly's pixel score tell real anomalies from mistakes inside DetSeg boxes?

For each OoD candidate box (label 0, score > 0.2) under the paper-style bars,
compute the mean M2A anomaly score inside the box and check how well it
separates boxes on GT anomalies (TP-like) from boxes elsewhere (FP-like),
separately for boxes that IE-NMS kept and boxes that an ID box suppressed.
Tuning datasets only.

  python src/analyze_m2a_box_signal.py
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, 'src')
from sweep_ie_nms_bars import CLASSES, ie_nms, iou_matrix  # noqa: E402

V1 = [json.load(open('results/A3/bars_things.json'))['classes'][c].get('bar_min', 0.2) for c in CLASSES[1:]]
MAPS = {'fs_lostfound': 'FS_LostFound', 'road_anomaly': 'RoadAnomaly'}
for ds, mdir in MAPS.items():
    rows = []   # (kept, is_tp, mean_score)
    for f in sorted(glob.glob(f'results/A2/{ds}/candidates/*.npz')):
        d = np.load(f)
        B, S, L = d['bboxes'], d['scores'], d['labels']
        sm = np.load(os.path.join('results/score_maps/m2a', mdir, os.path.basename(str(d['img_path'])) + '.npy')).squeeze()
        H, W = sm.shape
        keep = set(ie_nms(B, S, L, V1).tolist())
        gtb = d['gt_boxes'][d['gt_areas'] >= 10]
        for k in np.where((L == 0) & (S > 0.2))[0]:
            x1, y1, x2, y2 = np.clip(B[k], 0, [W, H, W, H]).astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            m = float(sm[y1:y2, x1:x2].mean())
            tp = bool(gtb.size) and iou_matrix(B[k:k + 1], gtb).max() >= 0.1
            rows.append((k in keep, tp, m))
    r = np.array(rows, dtype=float)
    print(f'### {ds}: OoD candidates {len(r)}')
    for name, sel in [('kept by IE-NMS', r[:, 0] == 1), ('suppressed', r[:, 0] == 0), ('all', np.ones(len(r), bool))]:
        x = r[sel]
        tp, fp = x[x[:, 1] == 1, 2], x[x[:, 1] == 0, 2]
        auc = roc_auc_score(x[:, 1], x[:, 2]) if len(tp) and len(fp) else float('nan')
        print(f'  {name:16s} TP {len(tp):4d} mean-M2A median {np.median(tp) if len(tp) else float("nan"):.3f} | '
              f'FP {len(fp):4d} median {np.median(fp) if len(fp) else float("nan"):.3f} | AUC {auc:.3f}')
