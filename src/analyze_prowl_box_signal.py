"""P1 pre-analysis: can PROWL prototype similarity tell real anomalies from mistakes in DetSeg boxes?

Same candidates and grouping as src/analyze_m2a_box_signal.py (paper-style
bars, OoD candidates with score > 0.2, TP-like = box IoU >= 0.1 with a GT
component, kept vs suppressed by IE-NMS), tuning datasets only. Anomaly
signals (higher = more anomalous), all from DINOv2 ViT-S/14 + the Cityscapes
train prototype bank (src/prowl_bank.py):
  proto_box   1 - max cosine(mean patch feature inside the box, any prototype)
  proto_patch 1 - mean over patches inside the box of (max cosine to any prototype)
  m2a         mean Mask2Anomaly score inside the box (reference, from I4)

  python src/analyze_prowl_box_signal.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, 'src')
from prowl_bank import load_model, patch_features  # noqa: E402
from sweep_ie_nms_bars import CLASSES, ie_nms, iou_matrix  # noqa: E402

DETSEG = 'external/DetSeg-official'
V1 = [json.load(open('results/A3/bars_things.json'))['classes'][c].get('bar_min', 0.2) for c in CLASSES[1:]]
P = torch.from_numpy(np.load('results/P1/prototypes.npz')['prototypes']).cuda()     # (K, L, D)
K, L, D = P.shape
MAPS = {'fs_lostfound': 'FS_LostFound', 'road_anomaly': 'RoadAnomaly'}


def main():
    model = load_model()
    out = {}
    for ds, mdir in MAPS.items():
        rows = []
        t_feat = []
        for f in sorted(glob.glob(f'results/A2/{ds}/candidates/*.npz')):
            d = np.load(f)
            B, S, Lb = d['bboxes'], d['scores'], d['labels']
            img_path = os.path.join(DETSEG, str(d['img_path']))
            img = np.array(Image.open(img_path).convert('RGB'))
            H, W = img.shape[:2]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            feat, (Hr, Wr) = patch_features(model, img)                        # (h, w, D)
            h, w = feat.shape[:2]
            sim = torch.einsum('hwd,kld->hwkl', feat, P).amax(-1)              # (h, w, K)
            patch_max = sim.amax(-1)                                           # (h, w)
            torch.cuda.synchronize(); t_feat.append(time.perf_counter() - t0)
            sm = np.load(os.path.join('results/score_maps/m2a', mdir, os.path.basename(img_path) + '.npy')).squeeze()
            keep = set(ie_nms(B, S, Lb, V1).tolist())
            gtb = d['gt_boxes'][d['gt_areas'] >= 10]
            for k in np.where((Lb == 0) & (S > 0.2))[0]:
                x1, y1, x2, y2 = np.clip(B[k], 0, [W, H, W, H])
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                # box on the patch grid (at least the patch containing the centre)
                gx1, gx2 = int(x1 / W * w), int(np.ceil(x2 / W * w))
                gy1, gy2 = int(y1 / H * h), int(np.ceil(y2 / H * h))
                gx2, gy2 = max(gx2, gx1 + 1), max(gy2, gy1 + 1)
                region = feat[gy1:gy2, gx1:gx2].reshape(-1, D)
                v = torch.nn.functional.normalize(region.mean(0), dim=0)
                proto_box = 1 - float((P @ v).max())
                proto_patch = 1 - float(patch_max[gy1:gy2, gx1:gx2].mean())
                m2a = float(sm[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))].mean())
                tp = bool(gtb.size) and iou_matrix(B[k:k + 1], gtb).max() >= 0.1
                rows.append((k in keep, tp, proto_box, proto_patch, m2a))
        r = np.array(rows, dtype=float)
        np.save(f'results/P1/box_signal_rows_{ds}.npy', r)   # kept, tp, proto_box, proto_patch, m2a
        print(f'### {ds}: {len(r)} OoD candidates | DINOv2 + prototype matching {np.mean(t_feat[3:]) * 1000:.1f} ms/image')
        res = {}
        for name, sel in [('suppressed', r[:, 0] == 0), ('kept', r[:, 0] == 1), ('all', np.ones(len(r), bool))]:
            x = r[sel]
            line = f'  {name:10s} TP {int(x[:, 1].sum()):4d} FP {int((1 - x[:, 1]).sum()):4d} | AUC'
            for j, sig in [(2, 'proto_box'), (3, 'proto_patch'), (4, 'm2a')]:
                auc = roc_auc_score(x[:, 1], x[:, j]) if 0 < x[:, 1].sum() < len(x) else float('nan')
                res[f'{name}/{sig}'] = auc
                line += f'  {sig} {auc:.3f}'
            print(line)
        out[ds] = res
    os.makedirs('results/P1', exist_ok=True)
    json.dump(out, open('results/P1/box_signal_auc.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
