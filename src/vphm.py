"""L2 (paper limitation "Support for dynamic objects"): VPHM, implemented from the paper.

The released code has no working VPHM (only an unfinished VanishingPointGuidedCost),
so this follows DetSeg §3.3 / Eq. 3:
  * vanishing point (VP) per frame from the camera calibration shipped with Lost and Found
    (flat straight road: u = u0 + fx*tan(yaw), v = v0 - fy*tan(pitch));
  * for an object a(t-1), the fan region is bounded by the rays VP -> midpoints of its left
    and right box edges; for each box b(t) a pseudo box is placed in the fan at b(t)'s
    vertical centre, with the fan's width there and a(t-1)'s aspect ratio;
  * cost = IoU(pseudo box, b(t)), Hungarian matching, pairs with IoU < 0.5 dropped;
  * smoothing: n^(t) = w * n(t) + (1 - w) * max(n(t), n^(t-1)), w = 0.2.
Consecutive frames = consecutive annotated frames of one Lost and Found sequence.

Evaluation on the held-out LAF test set (src/prepare_laf.py), with boxes dumped by
run_with_bars.py --dump-boxes, keep = smoothed objectness > tau (0.2):
  * box level: GT object (>= 50 px) found if kept boxes cover >= 50% of it; a kept box with
    no GT anomaly pixel and its centre on the evaluated (non-ignored) area is a false positive;
  * DetSeg-R pixel AP / FPR95: Mask2Anomaly score + 1 inside kept boxes (as the released code);
  * stability: mean |objectness change| of matched pairs, before and after smoothing;
  * lateral-motion test: matched pairs are shifted sideways to imitate a moving object and
    the share still matched is measured for VPHM and for VPHM + constant-velocity prediction.
  python src/vphm.py --boxes <dump dir> --tag <name>
"""
import argparse
import collections
import glob
import json
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

LAF = os.path.expanduser('~/datasets/LostAndFound')
GT = 'external/DetSeg-official/data/LAF_test/labels_masks'
M2A = 'results/score_maps/m2a/LAF_test'
TAU, W, MIN_AREA, COVER = 0.2, 0.2, 50, 0.5


def vanishing_point(stem):
    site = '_'.join(stem.split('_')[:-2])
    c = json.load(open(f'{LAF}/camera/test/{site}/{stem}_camera.json'))
    i, e = c['intrinsic'], c['extrinsic']
    return i['u0'] + i['fx'] * np.tan(e['yaw']), i['v0'] - i['fy'] * np.tan(e['pitch'])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def pseudo_box(a, b, vp):
    """Box placed inside a's fan region at b's vertical centre (paper §3.3)."""
    vx, vy = vp
    cy_a = (a[1] + a[3]) / 2
    yb = (b[1] + b[3]) / 2
    if abs(cy_a - vy) < 1e-3:
        return None
    s = (yb - vy) / (cy_a - vy)            # ray VP -> (x, cy_a) reaches x' = vx + s*(x - vx) at yb
    if s <= 0:
        return None
    xl, xr = vx + s * (a[0] - vx), vx + s * (a[2] - vx)
    wb = abs(xr - xl)
    hb = wb * (a[3] - a[1]) / max(a[2] - a[0], 1e-6)
    cx = (xl + xr) / 2
    return np.array([cx - wb / 2, yb - hb / 2, cx + wb / 2, yb + hb / 2])


def match(prev, cur, vp, prev_vel=None):
    """Hungarian on IoU(pseudo, cur); returns list of (i_prev, j_cur, iou). With prev_vel,
    each prev box is first moved by its last displacement (constant-velocity variant)."""
    if len(prev) == 0 or len(cur) == 0:
        return []
    C = np.zeros((len(prev), len(cur)))
    for i, a in enumerate(prev):
        a2 = a + (prev_vel[i] if prev_vel is not None else 0)
        for j, b in enumerate(cur):
            pb = pseudo_box(a2, b, vp)
            C[i, j] = iou(pb, b) if pb is not None else 0.0
    r, c = linear_sum_assignment(-C)
    return [(i, j, C[i, j]) for i, j in zip(r, c) if C[i, j] >= 0.5]


def load_seq(boxdir):
    seqs = collections.defaultdict(list)
    for p in sorted(glob.glob(f'{GT}/*.png')):
        stem = os.path.splitext(os.path.basename(p))[0]
        parts = stem.split('_')
        seqs[('_'.join(parts[:-2]), parts[-2])].append((int(parts[-1]), stem))
    out = {}
    for k, v in seqs.items():
        frames = []
        for _, stem in sorted(v):
            d = np.load(f'{boxdir}/{stem}.png.npz')
            ood = d['labels'] == 0
            frames.append(dict(stem=stem, boxes=d['bboxes'][ood].astype(np.float64), scores=d['scores'][ood].astype(np.float64)))
        out[k] = frames
    return out


def smooth(seqs):
    """Adds 'smoothed' scores; returns stability stats and the matched pairs (for the motion test)."""
    d_raw, d_sm, pairs = [], [], []
    for frames in seqs.values():
        prev = None
        for f in frames:
            f['smoothed'] = f['scores'].copy()
            if prev is not None:
                cand = prev['smoothed'] > 0.05          # objects carried over from t-1
                cur = f['scores'] > 0.05
                ip, jc = np.where(cand)[0], np.where(cur)[0]
                vp = vanishing_point(f['stem'])
                for i, j, _ in match(prev['boxes'][ip], f['boxes'][jc], vp):
                    a, b = ip[i], jc[j]
                    f['smoothed'][b] = W * f['scores'][b] + (1 - W) * max(f['scores'][b], prev['smoothed'][a])
                    d_raw.append(abs(f['scores'][b] - prev['scores'][a]))
                    d_sm.append(abs(f['smoothed'][b] - prev['smoothed'][a]))
                    pairs.append((prev['boxes'][a], f['boxes'][b], vp, prev.get('vel_src')))
            prev = f
    return dict(matched_pairs=len(d_raw), mean_abs_change_raw=float(np.mean(d_raw)) if d_raw else 0,
                mean_abs_change_smoothed=float(np.mean(d_sm)) if d_sm else 0), pairs


def components(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_AREA]


def evaluate(seqs, key, hist_bins=2000):
    gt_n = found = fp = 0
    pos = np.zeros(hist_bins); neg = np.zeros(hist_bins)
    for frames in seqs.values():
        for f in frames:
            g = cv2.imread(f'{GT}/{f["stem"]}.png', cv2.IMREAD_UNCHANGED)
            gt, void = g == 1, g == 255
            keep = f['boxes'][f[key] > TAU]
            cover = np.zeros_like(gt)
            for x1, y1, x2, y2 in keep.astype(int):
                x1, y1 = max(x1, 0), max(y1, 0)
                cover[y1:y2, x1:x2] = True
                cx, cy = min((x1 + x2) // 2, g.shape[1] - 1), min((y1 + y2) // 2, g.shape[0] - 1)
                if not gt[y1:y2, x1:x2].any() and not void[cy, cx]:
                    fp += 1
            objs = components(gt)
            gt_n += len(objs); found += sum(1 for o in objs if cover[o].mean() >= COVER)
            if gt.any():                          # pixel AP over images with anomalies (as the evaluator)
                s = np.load(f'{M2A}/{f["stem"]}.png.npy').squeeze().astype(np.float32)
                s = s + cover.astype(np.float32)   # DetSeg-R: +1 inside kept boxes
                idx = np.clip((s / 2.0 * hist_bins).astype(int), 0, hist_bins - 1)
                pos += np.bincount(idx[gt], minlength=hist_bins)
                neg += np.bincount(idx[~gt & ~void], minlength=hist_bins)
    tp = np.cumsum(pos[::-1]); fpc = np.cumsum(neg[::-1])
    prec = tp / np.maximum(tp + fpc, 1); rec = tp / tp[-1]
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    fpr95 = float(fpc[np.searchsorted(rec, 0.95)] / fpc[-1])
    return dict(gt_objects=gt_n, missed=gt_n - found, false_positive_boxes=fp, AP=ap * 100, FPR95=fpr95 * 100)


def motion_test(pairs, shifts=(0, 0.25, 0.5, 1.0, 1.5, 2.0)):
    """Shift b(t) sideways by k box widths (a moving object) and check whether it still matches.
    VPHM: fan prior only. +velocity: a(t-1) is first moved by the same displacement it had
    since t-2 (constant-velocity prediction, as a Kalman tracker would). The velocity is exact
    there (an upper bound); +velocity(half) predicts only half of the true displacement."""
    out = {}
    for k in shifts:
        ok_v = ok_cv = ok_half = 0
        for a, b, vp, _ in pairs:
            dx = k * (b[2] - b[0])
            b2 = b + np.array([dx, 0, dx, 0])
            ok_v += bool(match(a[None], b2[None], vp))
            ok_cv += bool(match(a[None], b2[None], vp, prev_vel=np.array([[dx, 0, dx, 0]])))
            ok_half += bool(match(a[None], b2[None], vp, prev_vel=np.array([[dx, 0, dx, 0]]) / 2))
        n = max(len(pairs), 1)
        out[str(k)] = dict(vphm=ok_v / n, vphm_velocity=ok_cv / n, vphm_velocity_half=ok_half / n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boxes', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    seqs = load_seq(a.boxes)
    stab, pairs = smooth(seqs)
    res = dict(tag=a.tag, sequences=len(seqs), frames=sum(len(v) for v in seqs.values()), stability=stab,
               without_vphm=evaluate(seqs, 'scores'), with_vphm=evaluate(seqs, 'smoothed'),
               lateral_motion=motion_test(pairs))
    os.makedirs('results/L2', exist_ok=True)
    json.dump(res, open(f'results/L2/{a.tag}.json', 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
