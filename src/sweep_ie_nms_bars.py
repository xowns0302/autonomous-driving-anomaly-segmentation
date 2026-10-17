"""Offline IE-NMS bar sweep on candidates dumped by diag_ie_nms_sources.py.

Re-runs the IE-NMS logic of models.py:518-541 with different per-class bars on
saved pre-NMS candidates, then applies DetSeg.predict's final filter
(label == 0 and score > 0.2). Box-level only (no SAM), so it runs in seconds.

Exploration tool: bars chosen here must NOT be reported as tuned on the same
evaluation set. Final bars must come from a separate calibration split.

  python src/sweep_ie_nms_bars.py \
      external/DetSeg-official/work_dirs/diag_ie_nms/fs_lostfound/candidates \
      external/DetSeg-official/work_dirs/diag_ie_nms/road_anomaly/candidates
"""
import argparse
import glob
import os

import numpy as np
import torch
from mmcv.ops import batched_nms

# label index -> class (label 0 = 'objects' / OoD candidate)
CLASSES = ('objects', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
NEVER = 1.0  # bar of 1.0 = never boosted (scores are sigmoid outputs < 1)


def bars(stuff, thing, overrides=None):
    """Bars for labels 1..19: labels 1-11 (road..sky) = stuff, 12-19 = thing."""
    b = [stuff] * 11 + [thing] * 8
    for name, v in (overrides or {}).items():
        b[CLASSES.index(name) - 1] = v
    return b


OBSTACLE_LIKE = {'pole': NEVER, 'traffic light': NEVER, 'traffic sign': NEVER, 'fence': NEVER}

CONFIGS = [
    ('baseline (stuff .2 / thing .5)', bars(0.2, 0.5)),
    ('thing .4', bars(0.2, 0.4)),
    ('thing .3', bars(0.2, 0.3)),
    ('thing .2', bars(0.2, 0.2)),
    ('thing .15', bars(0.2, 0.15)),
    ('thing .1', bars(0.2, 0.1)),
    ('all .1', bars(0.1, 0.1)),
    ('thing .2 + obstacle-like never', bars(0.2, 0.2, OBSTACLE_LIKE)),
    ('thing .15 + obstacle-like never', bars(0.2, 0.15, OBSTACLE_LIKE)),
    ('no IE-NMS boost', bars(NEVER, NEVER)),
]


def ie_nms(B, S, L, bar_list, iou_thr=0.5, boost=2.0):
    """models.py:518-541 with configurable bars; returns kept indices."""
    labels = torch.as_tensor(L).long()
    scores = torch.as_tensor(S).float()
    thres = torch.ones_like(scores)
    id_thr = torch.tensor(bar_list, dtype=torch.float32)
    m = (labels > 0) & (labels <= len(id_thr))
    thres[m] = id_thr[labels[m] - 1]
    boosted = (labels != 0) & (scores > thres)
    nms_scores = scores.clone()
    nms_scores[boosted] *= boost
    _, keep = batched_nms(torch.as_tensor(B).float(), nms_scores, labels,
                          nms_cfg=dict(iou_threshold=iou_thr), class_agnostic=True)
    return keep.numpy()


# C1: amorphous background ("stuff") classes. fence, pole, traffic light and
# traffic sign are Cityscapes stuff classes too, but they are object-like.
BACKGROUND = ('road', 'sidewalk', 'building', 'wall', 'vegetation', 'terrain', 'sky')


def ie_nms_rules(B, S, L, bar_list, iou_thr=0.5, boost=2.0,
                 background_exempt=False, protect_gamma=None,
                 verify_scores=None, verify_t=None):
    """IE-NMS with Phase 3 suppression rules; returns kept indices.

    Same as ie_nms (boost ID boxes above their bar, then class-agnostic greedy
    NMS), except that an OoD box (label 0) is NOT suppressed by an ID box when
      C1 background_exempt: the ID box is a BACKGROUND class, or
      C2 protect_gamma:     the OoD box score >= protect_gamma, or
      C5 verify_t:          verify_scores[OoD box] >= verify_t (e.g. the mean
                            Mask2Anomaly score inside the box).
    OoD-vs-OoD and ID-vs-ID suppression are unchanged. With both rules off the
    result equals ie_nms.
    """
    B = np.asarray(B, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    L = np.asarray(L).astype(np.int64)
    thres = np.ones_like(S)
    m = (L > 0) & (L <= len(bar_list))
    thres[m] = np.asarray(bar_list, dtype=np.float32)[L[m] - 1]
    boosted = (L != 0) & (S > thres)
    nms_s = np.where(boosted, S * boost, S)
    bg = np.isin(L, [CLASSES.index(c) for c in BACKGROUND])

    order = np.argsort(-nms_s, kind='stable')
    alive = np.ones(len(B), bool)
    keep = []
    for i in order:
        if not alive[i]:
            continue
        keep.append(i)
        alive[i] = False
        cand = np.where(alive)[0]
        if not len(cand):
            break
        sup = iou_matrix(B[i:i + 1], B[cand])[0] > iou_thr
        if L[i] != 0:   # suppressor is an ID box: apply the protection rules
            prot = L[cand] == 0
            rule = np.zeros(len(cand), bool)
            if background_exempt and bg[i]:
                rule |= True
            if protect_gamma is not None:
                rule |= S[cand] >= protect_gamma
            if verify_t is not None:
                rule |= np.asarray(verify_scores)[cand] >= verify_t
            sup &= ~(prot & rule)
        alive[cand[sup]] = False
    return np.array(keep, dtype=np.int64)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None] - inter, 1e-6)


def evaluate(files, bar_list, min_gt_area, final_thr=0.2):
    n_final = n_fp = 0
    rec = {'small': [0, 0], 'medium': [0, 0], 'large': [0, 0]}  # [recalled, total]
    for f in files:
        d = np.load(f)
        keep = ie_nms(d['bboxes'], d['scores'], d['labels'], bar_list)
        final = [k for k in keep if d['labels'][k] == 0 and d['scores'][k] > final_thr]
        fb = d['bboxes'][final] if final else np.zeros((0, 4))
        valid = d['gt_areas'] >= min_gt_area
        gtb, gta = d['gt_boxes'][valid], d['gt_areas'][valid]
        ious = iou_matrix(fb, gtb)
        n_final += len(fb)
        n_fp += int((ious.max(1) < 0.1).sum()) if gtb.size else len(fb)
        best = ious.max(0) if len(fb) else np.zeros(len(gtb))
        for a, b in zip(gta, best):
            key = 'small' if a < 32 ** 2 else 'medium' if a < 96 ** 2 else 'large'
            rec[key][1] += 1
            rec[key][0] += int(b >= 0.5)
    tot = [sum(v[i] for v in rec.values()) for i in (0, 1)]
    pct = lambda r: f'{r[0] / r[1] * 100:5.1f}' if r[1] else '  -  '
    return dict(boxes_per_img=n_final / len(files), fp_per_img=n_fp / len(files),
                recall=pct(tot), small=pct(rec['small']), medium=pct(rec['medium']),
                large=pct(rec['large']), n_gt=tot[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cand_dirs', nargs='+')
    p.add_argument('--min-gt-area', type=int, default=10,
                   help='ignore GT components smaller than this (Road Anomaly has 1px noise)')
    args = p.parse_args()
    for cd in args.cand_dirs:
        files = sorted(glob.glob(os.path.join(cd, '*.npz')))
        name = os.path.basename(os.path.dirname(cd.rstrip('/')))
        print(f'\n### {name}  ({len(files)} images, GT comps >= {args.min_gt_area}px)')
        print(f'| config | boxes/img | FP/img | recall | small<32² | 32²~96² | >=96² |')
        print('|---|---|---|---|---|---|---|')
        for cname, bl in CONFIGS:
            r = evaluate(files, bl, args.min_gt_area)
            print(f"| {cname} | {r['boxes_per_img']:.2f} | {r['fp_per_img']:.2f} | {r['recall']} | "
                  f"{r['small']} | {r['medium']} | {r['large']} |")


if __name__ == '__main__':
    main()
