"""What does IE-NMS actually do? (A2)

Uses pre-NMS candidates saved by diag_ie_nms_sources.py. For every OoD
candidate that would pass tau (label 0, score > 0.2), find its fate after the
released IE-NMS (bars stuff 0.2 / thing 0.5, boost x2, IoU 0.5):
  kept                     -> final OoD box
  suppressed by boosted ID -> the intended IE-NMS effect
  suppressed by other      -> plain NMS (another OoD box or an unboosted ID box)
Candidates are split by whether they overlap a GT OoD object (box IoU >= 0.1).

For kept boxes lying on an ID-class prediction (IoU >= 0.5), it also reports
the bar that ID box would have needed to be boosted (= its score).

  python src/analyze_ie_nms_effect.py results/A2/fs_lostfound/candidates ...
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np

from sweep_ie_nms_bars import CLASSES, bars, ie_nms, iou_matrix


def fate(c, keep_set, B, S, L, boosted):
    if c in keep_set:
        return 'kept'
    kept = np.array(sorted(keep_set))
    ious = iou_matrix(B[c:c + 1], B[kept])[0]
    nms_s = np.where(boosted[kept], S[kept] * 2, S[kept])
    cand = [(nms_s[i], kept[i]) for i in np.where(ious > 0.5)[0] if nms_s[i] >= S[c]]
    if not cand:
        return 'suppressed: other'
    k = max(cand)[1]
    if L[k] != 0 and boosted[k]:
        return 'suppressed: boosted ID'
    return 'suppressed: other'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cand_dirs', nargs='+')
    p.add_argument('--tau', type=float, default=0.2)
    args = p.parse_args()
    bl = bars(0.2, 0.5)
    for cd in args.cand_dirs:
        name = os.path.basename(os.path.dirname(cd.rstrip('/')))
        fates = {True: Counter(), False: Counter()}
        needed = Counter()
        needed_scores = {True: [], False: []}
        files = sorted(glob.glob(os.path.join(cd, '*.npz')))
        for f in files:
            d = np.load(f)
            B, S, L = d['bboxes'], d['scores'], d['labels']
            thres = np.ones_like(S)
            m = (L > 0) & (L <= 19)
            thres[m] = np.array(bl)[L[m] - 1]
            boosted = (L != 0) & (S > thres)
            keep_set = set(ie_nms(B, S, L, bl).tolist())
            gtb = d['gt_boxes'][d['gt_areas'] >= 10]
            cands = np.where((L == 0) & (S > args.tau))[0]
            on_gt = (iou_matrix(B[cands], gtb).max(1) >= 0.1) if gtb.size else np.zeros(len(cands), bool)
            id_idx = np.where(L > 0)[0]
            for c, g in zip(cands, on_gt):
                fo = fate(c, keep_set, B, S, L, boosted)
                fates[bool(g)][fo] += 1
                if fo == 'kept' and len(id_idx):
                    ious = iou_matrix(B[c:c + 1], B[id_idx])[0]
                    j = int(ious.argmax())
                    if ious[j] >= 0.5:
                        k = id_idx[j]
                        needed[(bool(g), CLASSES[L[k]])] += 1
                        needed_scores[bool(g)].append(float(S[k]))
        print(f'\n### {name} ({len(files)} images): OoD candidates with score > {args.tau}')
        for g, title in [(False, 'not on GT (FP-like)'), (True, 'on GT (TP-like)')]:
            n = sum(fates[g].values())
            print(f'  {title}: {n} -> ' + ', '.join(f'{k} {v} ({v / max(n, 1) * 100:.0f}%)'
                                                  for k, v in fates[g].most_common()))
        for g, title in [(False, 'FP kept on an ID box'), (True, 'TP kept on an ID box')]:
            sc = np.array(needed_scores[g])
            if len(sc):
                top = Counter({c: v for (gg, c), v in needed.items() if gg == g}).most_common(5)
                print(f'  {title}: {len(sc)}, ID score p10/p50/p90 = '
                      f'{np.percentile(sc, 10):.3f}/{np.percentile(sc, 50):.3f}/{np.percentile(sc, 90):.3f}, classes {dict(top)}')


if __name__ == '__main__':
    main()
