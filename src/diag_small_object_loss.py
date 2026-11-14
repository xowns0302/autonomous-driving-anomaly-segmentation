"""Where are GT OoD objects lost in DetSeg's box pipeline?

Uses pre-NMS candidates dumped by diag_ie_nms_sources.py (ID branch top-300 +
all 300 UN queries per image). For every GT component, the first failing stage:

  1 no_candidate     : no candidate box (any branch/label/score) with IoU >= thr
  2 id_label_only    : localized only by ID-class boxes (label 1..19)
  3 low_score        : a label-0 candidate localizes it, but all have score <= 0.2
  4 nms_removed      : a label-0 candidate with score > 0.2 exists but NMS removed it
                       (suppressor recorded: boosted ID box / unboosted ID box / label-0 box)
  5 detected         : a final OoD box (kept, label 0, score > 0.2) localizes it

  python src/diag_small_object_loss.py \
      external/DetSeg-official/work_dirs/diag_ie_nms/fs_lostfound/candidates \
      --thing-bar 0.5
"""
import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np

from sweep_ie_nms_bars import CLASSES, bars, ie_nms, iou_matrix

STAGES = ['no_candidate', 'id_label_only', 'low_score', 'nms_removed', 'detected']
BINS = [('<16²', 0, 16 ** 2), ('16²~32²', 16 ** 2, 32 ** 2), ('32²~96²', 32 ** 2, 96 ** 2), ('≥96²', 96 ** 2, 1e12)]


def suppressor_of(c, keep_set, B, S, L, boosted, iou_thr=0.5):
    """Highest-NMS-score kept box overlapping candidate c (IoU > thr)."""
    kept = np.array(sorted(keep_set))
    ious = iou_matrix(B[c:c + 1], B[kept])[0]
    nms_s = np.where(boosted[kept], S[kept] * 2, S[kept])
    cand = [(nms_s[i], kept[i]) for i in np.where(ious > iou_thr)[0]]
    if not cand:
        return 'unknown'
    k = max(cand)[1]
    if L[k] == 0:
        return 'label-0 box'
    return f'ID {CLASSES[L[k]]} ({"boosted" if boosted[k] else "not boosted"})'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cand_dir')
    p.add_argument('--stuff-bar', type=float, default=0.2)
    p.add_argument('--thing-bar', type=float, default=0.5)
    p.add_argument('--iou-thr', type=float, default=0.5, help='box IoU to count as localized')
    p.add_argument('--min-gt-area', type=int, default=10)
    p.add_argument('--final-thr', type=float, default=0.2)
    args = p.parse_args()

    bar_list = bars(args.stuff_bar, args.thing_bar)
    stage_by_bin = defaultdict(Counter)
    suppressors = Counter()
    best_lbl0_scores = defaultdict(list)   # best label-0 candidate score for localized GT, per bin
    best_any_iou = defaultdict(list)       # localization quality of best candidate, per bin

    for f in sorted(glob.glob(os.path.join(args.cand_dir, '*.npz'))):
        d = np.load(f)
        B, S, L = d['bboxes'], d['scores'], d['labels']
        thres = np.ones_like(S)
        m = (L > 0) & (L <= 19)
        thres[m] = np.array(bar_list)[L[m] - 1]
        boosted = (L != 0) & (S > thres)
        keep_set = set(ie_nms(B, S, L, bar_list).tolist())

        valid = d['gt_areas'] >= args.min_gt_area
        for gtb, area in zip(d['gt_boxes'][valid], d['gt_areas'][valid]):
            bname = next(n for n, lo, hi in BINS if lo <= area < hi)
            ious = iou_matrix(gtb[None], B)[0]
            best_any_iou[bname].append(float(ious.max()))
            hit = ious >= args.iou_thr
            lbl0 = hit & (L == 0)
            if lbl0.any():
                best_lbl0_scores[bname].append(float(S[lbl0].max()))
            final = [k for k in np.where(lbl0 & (S > args.final_thr))[0] if k in keep_set]
            if not hit.any():
                st = 'no_candidate'
            elif not lbl0.any():
                st = 'id_label_only'
            elif not (lbl0 & (S > args.final_thr)).any():
                st = 'low_score'
            elif not final:
                st = 'nms_removed'
                c = int(np.where(lbl0 & (S > args.final_thr))[0][np.argmax(S[lbl0 & (S > args.final_thr)])])
                suppressors[suppressor_of(c, keep_set, B, S, L, boosted)] += 1
            else:
                st = 'detected'
            stage_by_bin[bname][st] += 1

    print(f'bars: stuff {args.stuff_bar} / thing {args.thing_bar} | localized = box IoU >= {args.iou_thr} '
          f'| GT >= {args.min_gt_area}px\n')
    print('| GT 면적 | n | ' + ' | '.join(STAGES) + ' |')
    print('|---|---|' + '---|' * len(STAGES))
    total = Counter()
    for bname, _, _ in BINS + [('all', 0, 0)]:
        c = total if bname == 'all' else stage_by_bin[bname]
        n = sum(c.values())
        if bname != 'all':
            total.update(c)
        if n:
            print(f'| {bname} | {n} | ' + ' | '.join(f'{c[s]} ({c[s] / n * 100:.0f}%)' for s in STAGES) + ' |')
    print('\nNMS suppressors:', dict(suppressors.most_common()))
    print('\nbest candidate IoU with GT (any branch), median per bin:',
          {b: round(float(np.median(v)), 2) for b, v in best_any_iou.items()})
    print('best label-0 candidate score (if localized), median per bin:',
          {b: round(float(np.median(v)), 3) for b, v in best_lbl0_scores.items()})


if __name__ == '__main__':
    main()
