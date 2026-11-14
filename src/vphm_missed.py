"""L2 follow-up: why VPHM barely changes the missed count. For every GT object that is missed
after VPHM, check the best candidate box (any score > 0.05) covering it: no box at all, a box
that VPHM never matched (no partner in t-1), or a matched box whose smoothed score stays <= tau.
  python src/vphm_missed.py results/L2/boxes_released"""
import collections
import sys

import cv2
import numpy as np

import vphm


def main():
    seqs = vphm.load_seq(sys.argv[1])
    vphm.smooth(seqs)
    cnt = collections.Counter()
    for frames in seqs.values():
        for k, f in enumerate(frames):
            g = cv2.imread(f'{vphm.GT}/{f["stem"]}.png', cv2.IMREAD_UNCHANGED) == 1
            for o in vphm.components(g):
                cover = []
                for b, s, sm in zip(f['boxes'].astype(int), f['scores'], f['smoothed']):
                    m = np.zeros_like(o); m[max(b[1], 0):b[3], max(b[0], 0):b[2]] = True
                    if o[m].sum() >= 0.5 * o.sum():
                        cover.append((s, sm))
                if any(sm > vphm.TAU for _, sm in cover):
                    cnt['found'] += 1
                elif not any(s > 0.05 for s, _ in cover):
                    cnt['no candidate box (score <= 0.05)'] += 1
                elif k == 0:
                    cnt['box exists, first frame of sequence'] += 1
                elif all(sm == s for s, sm in cover):
                    prev = frames[k - 1]; vp = vphm.vanishing_point(f['stem'])
                    pc = prev['boxes'][prev['smoothed'] > 0.05]
                    best = max((vphm.iou(pb, b) for b, (s, _) in zip(f['boxes'], [(x, 0) for x in f['scores']])
                                if s > 0.05 for a in pc for pb in [vphm.pseudo_box(a, b, vp)] if pb is not None), default=0)
                    if len(pc) == 0:
                        cnt['not matched: no box at all in t-1'] += 1
                    elif best >= 0.3:
                        cnt['not matched: best fan IoU 0.3-0.5 (just below 0.5)'] += 1
                    else:
                        cnt['not matched: fan IoU < 0.3 (no partner in t-1)'] += 1
                else:
                    cnt['matched, smoothed score still <= tau'] += 1
    for k, v in cnt.most_common():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
