import json, sys, numpy as np
rows = json.load(open(sys.argv[1]))
gp = np.array([r['gt_pixels'] for r in rows]); pp = np.array([r['pred_pixels'] for r in rows])
print('pred px / gt px (median per image):', np.median(pp / np.maximum(gp, 1)).round(2), '| mean pred px', int(pp.mean()), '| mean gt px', int(gp.mean()))
nb = np.array([r['n_boxes'] for r in rows]); print('boxes/img pct[0,25,50,75,100]:', np.percentile(nb, [0, 25, 50, 75, 100]))
def iou(b, c):
    x1, y1, x2, y2 = max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i + 1e-6)
tp_s, fp_s, fp_a, tp_a = [], [], [], []
for r in rows:
    for b, s, a in zip(r['boxes'], r['scores'], r['box_areas']):
        m = max([iou(b, c['box']) for c in r['gt_components']] + [0])
        (tp_s if m >= 0.1 else fp_s).append(s); (tp_a if m >= 0.1 else fp_a).append(a)
print(f'boxes overlapping GT (IoU>=0.1): {len(tp_s)} | not overlapping (FP-like): {len(fp_s)}  -> FP share {len(fp_s)/(len(tp_s)+len(fp_s))*100:.1f}%')
print('score pct[10,50,90] TP:', np.percentile(tp_s, [10, 50, 90]).round(3), ' FP:', np.percentile(fp_s, [10, 50, 90]).round(3))
print('area  pct[10,50,90] TP:', np.percentile(tp_a, [10, 50, 90]).round(0), ' FP:', np.percentile(fp_a, [10, 50, 90]).round(0))
for thr in [0.2, 0.3, 0.4, 0.5]:
    print(f'  if score>{thr}: TP-like {sum(s>thr for s in tp_s)}, FP-like {sum(s>thr for s in fp_s)}')
print('gt component area pct[10,50,90]:', np.percentile([c['area'] for r in rows for c in r['gt_components']], [10, 50, 90]).round())
