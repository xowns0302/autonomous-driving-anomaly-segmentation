import glob, sys, os, numpy as np
sys.path.insert(0, os.path.expanduser('~/projects/autonomous-driving-anomaly-segmentation/src'))
from sweep_ie_nms_bars import bars, ie_nms, iou_matrix
cd = sys.argv[1]; bl = bars(0.2, 0.2)
cache = []
for f in sorted(glob.glob(f'{cd}/*.npz')):
    d = np.load(f); cache.append((d, ie_nms(d['bboxes'], d['scores'], d['labels'], bl)))
print(os.path.basename(os.path.dirname(cd)), '| base = thing .2 + small(<32²) .1')
print('| road rule | box/img | FP/img | recall | <32² recall | ≥32² recall |'); print('|---|---|---|---|---|---|')
for name, mode, alpha in [('none', None, 1), ('hard: drop off-road', 'hard', 0), ('soft x0.5 off-road', 'soft', 0.5), ('soft x0.7 off-road', 'soft', 0.7), ('hard, small boxes only', 'hard_small', 0)]:
    nb = nfp = 0; r = {'a': [0, 0], 's': [0, 0], 'l': [0, 0]}
    for d, keep in cache:
        B, S, L = d['bboxes'], d['scores'], d['labels']
        area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]); small = area < 1024
        road = B[(L == 1) & (S > 0.2) & d['is_id_row']]
        fx, fy = (B[:, 0] + B[:, 2]) / 2, B[:, 3] - 1
        on = np.array([((road[:, 0] <= x) & (x <= road[:, 2]) & (road[:, 1] <= y) & (y <= road[:, 3])).any() for x, y in zip(fx, fy)]) if len(road) else np.ones(len(B), bool)  # no road box -> no filtering
        s = S.copy()
        if mode == 'soft': s = np.where(on, S, S * alpha)
        thr = np.where(small, 0.1, 0.2)
        ok = (L == 0) & (s > thr)
        if mode == 'hard': ok &= on
        if mode == 'hard_small': ok &= on | ~small
        fin = [k for k in keep if ok[k]]
        fb = B[fin] if fin else np.zeros((0, 4))
        v = d['gt_areas'] >= 10; gtb, gta = d['gt_boxes'][v], d['gt_areas'][v]
        ious = iou_matrix(fb, gtb); nb += len(fb); nfp += int((ious.max(1) < 0.1).sum()) if gtb.size else len(fb)
        best = ious.max(0) if len(fb) else np.zeros(len(gtb))
        for a, b in zip(gta, best):
            for k in ('a', 's' if a < 1024 else 'l'): r[k][1] += 1; r[k][0] += int(b >= 0.5)
    pc = lambda x: f'{x[0]/x[1]*100:.1f}'
    print(f'| {name} | {nb/len(cache):.2f} | {nfp/len(cache):.2f} | {pc(r["a"])} | {pc(r["s"])} | {pc(r["l"])} |')
