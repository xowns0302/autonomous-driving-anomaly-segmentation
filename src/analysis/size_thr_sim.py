import glob, sys, os, numpy as np
sys.path.insert(0, os.path.expanduser('~/projects/autonomous-driving-anomaly-segmentation/src'))
from sweep_ie_nms_bars import bars, ie_nms, iou_matrix
cd, thing = sys.argv[1], float(sys.argv[2])
files = sorted(glob.glob(f'{cd}/*.npz')); bl = bars(0.2, thing)
cache = []
for f in files:
    d = np.load(f); keep = ie_nms(d['bboxes'], d['scores'], d['labels'], bl)
    cache.append((d, keep))
print(f'{os.path.basename(os.path.dirname(cd))} thing={thing}')
print('| small box thr (area<A) | A | box/img | FP/img | recall all | <16² | 16²~32² | ≥32² |')
print('|---|---|---|---|---|---|---|---|')
for A, ts in [(0, 0.2), (32**2, 0.15), (32**2, 0.1), (48**2, 0.15), (48**2, 0.1)]:
    nb = nfp = 0; r = {'a': [0, 0], 's1': [0, 0], 's2': [0, 0], 'l': [0, 0]}
    for d, keep in cache:
        B, S, L = d['bboxes'], d['scores'], d['labels']
        area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        thr = np.where(area < A, ts, 0.2)
        fin = [k for k in keep if L[k] == 0 and S[k] > thr[k]]
        fb = B[fin] if fin else np.zeros((0, 4))
        v = d['gt_areas'] >= 10; gtb, gta = d['gt_boxes'][v], d['gt_areas'][v]
        ious = iou_matrix(fb, gtb); nb += len(fb)
        nfp += int((ious.max(1) < 0.1).sum()) if gtb.size else len(fb)
        best = ious.max(0) if len(fb) else np.zeros(len(gtb))
        for a, b in zip(gta, best):
            k = 's1' if a < 256 else 's2' if a < 1024 else 'l'
            for kk in ('a', k): r[kk][1] += 1; r[kk][0] += int(b >= 0.5)
    pc = lambda x: f'{x[0]/x[1]*100:.1f}' if x[1] else '-'
    print(f'| {ts} | {int(A**0.5) if A else "-"}² | {nb/len(files):.2f} | {nfp/len(files):.2f} | {pc(r["a"])} | {pc(r["s1"])} | {pc(r["s2"])} | {pc(r["l"])} |')
