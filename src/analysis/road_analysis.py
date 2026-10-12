import glob, sys, os, numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser('~/projects/autonomous-driving-anomaly-segmentation/src'))
from sweep_ie_nms_bars import bars, ie_nms, iou_matrix
cd = sys.argv[1]; root = sys.argv[2]
bl = bars(0.2, 0.2)
def trapezoid(x, y, W, H, yh=0.45, top_half=0.08, bot_half=0.6):
    t = (y / H - yh) / (1 - yh)
    return (t >= 0) & (np.abs(x / W - 0.5) <= top_half + (bot_half - top_half) * np.clip(t, 0, 1))
stats = {k: {'TP': [0, 0, 0], 'FP': [0, 0, 0]} for k in ('all', 'small')}  # [n, on_roadbox, on_trap]
gt_stats = [0, 0, 0]
for f in sorted(glob.glob(f'{cd}/*.npz')):
    d = np.load(f); B, S, L = d['bboxes'], d['scores'], d['labels']
    W, H = Image.open(os.path.join(root, str(d['img_path']))).size
    keep = ie_nms(B, S, L, bl)
    area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    fin = [k for k in keep if L[k] == 0 and S[k] > (0.1 if area[k] < 1024 else 0.2)]
    road = B[(L == 1) & (S > 0.2) & d['is_id_row']]
    fb = B[fin] if fin else np.zeros((0, 4))
    v = d['gt_areas'] >= 10; gtb = d['gt_boxes'][v]
    gi = iou_matrix(fb, gtb).max(1) if gtb.size and len(fb) else np.zeros(len(fb))
    fx, fy = (fb[:, 0] + fb[:, 2]) / 2, fb[:, 3] - 1
    on_rb = np.array([((road[:, 0] <= x) & (x <= road[:, 2]) & (road[:, 1] <= y) & (y <= road[:, 3])).any() for x, y in zip(fx, fy)], bool) if len(road) else np.zeros(len(fb), bool)
    on_tr = trapezoid(fx, fy, W, H)
    for j in range(len(fb)):
        t = 'TP' if gi[j] >= 0.1 else 'FP'
        for k in ('all',) + (('small',) if area[fin[j]] < 1024 else ()):
            st = stats[k][t]; st[0] += 1; st[1] += int(on_rb[j]); st[2] += int(on_tr[j])
    gx, gy = (gtb[:, 0] + gtb[:, 2]) / 2, gtb[:, 3] - 1
    grb = [((road[:, 0] <= x) & (x <= road[:, 2]) & (road[:, 1] <= y) & (y <= road[:, 3])).any() for x, y in zip(gx, gy)] if len(road) else [False] * len(gtb)
    gt_stats[0] += len(gtb); gt_stats[1] += int(np.sum(grb)); gt_stats[2] += int(trapezoid(gx, gy, W, H).sum())
print(os.path.basename(os.path.dirname(cd)), '(thing .2 + small .1 final boxes)')
print('| boxes | n | foot on GDINO road box | foot in trapezoid prior |')
print('|---|---|---|---|')
for k in ('all', 'small'):
    for t in ('TP', 'FP'):
        n, a, b = stats[k][t]
        if n: print(f'| {k} {t} | {n} | {a/n*100:.0f}% | {b/n*100:.0f}% |')
n, a, b = gt_stats; print(f'| GT components | {n} | {a/n*100:.0f}% | {b/n*100:.0f}% |')
