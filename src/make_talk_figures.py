"""Explanatory figures for the 5-minute talk, drawn from real outputs.

Inputs: A2 candidate dumps (boxes before IE-NMS), A3 bars, M2A score maps,
F1 masks, GT masks. Boxes are re-run through the same IE-NMS code
(sweep_ie_nms_bars.ie_nms_rules) so every drawn box is what the method keeps
or deletes.

  conda run -n detseg python src/make_talk_figures.py   -> assets/figures/talk/
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from sweep_ie_nms_bars import CLASSES, ie_nms_rules, iou_matrix  # noqa: E402

D = 'external/DetSeg-official/data'
OUT = 'assets/figures/talk'
FONT = os.path.expanduser('~/.local/share/fonts/NanumGothic-Bold.ttf')
ORANGE, BLUE, GRAY, GREEN, RED = (234, 120, 30), (40, 90, 200), (150, 150, 150), (40, 200, 70), (220, 40, 40)
THING = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')


def bars(kind):
    if kind == 'released':
        return [0.2] * 11 + [0.5] * 8
    cal = json.load(open('results/A3/bars_things.json'))['classes']
    return [cal[c]['bar_min'] if c in THING else 0.2 for c in CLASSES[1:]]


def load(ds, name):
    import glob
    for f in sorted(glob.glob(f'results/A2/{ds}/candidates/*.npz')):
        d = np.load(f, allow_pickle=True)
        if str(d['img_path']).endswith('/' + name):
            return d
    raise FileNotFoundError(name)


def font(size):
    return ImageFont.truetype(FONT, size)


def draw_box(dr, b, color, width=6, label=None, size=30, dashed=False, below=False):
    x1, y1, x2, y2 = [float(v) for v in b]
    if dashed:
        step = 18
        for x in np.arange(x1, x2, step * 2):
            dr.line([(x, y1), (min(x + step, x2), y1)], fill=color, width=width)
            dr.line([(x, y2), (min(x + step, x2), y2)], fill=color, width=width)
        for y in np.arange(y1, y2, step * 2):
            dr.line([(x1, y), (x1, min(y + step, y2))], fill=color, width=width)
            dr.line([(x2, y), (x2, min(y + step, y2))], fill=color, width=width)
    else:
        dr.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        f = font(size)
        w = dr.textlength(label, font=f)
        W, H = dr._image.size
        ty = y1 - size - 10 if (y1 - size - 10 > 0 and not below) else y2 + 4
        ty = min(ty, H - size - 10)
        tx = min(x1, W - w - 20)
        dr.rectangle([tx, ty, tx + w + 16, ty + size + 8], fill=color)
        dr.text((tx + 8, ty + 2), label, font=f, fill=(255, 255, 255))


def save(im, name, width=960):
    h = int(im.height * width / im.width)
    im.resize((width, h), Image.LANCZOS).convert('RGB').save(f'{OUT}/{name}', quality=90)
    print(name, (width, h))


def overlay(img, mask, color, alpha=0.55):
    a = np.array(img).astype(np.float32)
    m = mask.astype(bool)
    a[m] = a[m] * (1 - alpha) + np.array(color) * alpha
    return Image.fromarray(a.astype(np.uint8))


def gt_mask(path, ood=1):
    g = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return g == ood, g == 255


def run_nms(d, kind, m2a=None):
    B, S, L = d['bboxes'], d['scores'], d['labels']
    vs = None
    if m2a is not None:
        vs = np.array([box_mean(m2a, b) for b in B])
    keep = ie_nms_rules(B, S, L, bars(kind), verify_scores=vs, verify_t=0.5 if vs is not None else None)
    return set(int(k) for k in keep)


def box_mean(m, b):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    x1, y1 = max(x1, 0), max(y1, 0)
    sub = m[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    return float(sub.mean()) if sub.size else 0.0


def un_candidates(d, thr=0.2, top=6, gt=None):
    """Label-0 boxes above tau, deduplicated among themselves (for drawing).
    Boxes covering the GT object are always included."""
    B, S, L = d['bboxes'], d['scores'], d['labels']
    idx = [i for i in np.argsort(-S) if L[i] == 0 and S[i] > thr]
    if gt is not None:
        idx = [i for i in idx if on_gt(B[i], gt)] + [i for i in idx if not on_gt(B[i], gt)]
    kept = []
    for i in idx:
        if all(iou_matrix(B[[i]], B[[j]])[0, 0] < 0.5 for j in kept):
            kept.append(i)
    n_gt = sum(on_gt(B[i], gt) for i in kept) if gt is not None else 0
    return kept[:top + n_gt]


def on_gt(b, gt):
    x1, y1, x2, y2 = [int(v) for v in b]
    sub = gt[max(y1, 0):y2, max(x1, 0):x2]
    return sub.size > 0 and sub.mean() > 0.2


def best_id_over(d, i, kind):
    """The ID box (label>0) that overlaps box i most (IoU>=0.5) and outranks it."""
    B, S, L = d['bboxes'], d['scores'], d['labels']
    bl = bars(kind)
    best = None
    for j in np.where(L > 0)[0]:
        if iou_matrix(B[[i]], B[[j]])[0, 0] >= 0.5:
            s = S[j] * (2 if S[j] > bl[L[j] - 1] else 1)
            if best is None or s > best[1]:
                best = (int(j), float(s))
    return best


def fig_pipeline():
    """FS L&F 60 with paper-style bars: candidates -> IE-NMS -> final mask."""
    name = '60.png'
    d = load('fs_lostfound', name)
    img = Image.open(f'{D}/FS_LostFound/images/{name}').convert('RGB')
    gt, _ = gt_mask(f'{D}/FS_LostFound/labels_masks/{name}')
    save(img, 'pipe_0_input.jpg')
    cand = un_candidates(d, top=6, gt=gt)
    keep = run_nms(d, 'paper')
    print('pipeline', [(int(i), on_gt(d['bboxes'][i], gt), i in keep) for i in cand])
    im = img.copy(); dr = ImageDraw.Draw(im)
    for i in cand:
        draw_box(dr, d['bboxes'][i], ORANGE, 12)
    for i in cand:
        bj = best_id_over(d, i, 'paper')
        if bj and i not in keep:
            draw_box(dr, d['bboxes'][bj[0]], BLUE, 10, f"{CLASSES[d['labels'][bj[0]]]} {d['scores'][bj[0]]:.2f}", 64)
    save(im, 'pipe_1_candidates.jpg')
    im = img.copy(); dr = ImageDraw.Draw(im)
    for i in cand:
        if i in keep:
            draw_box(dr, d['bboxes'][i], ORANGE, 14)
        else:
            draw_box(dr, d['bboxes'][i], GRAY, 9, dashed=True)
            x1, y1, x2, y2 = d['bboxes'][i]
            dr.line([(x1, y1), (x2, y2)], fill=GRAY, width=8); dr.line([(x1, y2), (x2, y1)], fill=GRAY, width=8)
    save(im, 'pipe_2_after_nms.jpg')
    m = cv2.imread('results/F1/masks/fs_lostfound/paper/60.png', 0) > 127
    save(overlay(img, m, RED, 0.6), 'pipe_3_mask.jpg')
    for n in ('pipe_1_candidates', 'pipe_2_after_nms', 'pipe_3_mask'):
        Image.open(f'{OUT}/{n}.jpg').crop((80, 100, 880, 480)).save(f'{OUT}/{n}_crop.jpg', quality=92)


def fig_cause():
    """FS L&F 65: public bar keeps parked-car boxes; calibrated bar removes them
    but also deletes the toy car (mistaken for an ID class)."""
    name = '65.png'
    d = load('fs_lostfound', name)
    B, S, L = d['bboxes'], d['scores'], d['labels']
    img = Image.open(f'{D}/FS_LostFound/images/{name}').convert('RGB')
    gt, _ = gt_mask(f'{D}/FS_LostFound/labels_masks/{name}')
    cand = un_candidates(d, top=20, gt=gt)
    keep_pub = run_nms(d, 'released')
    toy = [i for i in cand if on_gt(B[i], gt)]
    others = [i for i in cand if not on_gt(B[i], gt) and i in keep_pub][:5]
    big = max(others, key=lambda i: (B[i][2] - B[i][0]) * (B[i][3] - B[i][1]))
    for kind in ('released', 'paper'):
        keep = run_nms(d, kind)
        bl = bars(kind)
        print(kind, 'toy', [(int(i), i in keep, best_id_over(d, i, kind)) for i in toy],
              'others', [(int(i), i in keep) for i in others])
        im = img.copy(); dr = ImageDraw.Draw(im)
        for i in others:
            if i in keep:
                draw_box(dr, B[i], ORANGE, 7)
            else:
                draw_box(dr, B[i], GRAY, 5, dashed=True)
        bj = best_id_over(d, big, kind)
        if bj:
            j = bj[0]; c = CLASSES[L[j]]
            if big in keep:
                draw_box(dr, B[j], BLUE, 5, f"{c} {S[j]:.2f} < 기준선 {bl[L[j]-1]:.2f} → 못 지움", 30)
            else:
                draw_box(dr, B[j], BLUE, 5, f"{c} {S[j]:.2f} > 기준선 {bl[L[j]-1]:.3f} → 지움", 30)
        for i in toy:
            if i in keep:
                draw_box(dr, B[i], ORANGE, 8, '장난감 차: 찾음', 30)
            else:
                tj = best_id_over(d, i, kind)
                draw_box(dr, B[i], GRAY, 6, dashed=True)
                if tj:
                    draw_box(dr, B[tj[0]], BLUE, 6, f"{CLASSES[L[tj[0]]]} {S[tj[0]]:.2f} → 장난감 차 지움", 30)
        save(im, f'cause_{kind}.jpg')


def fig_c5():
    """SMIYC Anomaly 5: 'truck' box vs trailer box, M2A heatmap, C5 result."""
    name = '5.png'
    d = load('smiyc_anomaly', name)
    img = Image.open(f'{D}/SMIYC_Anomaly_val/images/{name}').convert('RGB')
    m2a = np.load(f'results/score_maps/m2a/SMIYC_Anomaly_val/{name}.npy').astype(np.float32)
    if m2a.shape != (img.height, img.width):
        m2a = cv2.resize(m2a, (img.width, img.height))
    gt, _ = gt_mask(f'{D}/SMIYC_Anomaly_val/labels_masks/{name}')
    keep_p = run_nms(d, 'paper')
    keep_c5 = run_nms(d, 'paper', m2a)
    B = d['bboxes']
    # the trailer candidate: label-0 box with the largest overlap with the GT
    best, bi = -1, None
    for i in np.where((d['labels'] == 0) & (d['scores'] > 0.2))[0]:
        x1, y1, x2, y2 = [int(v) for v in B[i]]
        inter = gt[y1:y2, x1:x2].sum(); union = gt.sum() + (x2 - x1) * (y2 - y1) - inter
        if inter / union > best:
            best, bi = inter / union, int(i)
    bj = best_id_over(d, bi, 'paper')
    mean = box_mean(m2a, B[bi])
    print('trailer box', bi, round(float(d['scores'][bi]), 3), 'kept paper', bi in keep_p, 'kept c5', bi in keep_c5,
          'id', CLASSES[d['labels'][bj[0]]], round(float(d['scores'][bj[0]]), 3), 'm2a mean', round(mean, 3))
    im = img.copy(); dr = ImageDraw.Draw(im)
    draw_box(dr, B[bi], ORANGE, 10, '이상 물체 후보 (0.75)', 30)
    draw_box(dr, B[bj[0]], BLUE, 5, f"검출 AI: “{CLASSES[d['labels'][bj[0]]]}” {d['scores'][bj[0]]:.2f} → 지우려 함", 30, below=True)
    save(im, 'c5_1_boxes.jpg')
    heat = cv2.applyColorMap((np.clip(m2a, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    mix = Image.fromarray((0.35 * np.array(img) + 0.65 * heat).astype(np.uint8))
    dr = ImageDraw.Draw(mix)
    draw_box(dr, B[bi], ORANGE, 8, f'상자 안 평균 {mean:.2f} ≥ 0.5', 30)
    save(mix, 'c5_2_heatmap.jpg')
    for k in ('paper', 'c5'):
        m = cv2.imread(f'results/F1/masks/smiyc_anomaly/{k}/5.png', 0) > 127
        save(overlay(img, m, RED, 0.6), f'c5_3_{k}.jpg')


def fig_pixel_method():
    """Pixel-score method (M2A) on FS L&F 60: heat map and thresholds.
    Void (not scored) is darkened; red only where it counts."""
    name = '60.png'
    img = Image.open(f'{D}/FS_LostFound/images/{name}').convert('RGB')
    _, void = gt_mask(f'{D}/FS_LostFound/labels_masks/{name}')
    m2a = np.load(f'results/score_maps/m2a/FS_LostFound/{name}.npy').astype(np.float32)
    if m2a.shape != (img.height, img.width):
        m2a = cv2.resize(m2a, (img.width, img.height))
    base = np.array(img).astype(np.float32)
    base[void] *= 0.3
    dim = Image.fromarray(base.astype(np.uint8))
    heat = cv2.applyColorMap((np.clip(m2a, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1].astype(np.float32)
    mix = 0.3 * base + 0.7 * heat
    mix[void] = base[void]
    save(Image.fromarray(mix.astype(np.uint8)), 'old_heat.jpg')
    for t in (0.3, 0.5, 0.7):
        save(overlay(dim, (m2a > t) & ~void, RED, 0.75), f'old_t{int(t * 10)}.jpg')
    m = cv2.imread('results/F1/masks/fs_lostfound/paper/60.png', 0) > 127
    save(overlay(dim, m & ~void, RED, 0.75), 'new_mask.jpg')


def fig_iou():
    """Overlap score illustration on SMIYC Anomaly 5 (paper bars vs C5)."""
    name = '5.png'
    img = np.array(Image.open(f'{D}/SMIYC_Anomaly_val/images/{name}').convert('RGB')).astype(np.float32)
    gt, void = gt_mask(f'{D}/SMIYC_Anomaly_val/labels_masks/{name}')
    for k in ('paper', 'c5'):
        p = cv2.imread(f'results/F1/masks/smiyc_anomaly/{k}/5.png', 0) > 127
        a = img * 0.35
        a[gt & p] = np.array((245, 200, 40))
        a[gt & ~p] = np.array(GREEN)
        a[~gt & p & ~void] = np.array(RED)
        inter, union = (gt & p).sum(), (gt | (p & ~void)).sum()
        print('iou', k, round(100 * inter / union, 1))
        save(Image.fromarray(a.astype(np.uint8)), f'iou_{k}.jpg')


def fig_thumbs():
    """Dataset thumbnails cut from the qualitative figures (GT tiles)."""
    for src, row, th, out in [('qual_fs_lostfound.jpg', 0, 320, 'thumb_fs.jpg'),
                              ('qual_road_anomaly.jpg', 0, 360, 'thumb_ra.jpg'),
                              ('qual_smiyc_obstacle.jpg', 0, 360, 'thumb_smiyc_o.jpg'),
                              ('qual_smiyc_anomaly.jpg', 1, 360, 'thumb_smiyc_a.jpg')]:
        im = Image.open(f'assets/figures/{src}')
        im.crop((0, row * th + 28, 640, (row + 1) * th)).save(f'{OUT}/{out}', quality=90)
        print(out)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_pipeline()
    fig_cause()
    fig_c5()
    fig_pixel_method()
    fig_iou()
    fig_thumbs()
