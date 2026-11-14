"""Demo page for the talk: per-experiment outputs side by side, one scene per screen.

Uses the masks saved by scripts/save_masks_for_figures.sh (results/F1/masks). Scenes per dataset:
the images named in the talk (FS 7 parked car, SMIYC Anomaly 9 tractor and 5 trailer), then by
fixed rules the 5 largest IoU gains of C9 over the baseline and the 2 largest losses of C9 vs C5
(failure case). Writes one offline HTML file with the images embedded (opens without internet).

  python src/make_demo_page.py   # -> results/demo/demo.html
"""
import base64
import html
import json
import os

import cv2
import numpy as np

from make_qualitative import DATASETS, iou

METHODS = [('released', '공개 DetSeg'), ('paper', '베이스라인'), ('c5', '1단계 C5'),
           ('c8', '1+2단계 C8'), ('c9', '최종 C9')]
NAMES = {'fs_lostfound': 'FS Lost & Found (연습용)', 'road_anomaly': 'Road Anomaly (연습용)',
         'smiyc_obstacle': 'SMIYC Obstacle (검증용)', 'smiyc_anomaly': 'SMIYC Anomaly (검증용)'}
NAMED = {('fs_lostfound', '7'): '2단계: 주차 차 오탐 삭제',
         ('smiyc_anomaly', '9'): '1단계: 트랙터 지키기',
         ('smiyc_anomaly', '5'): '3단계: 트레일러 거부권'}


def overlay(img, gt, pred, width=720):
    over = img.copy()
    void = gt == 255
    over[void] = (0.45 * img[void]).astype(np.uint8)
    over[pred] = (0.45 * img[pred] + np.array([0, 0, 140])).astype(np.uint8)
    cnt, _ = cv2.findContours((gt == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnt, -1, (0, 255, 0), max(2, img.shape[1] // 500))
    t = cv2.resize(over, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', t, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()


def main():
    scenes = []
    for ds, (img_t, gt_t, ood) in DATASETS.items():
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f'results/F1/masks/{ds}/c9'))
        gts, scores = {}, {}
        for s in stems:
            gt = cv2.imread(gt_t.format(s), cv2.IMREAD_UNCHANGED)
            gts[s] = np.where(gt == ood, 1, np.where(gt == 255, 255, 0))
            scores[s] = {m: iou(gts[s], cv2.imread(f'results/F1/masks/{ds}/{m}/{s}.png', 0) > 127) for m, _ in METHODS}
        picks = [(s, why) for (d, s), why in NAMED.items() if d == ds]
        gain = [s for s in sorted(stems, key=lambda s: scores[s]['c9'] - scores[s]['paper'], reverse=True)
                if scores[s]['c9'] - scores[s]['paper'] > 0.01]
        picks += [(s, '베이스라인 대비 개선 큰 사진') for s in gain if s not in dict(picks)][:5]
        loss = [s for s in sorted(stems, key=lambda s: scores[s]['c9'] - scores[s]['c5'])
                if scores[s]['c9'] - scores[s]['c5'] < -0.01]
        picks += [(s, '실패 사례: 1단계보다 나빠짐') for s in loss if s not in dict(picks)][:2]
        for s, why in picks:
            img = cv2.imread(img_t.format(s))
            tiles = [{'label': '입력 + 정답(초록 선)', 'src': overlay(img, gts[s], np.zeros(gts[s].shape, bool)), 'iou': None}]
            for m, lab in METHODS:
                pred = cv2.imread(f'results/F1/masks/{ds}/{m}/{s}.png', 0) > 127
                tiles.append({'label': lab, 'src': overlay(img, gts[s], pred), 'iou': round(scores[s][m] * 100, 1)})
            scenes.append({'ds': NAMES[ds], 'img': s, 'why': why, 'tiles': tiles})
            print(ds, s, why, {m: round(v * 100, 1) for m, v in scores[s].items()})
    os.makedirs('results/demo', exist_ok=True)
    with open('src/demo_template.html', encoding='utf-8') as f:
        page = f.read().replace('__SCENES__', json.dumps(scenes, ensure_ascii=False))
    with open('results/demo/demo.html', 'w', encoding='utf-8') as f:
        f.write(page)
    print('results/demo/demo.html', len(scenes), 'scenes', os.path.getsize('results/demo/demo.html') // 1024, 'KB')


if __name__ == '__main__':
    main()
