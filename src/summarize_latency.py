"""Summarise a latency run folder (scripts/latency_baseline.sh) as markdown tables.

Pipelines that need Mask2Anomaly (DetSeg-R, C5) are reported with and without
the M2A score-map time, because the profilers read precomputed score maps.

  python src/summarize_latency.py results/T2 > results/T2/summary.md
"""
import json
import os
import sys

import numpy as np

DATASETS = ['fs_lostfound', 'road_anomaly', 'smiyc_obstacle', 'smiyc_anomaly']
METHODS = [('S_released', 'DetSeg-S (공개본)', False), ('S_paper', 'DetSeg-S (논문 방식 bar)', False),
           ('S_c5', 'DetSeg-S + C5', True), ('S_c7', 'DetSeg-S + C7', False), ('S_c8', 'DetSeg-S + C8 (C5+C7)', True),
           ('R_released', 'M2A + DetSeg-R (공개본)', True), ('R_c5', 'M2A + DetSeg-R + C5', True),
           ('R_c8', 'M2A + DetSeg-R + C8', True)]
DET = ['backbone', 'neck', 'text', 'encoder', 'query_sel', 'decoder', 'head_excl_ie_nms', 'ie_nms']
SAM = ['sam_total', 'sam_prep', 'sam_image_encoder', 'sam_mask_decoder', 'sam_post']


def load(d, name):
    p = os.path.join(d, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def ms(v):
    return f'{np.mean(v):.0f} ± {np.std(v):.0f}'


def main():
    d = sys.argv[1]
    print(f'# Latency summary: {d}\n')
    env = os.path.join(d, 'environment.txt')
    if os.path.exists(env):
        print('```\n' + open(env).read().strip() + '\n```\n')
    print('30 images after 5 warm-up, 3 repeats, batch 1. Values: mean ± std over repeats (ms/image).\n')
    print('| Dataset | 방법 | DetSeg 모델 | 데이터 읽기 | M2A 점수 지도 | **이미지당 합계** | 검출 | SAM | C5 | C7 (DINOv2) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for ds in DATASETS:
        m2a = load(d, f'M2A_{ds}')
        m2a_runs = m2a['run_needed_total'] if m2a else [0.0]
        print(f'| {ds} | M2A 단독 | – | – | {ms(m2a_runs)} | **{ms(m2a_runs)}** | – | – | – | – |')
        for key, label, needs_m2a in METHODS:
            r = load(d, f'{key}_{ds}')
            if r is None:
                continue
            mm = r['mean_ms']
            model, data = r['run_means']['model_total'], r['run_means']['data']
            tot = [a + b + (np.mean(m2a_runs) if needs_m2a else 0) for a, b in zip(model, data)]
            det = sum(mm.get(k, 0) for k in DET)
            sam = sum(mm.get(k, 0) for k in SAM)
            c5 = mm.get('c5_box_means', 0)
            m2a_s = f'{np.mean(m2a_runs):.0f}' if needs_m2a else '–'
            sam_s = f'{sam:.0f}' if sam else '–'
            c5_s = f'{c5:.1f}' if c5 else '–'
            c7 = mm.get('c7_prowl', 0)
            c7_s = f'{c7:.1f}' if c7 else '–'
            det -= c5 + c7      # ie_nms time includes the C5 / C7 work
            print(f'| {ds} | {label} | {ms(model)} | {ms(data)} | {m2a_s} | **{ms(tot)}** | {det:.0f} | {sam_s} | {c5_s} | {c7_s} |')
    print('\n## 단계별 (DetSeg-S 공개본)\n')
    print('| 단계 | ' + ' | '.join(DATASETS) + ' |')
    print('|---|' + '---|' * len(DATASETS))
    rows = {ds: load(d, f'S_released_{ds}') for ds in DATASETS}
    for k in ['data'] + DET + SAM + ['other_in_model', 'model_total', 'total']:
        vals = [f'{rows[ds]["mean_ms"].get(k, 0):.1f}' if rows[ds] else '' for ds in DATASETS]
        print(f'| {k} | ' + ' | '.join(vals) + ' |')


if __name__ == '__main__':
    main()
