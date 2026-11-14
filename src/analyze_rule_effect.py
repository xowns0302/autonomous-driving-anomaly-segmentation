"""I1: compare final OoD boxes of V1 (paper bars) vs C1 / C2 rules on saved candidates.

Run from the repo root: python src/analyze_rule_effect.py
"""
import sys, glob, json, numpy as np
from collections import Counter
sys.path.insert(0, 'src')
from sweep_ie_nms_bars import ie_nms_rules, CLASSES, iou_matrix
bl = [json.load(open('results/A3/bars_things.json'))['classes'][c].get('bar_min', 0.2) for c in CLASSES[1:]]
for ds in ['fs_lostfound', 'road_anomaly']:
    files = sorted(glob.glob(f'results/A2/{ds}/candidates/*.npz'))
    print(f'### {ds}')
    base_sets = {}
    for name, kw in [('V1', {}), ('C1', dict(background_exempt=True)), ('C2 g0.8', dict(protect_gamma=0.8))]:
        n = fp = 0; areas = []; extra = Counter(); extra_area = []; extra_score = []; extra_fp = 0; extra_n = 0
        for fi, f in enumerate(files):
            d = np.load(f); B, S, L = d['bboxes'], d['scores'], d['labels']
            keep = ie_nms_rules(B, S, L, bl, **kw)
            fin = set(k for k in keep if L[k] == 0 and S[k] > 0.2)
            if name == 'V1': base_sets[fi] = fin
            gtb = d['gt_boxes'][d['gt_areas'] >= 10]
            for k in fin:
                a = (B[k,2]-B[k,0])*(B[k,3]-B[k,1]); n += 1; areas.append(a)
                isfp = (iou_matrix(B[k:k+1], gtb).max() < 0.1) if gtb.size else True
                fp += isfp
                if name != 'V1' and k not in base_sets[fi]:
                    extra_n += 1; extra_fp += isfp; extra_area.append(a); extra_score.append(S[k])
        msg = f'  {name}: final boxes {n} ({n/len(files):.1f}/img), FP {fp}, median area {np.median(areas):.0f}'
        if extra_n:
            msg += f' | NEW boxes vs V1: {extra_n} (FP {extra_fp}), area median {np.median(extra_area):.0f} p90 {np.percentile(extra_area,90):.0f}, score median {np.median(extra_score):.2f}'
        print(msg)
