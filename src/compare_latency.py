"""Compare two latency folders (T2 baseline vs T3 speed-ups), end-to-end ms/image incl. M2A where needed.

  python src/compare_latency.py   (reads results/T2 and results/T3)
"""
import json, numpy as np
M = ['S_released','S_paper','S_c5','S_c7','S_c8','R_released','R_c5','R_c8']
DS = ['fs_lostfound','road_anomaly','smiyc_obstacle','smiyc_anomaly']
m2a = {ds: np.mean(json.load(open(f'results/T2/M2A_{ds}.json'))['run_needed_total']) for ds in DS}
need = {'S_c5','S_c8','R_released','R_c5','R_c8'}
print('| method | ' + ' | '.join(DS) + ' |')
for m in M:
    row = []
    for ds in DS:
        a = json.load(open(f'results/T2/{m}_{ds}.json')); b = json.load(open(f'results/T3/{m}_{ds}.json'))
        add = m2a[ds] if m in need else 0
        ta = np.mean(a['run_means']['model_total']) + np.mean(a['run_means']['data']) + add
        tb = np.mean(b['run_means']['model_total']) + np.mean(b['run_means']['data']) + add
        row.append(f'{ta:.0f} -> {tb:.0f} ({(tb-ta)/ta*100:+.0f}%)')
    print(f'| {m} | ' + ' | '.join(row) + ' |')
for ds in ['fs_lostfound', 'smiyc_obstacle']:
    b = json.load(open(f'results/T3/S_released_{ds}.json'))['mean_ms']; a = json.load(open(f'results/T2/S_released_{ds}.json'))['mean_ms']
    print(ds, {k: (round(a.get(k,0),1), round(b.get(k,0),1)) for k in ['text','sam_total','data','other_in_model','model_total']},
          'T2 sam sum', round(sum(a.get(k,0) for k in ['sam_prep','sam_image_encoder','sam_mask_decoder','sam_post']),1))
    c = json.load(open(f'results/T3/S_c8_{ds}.json'))['mean_ms']; d = json.load(open(f'results/T2/S_c8_{ds}.json'))['mean_ms']
    print('  c8 c7_prowl', round(d['c7_prowl'],1), '->', round(c['c7_prowl'],1), '| ie_nms', round(d['ie_nms'],1), '->', round(c['ie_nms'],1))
