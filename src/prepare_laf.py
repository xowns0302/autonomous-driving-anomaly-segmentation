"""Prepare the Lost and Found test split as a new held-out set (plan docs/05 §10).

  python src/prepare_laf.py --laf ~/datasets/LostAndFound \
      --out external/DetSeg-official/data/LAF_test

* Drops test frames that are also in FS Lost & Found (our tuning set): images are
  shrunk to 32x16 grayscale and a pair with mean absolute difference < 2/255
  counts as the same picture.
* Labels (gtCoarse labelIds -> {0 road, 1 anomaly, 255 ignore}), §10.1:
    1 (free road)                          -> 0
    2-22, 24-30, 32, 35 (hazard obstacles) -> 1
    everything else -> 255: id 0 (background / unlabeled / ego / border),
      23 bicycle and 40-43 humans (Cityscapes classes), 31,33,34,36-39 non-hazards
* Drops the whole test place that holds those frames (04_Maurener_Weg_8): its other
  frames are the same scenes as the tuning images.
* Writes images/<stem>.png (symlink to the original) and labels_masks/<stem>.png,
  plus overlap.csv and summary.json.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

ANOMALY = set(range(2, 23)) | set(range(24, 31)) | {32, 35}


def thumb(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, (32, 16), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--laf', required=True)
    p.add_argument('--fs', default='external/DetSeg-official/data/FS_LostFound/images')
    p.add_argument('--out', required=True)
    p.add_argument('--drop-site', action='append', default=['04_Maurener_Weg_8'],
                   help='test places that hold FS L&F frames: dropped entirely (plan §10)')
    a = p.parse_args()
    imgs = sorted(glob.glob(f'{a.laf}/leftImg8bit/test/*/*_leftImg8bit.png'))
    fs = sorted(glob.glob(f'{a.fs}/*.png'))
    T = np.stack([thumb(f) for f in imgs])
    F = np.stack([thumb(f) for f in fs])
    overlap, drop = [], set()
    for j, f in enumerate(fs):
        d = np.abs(T - F[j]).mean(axis=(1, 2))
        i = int(d.argmin())
        if d[i] < 2.0:
            overlap.append((os.path.basename(f), os.path.basename(imgs[i]), float(d[i])))
            drop.add(i)
    os.makedirs(f'{a.out}/images', exist_ok=True)
    os.makedirs(f'{a.out}/labels_masks', exist_ok=True)
    kept = no_anom = 0
    sites = {os.path.basename(os.path.dirname(imgs[i])) for i in drop}
    assert sites <= set(a.drop_site), sites
    dropped_site = 0
    for i, f in enumerate(imgs):
        if i in drop:
            continue
        if os.path.basename(os.path.dirname(f)) in a.drop_site:
            dropped_site += 1
            continue
        stem = os.path.basename(f).replace('_leftImg8bit.png', '')
        city = os.path.basename(os.path.dirname(f))
        lab = cv2.imread(f'{a.laf}/gtCoarse/test/{city}/{stem}_gtCoarse_labelIds.png', cv2.IMREAD_UNCHANGED)
        out = np.full(lab.shape, 255, np.uint8)
        out[lab == 1] = 0
        out[np.isin(lab, list(ANOMALY))] = 1
        no_anom += int(not (out == 1).any())
        cv2.imwrite(f'{a.out}/labels_masks/{stem}.png', out)
        dst = f'{a.out}/images/{stem}.png'
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(f), dst)
        kept += 1
    with open(f'{a.out}/overlap.csv', 'w') as fh:
        fh.write('fs_lostfound,laf_test,mean_abs_diff\n')
        fh.writelines(f'{x},{y},{d:.3f}\n' for x, y, d in overlap)
    s = dict(test_images=len(imgs), fs_lostfound=len(fs), overlap=len(overlap), dropped_same_site=dropped_site, kept=kept,
             kept_without_anomaly_pixels=no_anom)
    json.dump(s, open(f'{a.out}/summary.json', 'w'), indent=2)
    print(s)


if __name__ == '__main__':
    main()
