import numpy as np, glob, os, sys
from PIL import Image
root = sys.argv[1]
for name, pat in [('FS_LostFound', 'Validation_Dataset/FS_LostFound_full/labels_masks/*.png'),
                  ('RoadAnomaly(M2A)', 'Validation_Dataset/RoadAnomaly/labels_masks/*.png'),
                  ('fs_static', 'Validation_Dataset/fs_static/labels_masks/*.png'),
                  ('RoadObsticle21', 'Validation_Dataset/RoadObsticle21/labels_masks/*.png'),
                  ('RoadAnomaly21', 'Validation_Dataset/RoadAnomaly21/labels_masks/*.png'),
                  ('RoadAnomaly(EPFL)', 'RoadAnomaly_dl/RoadAnomaly/frames/*.labels/labels_semantic.png')]:
    files = sorted(glob.glob(os.path.join(root, pat))); vals = set(); noood = 0; sizes = set()
    for p in files:
        a = np.array(Image.open(p)); u = set(np.unique(a).tolist()); vals |= u; sizes.add(a.shape)
        if 1 not in u and 2 not in u: noood += 1
    print(f'{name:20s} n={len(files):3d} values={sorted(vals)} no-OoD={noood} shapes={sorted(sizes)[:3]}')
