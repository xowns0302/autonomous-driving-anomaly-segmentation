"""Per-stage latency of Mask2Anomaly score-map generation (m2a env).

Same computation as src/m2a_save_scores.py (official anomaly_inference.py):
two forward passes (original + left-right flip). The official demo predictor
also draws a visualisation for every call (VisualizationDemo.run_on_image),
which the score map does not need; it is timed separately.

  python src/profile_m2a_latency.py --m2a-repo external/Mask2Anomaly \
      --weights ~/datasets/m2a_validation_dl/Mask2Anomaly/best_contrastive.pth \
      --images 'external/DetSeg-official/data/FS_LostFound/images/*.png' --out results/T1/m2a.json
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m2a-repo', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    repo = os.path.abspath(args.m2a_repo)
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from detectron2.config import get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    from demo.predictor import VisualizationDemo
    from m2a_save_scores import official_score

    cfg_file = os.path.join(repo, 'configs/cityscapes/semantic-segmentation/anomaly_inference.yaml')
    cfg = get_cfg(); add_deeplab_config(cfg); add_maskformer2_config(cfg)
    cwd = os.getcwd(); os.chdir(os.path.dirname(cfg_file)); cfg.merge_from_file(cfg_file); os.chdir(cwd)
    cfg.merge_from_list(['MODEL.WEIGHTS', os.path.expanduser(args.weights)]); cfg.freeze()
    demo = VisualizationDemo(cfg)

    def sync():
        torch.cuda.synchronize()

    paths = sorted(glob.glob(os.path.expanduser(args.images)))[:args.n + args.warmup]
    rows = []
    for i, path in enumerate(paths):
        t = {}
        s = time.perf_counter(); img = read_image(path, format='BGR'); t['read'] = time.perf_counter() - s
        sync(); s = time.perf_counter()
        pa = demo.predictor(img); pb = demo.predictor(np.fliplr(img).copy())
        sync(); t['model_2x'] = time.perf_counter() - s
        s = time.perf_counter()
        score = (official_score(pa) + np.flip(official_score(pb).squeeze(), 1)) / 2.0
        t['score'] = time.perf_counter() - s
        # what the official demo additionally does per call (visualisation), x2
        s = time.perf_counter(); demo.run_on_image(img); demo.run_on_image(np.fliplr(img).copy()); sync()
        t['official_run_on_image_2x'] = time.perf_counter() - s
        if i >= args.warmup:
            rows.append({k: v * 1000 for k, v in t.items()})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    mean['visualisation_overhead_2x'] = mean['official_run_on_image_2x'] - mean['model_2x']
    mean['needed_total'] = mean['read'] + mean['model_2x'] + mean['score']
    print(f'Mask2Anomaly | {len(rows)} images | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    for k, v in mean.items():
        print(f'  {k:28s} {v:8.1f} ms')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(images=len(rows), mean_ms=mean, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                   gpu=torch.cuda.get_device_name(0)), open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
