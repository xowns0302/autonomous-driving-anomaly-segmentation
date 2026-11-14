"""Save Mask2Anomaly (M2A) anomaly score maps for DetSeg-R.

The score computation is copied from the official
external/Mask2Anomaly/anomaly_utils/anomaly_inference.py (lines 158-186):
  score = 1 - max softmax over the 19 Cityscapes classes, with the official
  anomaly-mask correction, averaged over the original and the left-right
  flipped image.
Each map is saved as `{out_dir}/{basename(image)}.npy` (float32, H x W), which
is the name DetSeg's GetAnomalyScoreMap transform reads.

Run in the `m2a` env:
  python src/m2a_save_scores.py \
      --m2a-repo external/Mask2Anomaly \
      --weights ~/datasets/m2a_validation_dl/Mask2Anomaly/best_contrastive.pth \
      --images 'external/DetSeg-official/data/FS_LostFound/images/*.png' \
      --out-dir results/score_maps/m2a/FS_LostFound
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--m2a-repo', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--images', required=True, help='glob of input images')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--config-file', default=None,
                   help='default: <m2a-repo>/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml')
    return p.parse_args()


def official_score(predictions):
    """Lines 160-169 of the official anomaly_inference.py, unchanged in logic."""
    pred = predictions['sem_seg'].unsqueeze(0)
    out = 1 - torch.max(pred[0:19, :, :], axis=1)[0]
    if predictions['sem_seg'][19:, :, :].shape[0] > 1:
        mask = torch.max(predictions['sem_seg'][19:, :, :].unsqueeze(0), axis=1)[0]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = 1 - mask
        out = out * mask.detach()
    return out.detach().cpu().numpy().squeeze().squeeze()


def main():
    args = parse_args()
    repo = os.path.abspath(args.m2a_repo)
    sys.path.insert(0, repo)
    from detectron2.config import get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    from demo.predictor import VisualizationDemo

    cfg_file = args.config_file or os.path.join(
        repo, 'configs/cityscapes/semantic-segmentation/anomaly_inference.yaml')
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(cfg_file))   # _BASE_ in the yaml is relative
    cfg.merge_from_file(cfg_file)
    os.chdir(cwd)
    cfg.merge_from_list(['MODEL.WEIGHTS', os.path.expanduser(args.weights)])
    cfg.freeze()
    demo = VisualizationDemo(cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.expanduser(args.images)))
    assert paths, f'no images match {args.images}'
    t0 = time.time()
    for i, path in enumerate(paths):
        dst = os.path.join(args.out_dir, os.path.basename(path) + '.npy')
        if os.path.exists(dst):            # resume: keep maps that load completely
            try:
                np.load(dst, mmap_mode='r').sum()
                continue
            except Exception:
                os.remove(dst)
        img = read_image(path, format='BGR')
        pred_na, _ = demo.run_on_image(img)
        pred_lr, _ = demo.run_on_image(np.fliplr(img))
        out_na = official_score(pred_na)
        out_lr = np.flip(official_score(pred_lr).squeeze(), 1)
        score = ((out_lr + out_na) / 2.0).astype(np.float32)
        np.save(dst + '.tmp.npy', score)
        os.replace(dst + '.tmp.npy', dst)      # never leave a half-written map
        if (i + 1) % 10 == 0 or i + 1 == len(paths):
            print(f'{i + 1}/{len(paths)} done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
