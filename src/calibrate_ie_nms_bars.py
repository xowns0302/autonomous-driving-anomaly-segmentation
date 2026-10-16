"""IE-NMS bar calibration as described in the DetSeg paper (Sec. 3.1, Fig. 3).

Paper: "The process begins with a statistical calibration step performed on a
subset of the Cityscapes dataset. For each category, we establish one-to-one
matching between predicted boxes and ground truth boxes, removing pairs with
IoU < 0.5 (the threshold used when calculating AP50), and record the minimum
classification score as the category bar."

The public code has no calibration step (bars are hard-coded 0.2 / 0.5).
Our implementation:
  * predictions: the ID-branch boxes (label 1..19, top-300 per image) that
    enter IE-NMS, from the released DetSeg checkpoint
  * GT boxes: Cityscapes gtFine val instances (instanceIds = labelId*1000 + k)
    for the 8 thing classes. Crowd regions without instance ids are ignored.
    Stuff classes have no instances; optionally (--stuff) each connected
    component of a stuff class is used as a GT box (our interpretation)
  * matching: AP50-style, per image and class: predictions sorted by score,
    each greedily matched to the unmatched GT with the highest IoU >= 0.5
  * bar = minimum score over matched predictions (paper). Percentiles are
    also reported because a minimum is sensitive to single hard instances

Step 1 (GPU, detseg env, run from external/DetSeg-official):
  PYTHONPATH=. python ../../src/calibrate_ie_nms_bars.py dump \
      --checkpoint ckpts/detseg_swin-b_coco_20260623-207453f5.pth \
      --data-root data/cityscapes --out-dir ../../results/A3/cityscapes_val_preds
Step 2 (CPU):
  python src/calibrate_ie_nms_bars.py calibrate \
      --pred-dir results/A3/cityscapes_val_preds --gt-root ~/datasets/cityscapes \
      --out results/A3/bars.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'train', 'motorcycle', 'bicycle')          # DetSeg label 1..19
LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23,                # Cityscapes labelIds
             24, 25, 26, 27, 28, 31, 32, 33)
THING = set(range(11, 19))   # indices into CLASSES (person .. bicycle)


def dump(args):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from torch.utils.data import DataLoader
    import mmdet.detseg_utils  # noqa: F401
    from mmdet.detseg_utils.models import GroundingDINOHeadWithUniversalObjectness
    from mmdet.registry import DATASETS, MODELS

    captured = {}
    orig = GroundingDINOHeadWithUniversalObjectness._ie_nms

    def capture(self, bboxes, scores, labels):
        captured['b'], captured['s'], captured['l'] = (x.detach().cpu().numpy() for x in (bboxes, scores, labels))
        return orig(self, bboxes, scores, labels)
    GroundingDINOHeadWithUniversalObjectness._ie_nms = capture

    here = Path(__file__).resolve().parent.parent
    cfg = Config.fromfile(str(here / 'configs/detseg-s_fs_lostfound.py'))
    init_default_scope('mmdet')
    ds_cfg = cfg.test_dataloader.dataset
    ds_cfg.data_root = args.data_root
    ds_cfg.img_suffix = '_leftImg8bit.png'
    ds_cfg.seg_map_suffix = '_gtFine_labelIds.png'
    ds_cfg.data_prefix = dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val')
    dataset = DATASETS.build(ds_cfg)
    n = len(dataset) if not args.max_images else min(args.max_images, len(dataset))
    loader = DataLoader(torch.utils.data.Subset(dataset, range(n)), batch_size=1,
                        shuffle=False, num_workers=2, collate_fn=pseudo_collate)
    cfg.model.type = 'DetSeg'          # detection only, no SAM
    cfg.model.pop('sam_model_name', None)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.to(args.device).eval()
    n_id = model.bbox_head.test_cfg.get('max_per_img', 300)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(loader):
        captured.clear()
        with torch.no_grad():
            model.test_step(batch)
        b, s, l = captured['b'], captured['s'], captured['l']
        idr = (np.arange(len(b)) < n_id) & (l > 0)          # ID-branch, ID-class boxes
        name = Path(batch['data_samples'][0].metainfo['img_path']).name.replace('_leftImg8bit.png', '')
        np.savez_compressed(out / f'{name}.npz', bboxes=b[idr], scores=s[idr], labels=l[idr])
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f'{i + 1}/{n}', flush=True)


def gt_boxes(gt_root, name, stuff, min_area):
    """Return {class_index: [xyxy, ...]} for one Cityscapes val image."""
    import cv2
    from PIL import Image
    city = name.split('_')[0]
    inst = np.array(Image.open(os.path.join(gt_root, 'gtFine/val', city, f'{name}_gtFine_instanceIds.png')))
    boxes = defaultdict(list)
    for v in np.unique(inst):
        if v < 1000:
            continue
        lid = v // 1000
        if lid not in LABEL_IDS:
            continue
        ys, xs = np.nonzero(inst == v)
        if len(xs) < min_area:
            continue
        boxes[LABEL_IDS.index(lid)].append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    if stuff:
        lab = np.array(Image.open(os.path.join(gt_root, 'gtFine/val', city, f'{name}_gtFine_labelIds.png')))
        for ci in range(11):
            m = (lab == LABEL_IDS[ci]).astype(np.uint8)
            if not m.any():
                continue
            k, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for x, y, w, h, a in stats[1:]:
                if a >= min_area:
                    boxes[ci].append([x, y, x + w, y + h])
    return {k: np.array(v, dtype=np.float32) for k, v in boxes.items()}


def iou(a, b):
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-6)


def calibrate(args):
    matched = defaultdict(list)
    n_gt = defaultdict(int)
    files = sorted(glob.glob(os.path.join(args.pred_dir, '*.npz')))
    for f in files:
        name = Path(f).stem
        d = np.load(f)
        gts = gt_boxes(os.path.expanduser(args.gt_root), name, args.stuff, args.min_area)
        for ci, g in gts.items():
            n_gt[ci] += len(g)
            sel = d['labels'] == ci + 1
            pb, ps = d['bboxes'][sel], d['scores'][sel]
            used = np.zeros(len(g), bool)
            for k in np.argsort(-ps):
                ious = iou(pb[k], g)
                ious[used] = -1
                j = int(ious.argmax())
                if ious[j] >= 0.5:
                    used[j] = True
                    matched[ci].append(float(ps[k]))
    released = [0.2] * 11 + [0.5] * 8
    result = dict(images=len(files), stuff_included=args.stuff, min_gt_area=args.min_area, classes={})
    print(f'{len(files)} images | GT min area {args.min_area}px | stuff boxes: {args.stuff}')
    print(f'{"class":14s} {"GT":>5s} {"matched":>7s} {"min(paper)":>10s} {"p1":>6s} {"p5":>6s} {"p10":>6s} {"released":>8s}')
    for ci, c in enumerate(CLASSES):
        sc = np.array(matched[ci])
        row = dict(n_gt=n_gt[ci], n_matched=len(sc), released_bar=released[ci])
        if len(sc):
            row.update(bar_min=float(sc.min()), p1=float(np.percentile(sc, 1)),
                       p5=float(np.percentile(sc, 5)), p10=float(np.percentile(sc, 10)),
                       median=float(np.median(sc)))
            print(f'{c:14s} {n_gt[ci]:5d} {len(sc):7d} {sc.min():10.3f} {row["p1"]:6.3f} {row["p5"]:6.3f} {row["p10"]:6.3f} {released[ci]:8.2f}')
        else:
            print(f'{c:14s} {n_gt[ci]:5d} {0:7d} {"-":>10s} {"-":>6s} {"-":>6s} {"-":>6s} {released[ci]:8.2f}')
        result['classes'][c] = row
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, 'w'), indent=2)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('dump')
    a.add_argument('--checkpoint', required=True)
    a.add_argument('--data-root', required=True)
    a.add_argument('--out-dir', required=True)
    a.add_argument('--max-images', type=int, default=0)
    a.add_argument('--device', default='cuda:0')
    b = sub.add_parser('calibrate')
    b.add_argument('--pred-dir', required=True)
    b.add_argument('--gt-root', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--stuff', action='store_true')
    b.add_argument('--min-area', type=int, default=0)
    args = p.parse_args()
    dump(args) if args.cmd == 'dump' else calibrate(args)


if __name__ == '__main__':
    main()
