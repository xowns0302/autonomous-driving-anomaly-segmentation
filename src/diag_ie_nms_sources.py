"""Trace where DetSeg's final OoD boxes come from (no DetSeg code modified).

For every final OoD box (label 0, score > 0.2 after IE-NMS) this records:
  - source branch: ID branch ('objects' entity, label 0) or UN (universal) branch
  - whether it overlaps a GT OoD component
  - the best-overlapping pre-NMS ID-class box (label 1..19): class, score,
    whether it passed the IE-NMS bar, and its IoU with the final box

This separates two hypotheses for "ID objects (cars, trucks) survive as OoD":
  (a) UN box on a car survives because the ID car box did not pass the bar
  (b) the ID branch itself labels the car as 'objects' (label 0)

Only detection runs (model type switched to DetSeg, SAM is not used).
Run from external/DetSeg-official:
  PYTHONPATH=. python ../../src/diag_ie_nms_sources.py \
      --config configs/detseg/detseg-s_swin-b.py \
      --checkpoint ckpts/detseg_swin-b_coco_20260623-207453f5.pth \
      --out-dir work_dirs/diag_ie_nms/fs_lostfound
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv.ops import batched_nms
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from torch.utils.data import DataLoader

ENTITIES = ('objects', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle')

CAPTURE = {}


def patched_ie_nms(self, bboxes, scores, labels):
    """Identical to GroundingDINOHeadWithUniversalObjectness._ie_nms
    (models.py:518-541), plus capturing inputs and kept indices."""
    thres = labels.new_ones(labels.shape).float()
    id_thresholds = scores.new_tensor([0.2] * 11 + [0.5] * 8)
    id_mask = (labels > 0) & (labels <= len(id_thresholds))
    thres[id_mask] = id_thresholds[labels[id_mask] - 1]

    id_mask = (labels != 0) & (scores > thres)
    nms_scores = scores.clone()
    nms_scores[id_mask] *= 2
    det_bboxes, keep = batched_nms(
        bboxes, nms_scores, labels,
        nms_cfg=dict(iou_threshold=0.5), class_agnostic=True)

    results = InstanceData()
    results.bboxes = det_bboxes[:, :-1]
    results.scores = det_bboxes[:, -1]
    results.labels = labels[keep]
    results.scores[id_mask[keep]] /= 2

    CAPTURE.setdefault('calls', []).append(dict(
        bboxes=bboxes.detach().cpu().numpy(),
        scores=scores.detach().cpu().numpy(),
        labels=labels.detach().cpu().numpy(),
        boosted=id_mask.detach().cpu().numpy(),
        keep=keep.detach().cpu().numpy()))
    return results


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-6)


def gt_boxes(gt):
    """OoD connected components as xyxy boxes and their pixel areas."""
    num, _, stats, _ = cv2.connectedComponentsWithStats((gt == 1).astype(np.uint8), connectivity=8)
    boxes = np.array([[s[0], s[1], s[0] + s[2], s[1] + s[3]] for s in stats[1:]], dtype=np.float32).reshape(-1, 4)
    return boxes, stats[1:, cv2.CC_STAT_AREA].astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = p.parse_args()

    import mmdet.detseg_utils  # noqa: F401
    from mmdet.detseg_utils.models import GroundingDINOHeadWithUniversalObjectness
    from mmdet.registry import DATASETS, MODELS
    GroundingDINOHeadWithUniversalObjectness._ie_nms = patched_ie_nms

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope('mmdet')
    cfg.model.type = 'DetSeg'          # detection only: same DetSeg.predict as DetSeg-S, no SAM
    cfg.model.pop('sam_model_name', None)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=pseudo_collate)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.to(args.device).eval()
    n_id = model.bbox_head.test_cfg.get('max_per_img', 300)   # first n_id rows = ID branch

    records = []
    for idx, batch in enumerate(loader):
        CAPTURE.clear()
        with torch.no_grad():
            model.test_step(batch)
        call = CAPTURE['calls'][0]
        ds = batch['data_samples'][0]
        gt = ds.gt_sem_seg.sem_seg.squeeze(0).numpy().astype(np.int64)
        gtb, gt_areas = gt_boxes(gt)

        B, S, L, keep = call['bboxes'], call['scores'], call['labels'], call['keep']
        is_id_row = np.arange(len(B)) < n_id
        # Pre-NMS candidates + GT boxes, for offline IE-NMS sweeps
        cand_dir = Path(args.out_dir) / 'candidates'
        cand_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cand_dir / f'{idx:04d}.npz', bboxes=B, scores=S, labels=L,
                            is_id_row=is_id_row, gt_boxes=gtb, gt_areas=gt_areas,
                            img_path=np.array(ds.metainfo['img_path']))
        # ID-class candidates before NMS (label 1..19 from the ID branch)
        id_cls = np.where(is_id_row & (L > 0))[0]
        final = [k for k in keep if L[k] == 0 and S[k] > 0.2]   # = DetSeg.predict final filter

        fb = B[final] if final else np.zeros((0, 4))
        iou_gt = iou_matrix(fb, gtb)
        iou_id = iou_matrix(fb, B[id_cls])
        for j, k in enumerate(final):
            rec = dict(img=idx, source='ID-objects' if is_id_row[k] else 'UN',
                       score=float(S[k]),
                       gt_iou=float(iou_gt[j].max()) if gtb.size else 0.0)
            if len(id_cls):
                m = int(iou_id[j].argmax())
                kid = id_cls[m]
                rec.update(best_id_iou=float(iou_id[j, m]), best_id_class=ENTITIES[L[kid]],
                           best_id_score=float(S[kid]), best_id_boosted=bool(call['boosted'][kid]))
            records.append(rec)
        if (idx + 1) % 20 == 0:
            print(f'processed {idx + 1}/{len(dataset)}', flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'final_boxes.json', 'w') as f:
        json.dump(records, f, indent=1)

    def summarize(rs, title):
        n = len(rs)
        print(f'\n== {title}: {n} boxes')
        if not n:
            return
        print('  source:', dict(Counter(r['source'] for r in rs)))
        ov = [r for r in rs if r.get('best_id_iou', 0) >= 0.5]
        print(f'  overlapping an ID-class box (IoU>=0.5): {len(ov)} ({len(ov) / n * 100:.1f}%)')
        if ov:
            print('    that ID box class:', dict(Counter(r['best_id_class'] for r in ov).most_common(8)))
            print('    that ID box passed bar (boosted):', dict(Counter(r['best_id_boosted'] for r in ov)))
            sc = np.array([r['best_id_score'] for r in ov])
            print('    that ID box score pct[10,50,90]:', np.percentile(sc, [10, 50, 90]).round(3))

    fp = [r for r in records if r['gt_iou'] < 0.1]
    tp = [r for r in records if r['gt_iou'] >= 0.1]
    summarize(records, 'ALL final OoD boxes')
    summarize(fp, 'FP-like (GT IoU < 0.1)')
    summarize(tp, 'TP-like (GT IoU >= 0.1)')


if __name__ == '__main__':
    main()
