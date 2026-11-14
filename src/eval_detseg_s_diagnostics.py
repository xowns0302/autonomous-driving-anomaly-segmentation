"""DetSeg-S baseline diagnostics.

Runs DetSeg-S once over a config's test dataset and saves, per image:
  - IoU / F1 computed exactly like BinaryMaskAnomalyIoUMetric (per-image mean)
  - intersection / union pixels (for dataset-level "global" IoU)
  - predicted OoD boxes (bbox, score, area) and GT component box recall
  - the predicted binary mask (PNG) for later visual inspection

No DetSeg code is modified. Run from the DetSeg root so relative data/ckpt
paths in the config resolve, e.g.

  cd external/DetSeg-official
  PYTHONPATH=. python ../../src/eval_detseg_s_diagnostics.py \
      --config configs/detseg/detseg-s_swin-b.py \
      --checkpoint ckpts/detseg_swin-b_coco_20260623-207453f5.pth \
      --out-dir work_dirs/diag_detseg_s/fs_lostfound
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='DetSeg-S per-image diagnostics')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--box-iou-thr', type=float, default=0.5,
                        help='IoU threshold for counting a GT component as recalled')
    parser.add_argument('--no-save-masks', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    # IE-NMS bar override (default = original hard-coded 0.2 stuff / 0.5 thing)
    parser.add_argument('--stuff-bar', type=float, default=None,
                        help='bar for labels 1-11 (road..sky); original 0.2')
    parser.add_argument('--thing-bar', type=float, default=None,
                        help='bar for labels 12-19 (person..bicycle); original 0.5')
    parser.add_argument('--never-boost', nargs='*', default=[],
                        help='class names whose ID boxes are never boosted, e.g. pole "traffic sign"')
    # Size-aware final threshold: OoD boxes with area < small-area (original-image px)
    # are kept if score > small-thr instead of the hard-coded 0.2 (models.py:961)
    parser.add_argument('--small-area', type=float, default=0)
    parser.add_argument('--small-thr', type=float, default=0.2)
    # Road-aware re-ranking using the detector's own 'road' predictions (label 1,
    # score > 0.2, pre-NMS). A box is on-road if its bottom-center point lies inside
    # any road box. Images without a road box are left unchanged.
    parser.add_argument('--road-aware', choices=['none', 'soft', 'hard', 'hard_small'], default='none')
    parser.add_argument('--road-alpha', type=float, default=0.5, help='score multiplier for off-road boxes (soft)')
    return parser.parse_args()


def patch_ie_nms(stuff_bar, thing_bar, never_boost, small_area=0, small_thr=0.2,
                 road_aware='none', road_alpha=0.5):
    """Replace _ie_nms with the same logic but configurable bars."""
    from mmengine.structures import InstanceData
    from mmdet.detseg_utils.models import GroundingDINOHeadWithUniversalObjectness
    from sweep_ie_nms_bars import NEVER, bars, ie_nms

    bar_list = bars(stuff_bar, thing_bar, {c: NEVER for c in never_boost})

    def _ie_nms(self, bboxes, scores, labels):
        keep = torch.as_tensor(ie_nms(bboxes.cpu(), scores.cpu(), labels.cpu(), bar_list),
                               device=bboxes.device).long()
        results = InstanceData()
        results.bboxes = bboxes[keep]
        results.scores = scores[keep]
        results.labels = labels[keep]
        b = results.bboxes
        if road_aware != 'none':
            road = bboxes[(labels == 1) & (scores > 0.2)]
            if len(road):
                fx, fy = (b[:, 0] + b[:, 2]) / 2, b[:, 3] - 1
                on = ((road[None, :, 0] <= fx[:, None]) & (fx[:, None] <= road[None, :, 2]) &
                      (road[None, :, 1] <= fy[:, None]) & (fy[:, None] <= road[None, :, 3])).any(1)
                off = (results.labels == 0) & ~on
                if road_aware == 'hard_small':
                    off &= (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) < max(small_area, 1024)
                factor = road_alpha if road_aware == 'soft' else 0.0
                results.scores = results.scores.clone()
                results.scores[off] *= factor
        if small_area > 0:
            # DetSeg.predict keeps label-0 boxes with score > 0.2; rescaling small boxes'
            # scores by 0.2 / small_thr makes that filter equal to score > small_thr.
            # DetSeg-S only uses the boxes downstream, so the rescale has no other effect.
            small = (results.labels == 0) & ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) < small_area)
            results.scores = results.scores.clone()
            results.scores[small] *= 0.2 / small_thr
        return results

    GroundingDINOHeadWithUniversalObjectness._ie_nms = _ie_nms
    return bar_list


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def gt_components(gt):
    """Connected components of OoD pixels (gt == 1) as xyxy boxes + areas."""
    mask = (gt == 1).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        comps.append(dict(box=[int(x), int(y), int(x + w), int(y + h)], area=int(area)))
    return comps


def component_metrics(gt, pred, min_gt_area=10):
    """SMIYC-style component metrics for one image (approximate re-implementation
    of Chan et al., SegmentMeIfYouCan, NeurIPS 2021 D&B).

    sIoU(k) = |k ∩ P(k)| / |(k ∪ P(k)) \ A(k)|, P(k) = predicted components touching k,
              A(k) = other GT OoD pixels.
    PPV(p)  = |p ∩ K(p)| / |p|, K(p) = GT components touching p.
    Void pixels (255) are excluded from predicted components. GT components smaller
    than min_gt_area px are ignored (Road Anomaly has 1px label noise).
    Returns (list of sIoU per GT component, list of PPV per predicted component).
    """
    valid = gt != 255
    gt_ood = gt == 1
    n_g, g_lab = cv2.connectedComponents(gt_ood.astype(np.uint8), connectivity=8)
    n_p, p_lab = cv2.connectedComponents((pred & valid).astype(np.uint8), connectivity=8)

    sious = []
    for k in range(1, n_g):
        km = g_lab == k
        if km.sum() < min_gt_area:
            continue
        touching = np.unique(p_lab[km])
        touching = touching[touching > 0]
        pk = np.isin(p_lab, touching) if len(touching) else np.zeros_like(km)
        other_gt = gt_ood & ~km
        inter = np.logical_and(km, pk).sum()
        union = np.logical_and(np.logical_or(km, pk), ~other_gt).sum()
        sious.append(float(inter / union) if union else 0.0)

    ppvs = []
    for q in range(1, n_p):
        qm = p_lab == q
        touching = np.unique(g_lab[qm])
        touching = touching[touching > 0]
        kq = np.isin(g_lab, touching) if len(touching) else np.zeros_like(qm)
        ppvs.append(float(np.logical_and(qm, kq).sum() / qm.sum()))
    return sious, ppvs


def summarize_components(all_sious, all_ppvs):
    s = np.array(all_sious); p = np.array(all_ppvs)
    f1s = []
    for t in np.arange(0.25, 0.751, 0.05):
        tp = int((s > t).sum()); fn = len(s) - tp; fp = int((p <= t).sum())
        f1s.append(2 * tp / max(2 * tp + fn + fp, 1))
    return dict(
        sIoU_gt=round(float(s.mean()) * 100, 2) if len(s) else None,
        PPV=round(float(p.mean()) * 100, 2) if len(p) else None,
        mean_F1=round(float(np.mean(f1s)) * 100, 2),
        n_gt_components=len(s), n_pred_components=len(p),
        F1_at_0_5=round(float(f1s[5]) * 100, 2))


def main():
    args = parse_args()
    import mmdet.detseg_utils  # noqa: F401  (register DetSeg modules)
    from mmdet.registry import DATASETS, MODELS

    bar_list = None
    if (args.stuff_bar is not None or args.thing_bar is not None or args.never_boost
            or args.small_area > 0 or args.road_aware != 'none'):
        bar_list = patch_ie_nms(0.2 if args.stuff_bar is None else args.stuff_bar,
                                0.5 if args.thing_bar is None else args.thing_bar,
                                args.never_boost, args.small_area, args.small_thr,
                                args.road_aware, args.road_alpha)
        print('IE-NMS bars overridden:', bar_list, flush=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope('mmdet')

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=pseudo_collate)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.to(args.device).eval()

    out_dir = Path(args.out_dir)
    mask_dir = out_dir / 'pred_masks'
    if not args.no_save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, batch in enumerate(loader):
        with torch.no_grad():
            outputs = model.test_step(batch)
        for ds in outputs:
            gt = ds.gt_sem_seg.sem_seg.squeeze(0).cpu().numpy().astype(np.int64)
            pred = ds.pred_sem_seg.sem_seg.squeeze(0).cpu().numpy() > 0.5

            # Same definition as BinaryMaskAnomalyIoUMetric.calculate_miou
            inter = int(np.logical_and(gt == 1, pred).sum())
            union = int(np.logical_and(gt != 255, np.logical_or(gt == 1, pred)).sum())
            iou = inter / union if union else 0.0
            f1 = 2 * inter / (inter + union) if union else 0.0

            sious, ppvs = component_metrics(gt, pred)

            inst = ds.pred_instances
            boxes = inst.bboxes.detach().cpu().numpy()
            scores = inst.scores.detach().cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else np.zeros(0)

            comps = gt_components(gt)
            for c in comps:
                ious = box_iou(np.array(c['box'], dtype=np.float32), boxes)
                c['best_box_iou'] = float(ious.max()) if len(ious) else 0.0
                c['recalled'] = bool(c['best_box_iou'] >= args.box_iou_thr)

            img_path = ds.metainfo['img_path']
            name = Path(img_path).stem
            if name.endswith('.labels'):
                name = name[:-len('.labels')]
            if not args.no_save_masks:
                cv2.imwrite(str(mask_dir / f'{idx:04d}_{name}.png'), pred.astype(np.uint8) * 255)

            rows.append(dict(
                idx=idx, img_path=img_path,
                iou=iou, f1=f1, inter=inter, union=union,
                gt_pixels=int((gt == 1).sum()), pred_pixels=int(pred.sum()),
                has_ood=bool((gt == 1).any()),
                comp_sious=[round(v, 4) for v in sious], comp_ppvs=[round(v, 4) for v in ppvs],
                n_boxes=int(len(boxes)),
                boxes=boxes.round(1).tolist(), scores=scores.round(4).tolist(),
                box_areas=areas.round(0).tolist(),
                gt_components=comps))
        if (idx + 1) % 10 == 0:
            print(f'processed {idx + 1}/{len(dataset)}', flush=True)
        if args.max_samples and idx + 1 >= args.max_samples:
            break

    ious = np.array([r['iou'] for r in rows])
    has_ood = np.array([r['has_ood'] for r in rows])
    comps = [c for r in rows for c in r['gt_components']]
    comp_areas = np.array([c['area'] for c in comps]) if comps else np.zeros(0)
    recalled = np.array([c['recalled'] for c in comps]) if comps else np.zeros(0, bool)
    n_boxes = np.array([r['n_boxes'] for r in rows])

    area_bins = [(0, 32 ** 2), (32 ** 2, 96 ** 2), (96 ** 2, float('inf'))]
    binned = {}
    for lo, hi in area_bins:
        sel = (comp_areas >= lo) & (comp_areas < hi)
        key = f'[{int(lo)},{"inf" if hi == float("inf") else int(hi)})'
        binned[key] = dict(n=int(sel.sum()),
                           recall=round(float(recalled[sel].mean()) * 100, 2) if sel.any() else None)

    summary = dict(
        config=args.config, checkpoint=args.checkpoint, cfg_options=args.cfg_options,
        ie_nms_bars=bar_list, small_area=args.small_area, small_thr=args.small_thr,
        road_aware=args.road_aware, road_alpha=args.road_alpha,
        images=len(rows),
        images_with_ood=int(has_ood.sum()),
        iou_mean_per_image=round(float(ious.mean()) * 100, 2),            # official metric
        f1_mean_per_image=round(float(np.mean([r['f1'] for r in rows])) * 100, 2),
        iou_mean_ood_images_only=round(float(ious[has_ood].mean()) * 100, 2) if has_ood.any() else None,
        iou_global=round(sum(r['inter'] for r in rows) / max(sum(r['union'] for r in rows), 1) * 100, 2),
        iou_percentiles={p: round(float(np.percentile(ious, p)) * 100, 2) for p in (10, 25, 50, 75, 90)},
        images_iou_zero=int((ious == 0).sum()),
        images_no_pred_box=int((n_boxes == 0).sum()),
        boxes_per_image_mean=round(float(n_boxes.mean()), 2),
        gt_components=len(comps),
        box_recall=round(float(recalled.mean()) * 100, 2) if comps else None,
        box_recall_iou_thr=args.box_iou_thr,
        box_recall_by_area=binned,
        component=summarize_components([v for r in rows for v in r['comp_sious']],
                                       [v for r in rows for v in r['comp_ppvs']]),
    )

    with open(out_dir / 'per_image.json', 'w') as f:
        json.dump(rows, f, indent=1)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
