"""Table 6 first row: Grounding DINO prompted only with 'objects' + SAM.

Paper 4.3: "we set the Grounding DINO with prompt 'objects' as the baseline".
Uses the MM-Grounding-DINO Swin-B weights that DetSeg starts from
(no UN queries, no IE-NMS). Boxes with score > tau (0.2, as in DetSeg) are
passed to SAM exactly like DetSegS, and IoU is computed like
BinaryMaskAnomalyIoUMetric (per-image IoU, averaged).

Run from external/DetSeg-official in the `detseg` env:
  PYTHONPATH=. python ../../src/gdino_objects_baseline.py \
      --config ../../configs/detseg-s_fs_lostfound.py \
      --checkpoint ckpts/grounding_dino_swin-b_pretrain_all-f9818a7c.pth \
      --out-dir work_dirs/E4/fs_lostfound_gdino_objects
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mmcv.transforms.base import BaseTransform
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='a DetSeg-S dataset config (dataset/pipeline are reused)')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--tau', type=float, default=0.2)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()
    import mmdet.detseg_utils  # noqa: F401  (datasets, transforms)
    from mmdet.detseg_utils.models import SamPromptMixin
    from mmdet.registry import DATASETS, MODELS, TRANSFORMS

    @TRANSFORMS.register_module(force=True)
    class ObjectsOnlyPrompt(BaseTransform):
        def transform(self, results):
            results['text'] = ('objects',)
            results['custom_entities'] = True
            return results

    cfg = Config.fromfile(args.config)
    init_default_scope('mmdet')
    ds_cfg = cfg.test_dataloader.dataset
    ds_cfg.pipeline = [dict(type='ObjectsOnlyPrompt') if t['type'] == 'ConcatPrompt' else t
                       for t in ds_cfg.pipeline]
    dataset = DATASETS.build(ds_cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=pseudo_collate)

    # Same architecture as DetSeg's detector, as plain Grounding DINO.
    m = cfg.model.copy()
    m['type'] = 'GroundingDINO'
    m.pop('sam_model_name', None)
    m['bbox_head'] = dict(m['bbox_head'], type='GroundingDINOHead')
    model = MODELS.build(m)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.to(args.device).eval()

    class Sam(SamPromptMixin):
        pass
    sam = Sam()
    sam._init_sam('facebook/sam-vit-base')

    rows = []
    for idx, batch in enumerate(loader):
        with torch.no_grad():
            ds = model.test_step(batch)[0]
        inst = ds.pred_instances
        inst = inst[inst.scores > args.tau]
        shape = ds.metainfo['ori_shape']
        masks = sam.sam_masks_from_boxes(ds.metainfo['img_path'], inst.bboxes, shape, inst.bboxes.device)
        pred = masks.bool().any(dim=0).cpu().numpy()
        gt = ds.gt_sem_seg.sem_seg.squeeze(0).cpu().numpy().astype(np.int64)
        inter = int(np.logical_and(gt == 1, pred).sum())
        union = int(np.logical_and(gt != 255, np.logical_or(gt == 1, pred)).sum())
        rows.append(dict(img_path=ds.metainfo['img_path'], n_boxes=int(len(inst)),
                         iou=inter / union if union else 0.0,
                         f1=2 * inter / (inter + union) if union else 0.0))
        if (idx + 1) % 20 == 0:
            print(f'processed {idx + 1}/{len(dataset)}', flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(config=args.config, checkpoint=args.checkpoint, tau=args.tau, images=len(rows),
                   iou_mean_per_image=round(float(np.mean([r['iou'] for r in rows])) * 100, 2),
                   f1_mean_per_image=round(float(np.mean([r['f1'] for r in rows])) * 100, 2),
                   boxes_per_image_mean=round(float(np.mean([r['n_boxes'] for r in rows])), 2))
    json.dump(rows, open(out / 'per_image.json', 'w'), indent=1)
    json.dump(summary, open(out / 'summary.json', 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
