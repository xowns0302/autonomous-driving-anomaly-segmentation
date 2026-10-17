"""Per-stage latency of DetSeg-S / DetSeg-R (official code, unchanged).

Wraps each stage with torch.cuda.synchronize() + perf_counter and reports the
mean time per image (warm-up images excluded). Stages:

  data        dataloader: read image, resize, (DetSeg-R) load score map
  preprocess  normalise / pad / move to GPU (data_preprocessor)
  backbone    Swin-B                    neck      ChannelMapper
  text        BERT + text projection    encoder   image-text fusion (6 layers)
  query_sel   ID / UN query selection   decoder   6 decoder layers
  head        box / class / objectness heads + IE-NMS (ie_nms reported separately)
  sam_*       DetSeg-S: SAM preprocessing, image encoder, mask decoder, postprocess
  boost       DetSeg-R: score-map interpolation + box boost (inside predict)

Run from external/DetSeg-official (detseg env), nothing else using the GPU:
  PYTHONPATH=. python ../../src/profile_latency.py --config ../../configs/detseg-s_fs_lostfound.py \
      --checkpoint ckpts/detseg_swin-b_coco_20260623-207453f5.pth --images 30 --out ../../results/T1/detseg_s.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader, Subset

STAGES = ['data', 'preprocess', 'backbone', 'neck', 'text', 'encoder', 'query_sel', 'decoder',
          'head', 'ie_nms', 'sam_prep', 'sam_image_encoder', 'sam_mask_decoder', 'sam_post', 'boost']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    import mmdet.detseg_utils  # noqa: F401
    from mmdet.registry import DATASETS, MODELS

    cfg = Config.fromfile(args.config)
    init_default_scope('mmdet')
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    n = min(len(dataset), args.images + args.warmup)
    loader = DataLoader(Subset(dataset, range(n)), batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=pseudo_collate)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.cuda().eval()

    cur = defaultdict(float)

    def timed(name, fn):
        def w(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize()
            cur[name] += (time.perf_counter() - t) * 1000
            return r
        return w

    def wrap(obj, attr, name):
        setattr(obj, attr, timed(name, getattr(obj, attr)))

    wrap(model.data_preprocessor, 'forward', 'preprocess')
    wrap(model.backbone, 'forward', 'backbone')
    wrap(model.neck, 'forward', 'neck')
    wrap(model.language_model, 'forward', 'text')
    wrap(model.text_feat_map, 'forward', 'text')
    wrap(model, 'forward_encoder', 'encoder')
    wrap(model, 'pre_decoder', 'query_sel')
    wrap(model, 'forward_decoder', 'decoder')
    wrap(model.bbox_head, 'predict', 'head')
    wrap(model.bbox_head, '_ie_nms', 'ie_nms')
    is_r = hasattr(model, '_boost_anomaly_score')
    if is_r:
        wrap(model, '_boost_anomaly_score', 'boost')

    has_sam = hasattr(model, 'sam_predict_hf') and (not is_r or getattr(model, 'use_sam_refine', False))
    if has_sam:
        # Re-implementation of SamPromptMixin.sam_predict_hf (models.py:299) with timers.
        def sam_predict_hf(self, raw_image, boxes):
            from PIL import Image
            device = boxes.device
            sam_model = self._sam_on_device(device)
            processor = self.sam_processor
            torch.cuda.synchronize(); t = time.perf_counter()
            img = Image.open(raw_image).convert('RGB')
            inputs = processor(img, return_tensors='pt').to(device)
            torch.cuda.synchronize(); cur['sam_prep'] += (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            emb = sam_model.get_image_embeddings(inputs['pixel_values'])
            torch.cuda.synchronize(); cur['sam_image_encoder'] += (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            inputs = processor(img, input_boxes=[boxes.cpu().numpy().tolist()], return_tensors='pt').to(device)
            inputs.pop('pixel_values', None)
            inputs.update({'image_embeddings': emb})
            torch.cuda.synchronize(); cur['sam_prep'] += (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            with torch.no_grad():
                out = sam_model(**inputs)
            torch.cuda.synchronize(); cur['sam_mask_decoder'] += (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            masks = processor.image_processor.post_process_masks(
                out.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())
            cur['sam_post'] += (time.perf_counter() - t) * 1000
            return masks
        model.sam_predict_hf = sam_predict_hf.__get__(model)

    rows = []
    torch.cuda.reset_peak_memory_stats()
    it = iter(loader)
    for i in range(n):
        cur.clear()
        t0 = time.perf_counter()
        batch = next(it)
        cur['data'] += (time.perf_counter() - t0) * 1000
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            model.test_step(batch)
        torch.cuda.synchronize()
        model_ms = (time.perf_counter() - t1) * 1000
        if i >= args.warmup:
            rows.append(dict(cur, model_total=model_ms, total=model_ms + cur['data']))
    keys = [k for k in STAGES if any(k in r for r in rows)] + ['model_total', 'total']
    mean = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}
    nested = {'head': mean.get('head', 0) - mean.get('ie_nms', 0)}   # head time without IE-NMS
    accounted = sum(mean.get(k, 0) for k in STAGES if k not in ('data', 'head', 'ie_nms')) + \
        nested['head'] + mean.get('ie_nms', 0)
    mean['other_in_model'] = mean['model_total'] - accounted
    mean['head_excl_ie_nms'] = nested['head']

    print(f'{args.config} | {len(rows)} images (after {args.warmup} warm-up) | '
          f'peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    order = ['data', 'preprocess', 'backbone', 'neck', 'text', 'encoder', 'query_sel', 'decoder',
             'head_excl_ie_nms', 'ie_nms', 'sam_prep', 'sam_image_encoder', 'sam_mask_decoder', 'sam_post',
             'boost', 'other_in_model']
    for k in order:
        if k in mean and (k not in STAGES or k in keys or k in ('head_excl_ie_nms', 'other_in_model')):
            print(f'  {k:18s} {mean[k]:8.1f} ms  {mean[k] / mean["total"] * 100:5.1f}%')
    print(f'  {"model_total":18s} {mean["model_total"]:8.1f} ms\n  {"total":18s} {mean["total"]:8.1f} ms')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(config=args.config, images=len(rows), warmup=args.warmup,
                   peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, mean_ms=mean,
                   gpu=torch.cuda.get_device_name(0)), open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
