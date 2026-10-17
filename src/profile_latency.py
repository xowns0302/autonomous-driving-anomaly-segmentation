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

STAGES = ['data', 'c5_box_means', 'c7_prowl', 'preprocess', 'backbone', 'neck', 'text', 'encoder', 'query_sel', 'decoder',
          'head', 'ie_nms', 'sam_prep', 'sam_image_encoder', 'sam_mask_decoder', 'sam_post', 'boost']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--out', required=True)
    p.add_argument('--repeats', type=int, default=3, help='independent timing runs over the same images')
    p.add_argument('--bars', default=None, help='calibrated bar json (A3); default: released bars')
    p.add_argument('--m2a-dir', default=None, help='C5: Mask2Anomaly score-map folder')
    p.add_argument('--verify-t', type=float, default=None, help='C5 threshold')
    p.add_argument('--prowl-bank', default=None, help='C7: prototype bank npz')
    p.add_argument('--reject-t7', type=float, default=None, help='C7 threshold')
    args = p.parse_args()

    import mmdet.detseg_utils  # noqa: F401
    from mmdet.registry import DATASETS, MODELS

    cfg = Config.fromfile(args.config)
    init_default_scope('mmdet')
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    n = args.images + args.warmup
    idx = [i % len(dataset) for i in range(n)]       # small datasets are cycled
    loader = DataLoader(Subset(dataset, idx), batch_size=1, shuffle=False, num_workers=0,
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
    if args.bars or args.verify_t is not None or args.reject_t7 is not None:
        # Same replacement as src/run_with_bars.py (bars and/or C5), with C5's box-mean cost timed.
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from mmengine.structures import InstanceData
        from run_with_bars import load_bars
        from sweep_ie_nms_bars import ie_nms, ie_nms_rules
        bar_list = load_bars(args.bars or 'released', 'bar_min')
        state = {}
        orig_predict = model.bbox_head.predict      # already wrapped by the 'head' timer

        def predict(*a, batch_data_samples=None, **kw):
            state['img_path'] = batch_data_samples[0].metainfo['img_path']
            return orig_predict(*a, batch_data_samples=batch_data_samples, **kw)
        model.bbox_head.predict = predict

        def box_means(img_path, B):
            sm = np.load(os.path.join(args.m2a_dir, os.path.basename(img_path) + '.npy')).squeeze()
            ii = np.pad(sm.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            H, W = sm.shape
            x1 = np.clip(np.floor(B[:, 0]), 0, W).astype(int); x2 = np.clip(np.ceil(B[:, 2]), 0, W).astype(int)
            y1 = np.clip(np.floor(B[:, 1]), 0, H).astype(int); y2 = np.clip(np.ceil(B[:, 3]), 0, H).astype(int)
            tot = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]
            return np.where((x2 > x1) & (y2 > y1), tot / np.maximum((x2 - x1) * (y2 - y1), 1), 0.0)

        def _ie_nms(self, bboxes, scores, labels):
            if args.verify_t is not None:
                t = time.perf_counter()
                vs = box_means(state['img_path'], bboxes.cpu().numpy())
                cur['c5_box_means'] += (time.perf_counter() - t) * 1000
                k = ie_nms_rules(bboxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy(), bar_list,
                                 verify_scores=vs, verify_t=args.verify_t)
            else:
                k = ie_nms(bboxes.cpu(), scores.cpu(), labels.cpu(), bar_list)
            keep = torch.as_tensor(k, device=bboxes.device).long()
            r = InstanceData()
            r.bboxes, r.scores, r.labels = bboxes[keep], scores[keep], labels[keep]
            if args.reject_t7 is not None:
                torch.cuda.synchronize(); t = time.perf_counter()
                ood = (r.labels == 0).cpu().numpy()
                if ood.any():
                    pb = proto_box(state['img_path'], r.bboxes.cpu().numpy())
                    drop = torch.as_tensor(ood & (pb < args.reject_t7), device=r.scores.device)
                    r.scores = r.scores.clone()
                    r.scores[drop] = 0.0
                torch.cuda.synchronize(); cur['c7_prowl'] += (time.perf_counter() - t) * 1000
            return r

        if args.reject_t7 is not None:
            from PIL import Image
            from prowl_bank import load_model, patch_features
            dino = load_model()
            P = torch.from_numpy(np.load(args.prowl_bank)['prototypes']).cuda()

            def proto_box(img_path, B):
                img = np.array(Image.open(img_path).convert('RGB'))
                feat, _ = patch_features(dino, img)
                H, W = img.shape[:2]
                h, w, D = feat.shape
                Pf = P.reshape(-1, D)
                out = np.zeros(len(B), np.float32)
                for j, (x1, y1, x2, y2) in enumerate(B):
                    x1, x2 = np.clip([x1, x2], 0, W); y1, y2 = np.clip([y1, y2], 0, H)
                    gx1, gx2 = min(int(x1 / W * w), w - 1), int(np.ceil(x2 / W * w))
                    gy1, gy2 = min(int(y1 / H * h), h - 1), int(np.ceil(y2 / H * h))
                    gx2, gy2 = max(gx2, gx1 + 1), max(gy2, gy1 + 1)
                    v = torch.nn.functional.normalize(feat[gy1:gy2, gx1:gx2].reshape(-1, D).mean(0), dim=0)
                    out[j] = 1 - float((Pf @ v).max())
                return out
        model.bbox_head._ie_nms = _ie_nms.__get__(model.bbox_head)

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

    runs = []
    torch.cuda.reset_peak_memory_stats()
    for rep in range(args.repeats):
        rows = []
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
        runs.append(rows)
    rows = [r for run in runs for r in run]
    run_means = {k: [float(np.mean([r.get(k, 0.0) for r in run])) for run in runs]
                 for k in ('model_total', 'total', 'data')}
    keys = [k for k in STAGES if any(k in r for r in rows)] + ['model_total', 'total']
    mean = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}
    nested = {'head': mean.get('head', 0) - mean.get('ie_nms', 0)}   # head time without IE-NMS
    accounted = sum(mean.get(k, 0) for k in STAGES if k not in ('data', 'head', 'ie_nms', 'c5_box_means', 'c7_prowl')) + \
        nested['head'] + mean.get('ie_nms', 0)
    mean['other_in_model'] = mean['model_total'] - accounted
    mean['head_excl_ie_nms'] = nested['head']

    print(f'{args.config} | {args.repeats} runs x {args.images} images (after {args.warmup} warm-up) | '
          f'peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    order = ['data', 'c5_box_means', 'c7_prowl', 'preprocess', 'backbone', 'neck', 'text', 'encoder', 'query_sel', 'decoder',
             'head_excl_ie_nms', 'ie_nms', 'sam_prep', 'sam_image_encoder', 'sam_mask_decoder', 'sam_post',
             'boost', 'other_in_model']
    for k in order:
        if k in mean and (k not in STAGES or k in keys or k in ('head_excl_ie_nms', 'other_in_model')):
            print(f'  {k:18s} {mean[k]:8.1f} ms  {mean[k] / mean["total"] * 100:5.1f}%')
    for k in ('model_total', 'total'):
        v = run_means[k]
        print(f'  {k:18s} {np.mean(v):8.1f} ms  (std over runs {np.std(v):.1f}, runs {[round(x, 1) for x in v]})')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(config=args.config, images=args.images, warmup=args.warmup, repeats=args.repeats,
                   bars=args.bars, verify_t=args.verify_t, reject_t7=args.reject_t7, run_means=run_means,
                   peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, mean_ms=mean,
                   gpu=torch.cuda.get_device_name(0)), open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
