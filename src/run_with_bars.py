"""Run the official tools/test.py with per-class IE-NMS bars.

Replaces GroundingDINOHeadWithUniversalObjectness._ie_nms with the same logic
(models.py:518-541: boost x2 above the bar, class-agnostic NMS IoU 0.5) but
with bars read from a calibration json (src/calibrate_ie_nms_bars.py).
Everything else (pipeline, DetSeg-S / DetSeg-R, evaluators) is the official
code. With `--bars released` the released bars (0.2 / 0.5) are used, which
must reproduce the unmodified results.

Run from external/DetSeg-official:
  PYTHONPATH=. python ../../src/run_with_bars.py --bars ../../results/A3/bars_things.json --stat bar_min \
      -- ../../configs/detseg-s_fs_lostfound.py ckpts/... --work-dir work_dirs/A3/...
"""
import argparse
import json
import os
import runpy
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_ie_nms_bars import CLASSES, ie_nms, ie_nms_rules  # noqa: E402


def load_bars(path, stat):
    released = [0.2] * 11 + [0.5] * 8
    if path == 'released':
        return released
    cls = json.load(open(path))['classes']
    # Classes without calibration data keep the released bar.
    return [cls[c].get(stat, released[i]) for i, c in enumerate(CLASSES[1:])]


def main():
    if '--' not in sys.argv:
        sys.exit('usage: run_with_bars.py --bars FILE|released [--stat bar_min] -- <tools/test.py args>')
    i = sys.argv.index('--')
    p = argparse.ArgumentParser()
    p.add_argument('--bars', required=True)
    p.add_argument('--stat', default='bar_min', help='bar_min (paper), p1, p5, p10, median')
    p.add_argument('--background-exempt', action='store_true',
                   help='C1: background-class ID boxes cannot suppress OoD boxes')
    p.add_argument('--protect-gamma', type=float, default=None,
                   help='C2: OoD boxes with score >= gamma cannot be suppressed by ID boxes')
    p.add_argument('--road-alpha', type=float, default=None,
                   help='C3: OoD boxes whose bottom-center lies outside every predicted road box get score x alpha')
    p.add_argument('--road-score-thr', type=float, default=0.2,
                   help='C3: ID-branch road boxes with score above this define the road area')
    p.add_argument('--m2a-dir', default=None,
                   help='C5: folder with Mask2Anomaly score maps ({image name}.npy)')
    p.add_argument('--verify-t', type=float, default=None,
                   help='C5: OoD boxes whose mean M2A score >= t cannot be suppressed by ID boxes')
    p.add_argument('--prowl-bank', default=None,
                   help='C6/C7: prototype bank npz (src/prowl_bank.py)')
    p.add_argument('--protect-t6', type=float, default=None,
                   help='C6: OoD boxes with proto_box >= t6 cannot be suppressed by ID boxes')
    p.add_argument('--reject-t7', type=float, default=None,
                   help='C7: kept OoD boxes with proto_box < t7 are dropped')
    p.add_argument('--veto-v', type=float, default=None,
                   help='C9: C7 cannot drop an OoD box whose mean M2A score is >= v (needs --m2a-dir)')
    p.add_argument('--reject-mask-t7', type=float, default=None,
                   help='C10: DetSeg-S, drop SAM masks whose PROWL signal (inside the mask) is < t')
    p.add_argument('--dump-boxes', default=None,
                   help='save the boxes left after IE-NMS (and C7) per image as <dir>/<image name>.npz (L2: VPHM)')
    p.add_argument('--amp', action='store_true', help='L1: FP16 detector (src/speedups.patch_amp); may change results')
    p.add_argument('--fast', action='store_true', help='T3 speed-ups (src/speedups.py), results must not change')
    p.add_argument('--save-masks', default=None, help='DetSeg-S: save each predicted mask as <dir>/<image stem>.png')
    args = p.parse_args(sys.argv[1:i])
    bar_list = load_bars(args.bars, args.stat)
    print('IE-NMS bars:', {c: round(b, 4) for c, b in zip(CLASSES[1:], bar_list)}, flush=True)
    print('rules: background_exempt =', args.background_exempt, '| protect_gamma =', args.protect_gamma,
          '| road_alpha =', args.road_alpha, '| verify_t =', args.verify_t, args.m2a_dir, flush=True)
    print('PROWL: bank =', args.prowl_bank, '| t6 =', args.protect_t6, '| t7 =', args.reject_t7,
          '| veto_v =', args.veto_v, '| mask_t7 =', args.reject_mask_t7, flush=True)
    use_rules = (args.background_exempt or args.protect_gamma is not None or args.verify_t is not None
                 or args.protect_t6 is not None)
    current = {}
    prowl = {}
    if args.prowl_bank:
        import numpy as np
        from prowl_bank import load_model, patch_features
        prowl['model'] = load_model()
        prowl['P'] = torch.from_numpy(np.load(args.prowl_bank)['prototypes']).cuda()

    def proto_feats(img_path):
        """DINOv2 patch features of the image (cached, one pass per image)."""
        import numpy as np
        from PIL import Image
        if prowl.get('img_path') != img_path:          # one DINOv2 pass per image
            if args.fast:
                import speedups
                img = np.array(speedups.load_rgb(img_path))
            else:
                img = np.array(Image.open(img_path).convert('RGB'))
            prowl['feat'], _ = patch_features(prowl['model'], img)
            prowl['HW'], prowl['img_path'] = img.shape[:2], img_path
        return prowl['feat'], prowl['HW']

    def proto_mask(img_path, masks):
        """C10: 1 - max cosine(mean feature of the patches a SAM mask covers by >= half, prototypes).
        If no patch is half covered, the most covered patch is used."""
        import numpy as np
        feat, _ = proto_feats(img_path)
        h, w, D = feat.shape
        P = prowl['P'].reshape(-1, D)
        cov = torch.nn.functional.interpolate((masks > 0).float()[:, None], size=(h, w), mode='area')[:, 0]
        out = np.ones(len(masks), np.float32)      # empty mask -> 1 (never dropped)
        for j in range(len(masks)):
            c = cov[j]
            if c.max() <= 0:
                continue
            sel = c >= 0.5
            if not sel.any():
                sel = c == c.max()
            v = torch.nn.functional.normalize(feat[sel.to(feat.device)].mean(0), dim=0)
            out[j] = 1 - float((P @ v).max())
        return out

    def proto_box(img_path, B):
        """PROWL signal per box: 1 - max cosine(mean DINOv2 patch feature in box, prototypes)."""
        import numpy as np
        feat, (H, W) = proto_feats(img_path)
        if args.fast:
            import speedups
            return speedups.box_proto_signal(feat, B, H, W, prowl['P'])
        h, w, D = feat.shape
        out = np.zeros(len(B), np.float32)
        P = prowl['P'].reshape(-1, D)
        for j, (x1, y1, x2, y2) in enumerate(B):
            x1, x2 = np.clip([x1, x2], 0, W); y1, y2 = np.clip([y1, y2], 0, H)
            gx1, gx2 = int(x1 / W * w), int(np.ceil(x2 / W * w))
            gy1, gy2 = int(y1 / H * h), int(np.ceil(y2 / H * h))
            gx1, gy1 = min(gx1, w - 1), min(gy1, h - 1)
            gx2, gy2 = max(gx2, gx1 + 1), max(gy2, gy1 + 1)
            v = torch.nn.functional.normalize(feat[gy1:gy2, gx1:gx2].reshape(-1, D).mean(0), dim=0)
            out[j] = 1 - float((P @ v).max())
        return out

    import mmdet.detseg_utils  # noqa: F401
    from mmengine.structures import InstanceData
    if args.fast:
        import speedups
        speedups.apply_all()
        print('speed-ups: on', flush=True)
    if args.amp:
        import speedups
        speedups.patch_amp()
        print('FP16 detector: on', flush=True)
    from mmdet.detseg_utils.models import GroundingDINOHeadWithUniversalObjectness

    def box_means(img_path, B):
        """Mean M2A score inside each box (boxes are in original-image pixels)."""
        import numpy as np
        sm = np.load(os.path.join(args.m2a_dir, os.path.basename(img_path) + '.npy')).squeeze()
        ii = np.pad(sm.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)   # integral image
        H, W = sm.shape
        x1 = np.clip(np.floor(B[:, 0]), 0, W).astype(int); x2 = np.clip(np.ceil(B[:, 2]), 0, W).astype(int)
        y1 = np.clip(np.floor(B[:, 1]), 0, H).astype(int); y2 = np.clip(np.ceil(B[:, 3]), 0, H).astype(int)
        area = np.maximum((x2 - x1) * (y2 - y1), 1)
        tot = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]
        return np.where((x2 > x1) & (y2 > y1), tot / area, 0.0)

    def _ie_nms(self, bboxes, scores, labels):
        if use_rules:
            Bn = bboxes.cpu().numpy()
            if args.protect_t6 is not None:
                vs, vt = proto_box(current['img_path'], Bn), args.protect_t6
            elif args.verify_t is not None:
                vs, vt = box_means(current['img_path'], Bn), args.verify_t
            else:
                vs, vt = None, None
            k = ie_nms_rules(Bn, scores.cpu().numpy(), labels.cpu().numpy(), bar_list,
                             background_exempt=args.background_exempt, protect_gamma=args.protect_gamma,
                             verify_scores=vs, verify_t=vt)
        else:
            k = ie_nms(bboxes.cpu(), scores.cpu(), labels.cpu(), bar_list)
        keep = torch.as_tensor(k, device=bboxes.device).long()
        r = InstanceData()
        r.bboxes, r.scores, r.labels = bboxes[keep], scores[keep], labels[keep]
        if args.reject_t7 is not None:
            # C7: drop kept OoD boxes that look like a known class (score 0 -> removed by tau)
            ood = (r.labels == 0).cpu().numpy()
            if ood.any():
                pb = proto_box(current['img_path'], r.bboxes.cpu().numpy())
                d = ood & (pb < args.reject_t7)
                if args.veto_v is not None:          # C9: M2A veto
                    d &= box_means(current['img_path'], r.bboxes.cpu().numpy()) < args.veto_v
                drop = torch.as_tensor(d, device=r.scores.device)
                r.scores = r.scores.clone()
                r.scores[drop] = 0.0
        if args.dump_boxes:
            import numpy as np
            os.makedirs(args.dump_boxes, exist_ok=True)
            np.savez(os.path.join(args.dump_boxes, os.path.basename(current['img_path']) + '.npz'),
                     bboxes=r.bboxes.float().cpu().numpy(), scores=r.scores.float().cpu().numpy(),
                     labels=r.labels.cpu().numpy())
        if args.road_alpha is not None:
            road = bboxes[(labels == CLASSES.index('road')) & (scores > args.road_score_thr)]
            if len(road):
                b = r.bboxes
                fx, fy = (b[:, 0] + b[:, 2]) / 2, b[:, 3] - 1          # bottom-center point
                on = ((road[None, :, 0] <= fx[:, None]) & (fx[:, None] <= road[None, :, 2]) &
                      (road[None, :, 1] <= fy[:, None]) & (fy[:, None] <= road[None, :, 3])).any(1)
                off = (r.labels == 0) & ~on
                r.scores = r.scores.clone()
                r.scores[off] *= args.road_alpha
        return r
    GroundingDINOHeadWithUniversalObjectness._ie_nms = _ie_nms

    # _ie_nms has no image info; record the current image path (batch size 1).
    orig_predict = GroundingDINOHeadWithUniversalObjectness.predict

    def predict(self, *a, batch_data_samples=None, **kw):
        assert len(batch_data_samples) == 1, 'C5 needs batch size 1'
        current['img_path'] = batch_data_samples[0].metainfo['img_path']
        return orig_predict(self, *a, batch_data_samples=batch_data_samples, **kw)
    GroundingDINOHeadWithUniversalObjectness.predict = predict

    if args.reject_mask_t7 is not None:
        from mmdet.detseg_utils.models import DetSegS
        orig_sam = DetSegS.sam_masks_from_boxes

        def sam_masks_c10(self, img_path, boxes, output_shape, device):
            masks = orig_sam(self, img_path, boxes, output_shape, device)
            if len(boxes):
                pm = proto_mask(img_path, masks)
                drop = torch.as_tensor(pm < args.reject_mask_t7, device=masks.device)
                if drop.any():
                    masks = masks.clone()
                    masks[drop] = 0
            return masks
        DetSegS.sam_masks_from_boxes = sam_masks_c10

    if args.save_masks:
        import numpy as np
        from PIL import Image
        from mmdet.detseg_utils.models import DetSegS
        os.makedirs(args.save_masks, exist_ok=True)
        orig_s_predict = DetSegS.predict

        def s_predict(self, *a, **k):
            out = orig_s_predict(self, *a, **k)
            for ds in out:
                m = ds.pred_sem_seg.sem_seg.squeeze().cpu().numpy() > 0.5
                stem = os.path.splitext(os.path.basename(ds.metainfo['img_path']))[0]
                Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(args.save_masks, stem + '.png'))
            return out
        DetSegS.predict = s_predict

    sys.argv = ['tools/test.py'] + sys.argv[i + 1:]
    runpy.run_path('tools/test.py', run_name='__main__')


if __name__ == '__main__':
    main()
