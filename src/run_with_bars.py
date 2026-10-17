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
    args = p.parse_args(sys.argv[1:i])
    bar_list = load_bars(args.bars, args.stat)
    print('IE-NMS bars:', {c: round(b, 4) for c, b in zip(CLASSES[1:], bar_list)}, flush=True)
    print('rules: background_exempt =', args.background_exempt, '| protect_gamma =', args.protect_gamma,
          '| road_alpha =', args.road_alpha, '| verify_t =', args.verify_t, args.m2a_dir, flush=True)
    use_rules = args.background_exempt or args.protect_gamma is not None or args.verify_t is not None
    current = {}

    import mmdet.detseg_utils  # noqa: F401
    from mmengine.structures import InstanceData
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
            vs = box_means(current['img_path'], Bn) if args.verify_t is not None else None
            k = ie_nms_rules(Bn, scores.cpu().numpy(), labels.cpu().numpy(), bar_list,
                             background_exempt=args.background_exempt, protect_gamma=args.protect_gamma,
                             verify_scores=vs, verify_t=args.verify_t)
        else:
            k = ie_nms(bboxes.cpu(), scores.cpu(), labels.cpu(), bar_list)
        keep = torch.as_tensor(k, device=bboxes.device).long()
        r = InstanceData()
        r.bboxes, r.scores, r.labels = bboxes[keep], scores[keep], labels[keep]
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

    sys.argv = ['tools/test.py'] + sys.argv[i + 1:]
    runpy.run_path('tools/test.py', run_name='__main__')


if __name__ == '__main__':
    main()
