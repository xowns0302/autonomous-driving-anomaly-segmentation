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
from sweep_ie_nms_bars import CLASSES, ie_nms  # noqa: E402


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
    args = p.parse_args(sys.argv[1:i])
    bar_list = load_bars(args.bars, args.stat)
    print('IE-NMS bars:', {c: round(b, 4) for c, b in zip(CLASSES[1:], bar_list)}, flush=True)

    import mmdet.detseg_utils  # noqa: F401
    from mmengine.structures import InstanceData
    from mmdet.detseg_utils.models import GroundingDINOHeadWithUniversalObjectness

    def _ie_nms(self, bboxes, scores, labels):
        keep = torch.as_tensor(ie_nms(bboxes.cpu(), scores.cpu(), labels.cpu(), bar_list),
                               device=bboxes.device).long()
        r = InstanceData()
        r.bboxes, r.scores, r.labels = bboxes[keep], scores[keep], labels[keep]
        return r
    GroundingDINOHeadWithUniversalObjectness._ie_nms = _ie_nms

    sys.argv = ['tools/test.py'] + sys.argv[i + 1:]
    runpy.run_path('tools/test.py', run_name='__main__')


if __name__ == '__main__':
    main()
